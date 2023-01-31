from collections import defaultdict
from model.model import GraphLevelEncoder, MLP, MyTransformerDecoder, UnifyPoolingGraphLevelEncoder, CrossAttentionTransformer
from model.NodeEmbedding import NodeAsinEmbedding, NodeTextTransformer, PretrainedQAEAEncoder
from model.gnn import get_hetero_GNN, GraphPooling, HGT, HeteroGGNN, AttentionPooling, SRGNN_Pooling, PositionalAttentionPooling
from torch_geometric.loader import DataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import numpy as np
from transformers import AutoTokenizer, AutoModel
from config import CFG
import pickle
from train_subsession_embedding import to_subsession, get_next_product_asin_loss, get_next_query_loss, get_next_product_asin_accuracy, get_edge_index_max
from train_session_embedding import get_target, get_all_product_asin_loss, get_last_query_loss, get_all_product_asin_accuracy
#from train_subsession_embedding import get_next_query_mlm_loss, get_next_query_electra_loss
#from train_session_embedding import get_last_query_electra_loss, get_last_query_mlm_loss
import logging
from tqdm import tqdm
import torch.nn.functional as F
import os
import shutil
from torch_geometric.data import InMemoryDataset
import torch.nn as nn
#import psutil
from util_amazon_filtered import sequence_to_graph, session_to_text
from torch_geometric.nn import global_mean_pool
import random
import time
os.environ['CUDA_VISIBLE_DEVICES']='5'

def randomly_mask_tokens(data, mask_ratio, tokenizer):
    # mask tokens in product title
    cur_device = data['product'].input_ids.device
    mask = (torch.rand(data['product'].input_ids.shape).to(cur_device)<mask_ratio) & (data['product'].input_ids >= 5)
    data['product'].gt_input_ids = data['product'].input_ids
    data['product'].input_ids[mask] = tokenizer.mask_token_id
    data['product'].token_mask = mask

    # mask tokens in query
    mask = (torch.rand(data['query'].input_ids.shape).to(cur_device)<mask_ratio) & (data['query'].input_ids >= 5)
    data['query'].gt_input_ids = data['query'].input_ids
    data['query'].input_ids[mask] = tokenizer.mask_token_id
    data['query'].token_mask = mask

    return data

def sample_predict_word(data, node_type, logits):
    # logits shape is batch * seq_len * vocab_size
    dist = torch.distributions.categorical.Categorical(logits=logits)
    sample = dist.sample().to(data[node_type].token_mask.device)
    mask = data[node_type].token_mask

    data[node_type].input_ids[mask] = sample[mask]
    return data

def MLMLoss(data, node_type, logits):
    # node_type is either 'query' or 'product'
    # logits shape is batch * seq_len * vocab_size
    label = data[node_type].gt_input_ids
    mask = data[node_type].token_mask # batch * seq_len
    loss = F.cross_entropy(logits[mask], label[mask])
    return loss

def ElectraLoss(data, node_type, pred):
    # node_type is either 'query' or 'product'
    # logits shape is batch * seq_len, value from 0 to 1
    label = data[node_type].input_ids != data[node_type].gt_input_ids
    loss = F.binary_cross_entropy(pred, label.float())
    return loss



def ContrastiveLoss(view1_rep, view2_rep): 
    # the same row in view 1 and view 2 are positive samples, the rest are negative samples
    try:
        assert (view1_rep.shape == view2_rep.shape)
    except Exception as e:
        print(e)
        print(view1_rep.shape)
        print(view2_rep.shape)
        raise RuntimeError("size run")
    #score = torch.sigmoid(torch.matmul(view1_rep, view2_rep.T))
    normalized_view1 = view1_rep / (torch.sqrt(torch.clip(torch.sum(view1_rep**2,dim=1).view(-1,1), min=1e-6)))
    normalized_view2 = view2_rep / (torch.sqrt(torch.clip(torch.sum(view2_rep**2,dim=1).view(-1,1), min=1e-6)))
    score = normalized_view1 @ normalized_view2.T
    score = torch.clip(score, min=1e-4, max=0.9999)
    JS_est = torch.log(1-score)
    JS_est.fill_diagonal_(0)
    pos_est = torch.diag(torch.diag(torch.log(score))) * 10
    JS_est = JS_est + pos_est
    return -torch.sum(JS_est)/(view1_rep.size(0)*view1_rep.size(0)+9*view1_rep.size(0))

"""
def random_drop_node(data, asin_num):
    seq = data['ori_product'].x.detach().cpu().tolist()
    if len(seq) == 1:
        return data
    tar = data['product_target'].y.item()
    i = np.random.randint(len(seq))
    del seq[i]
    return sequence_to_graph(seq, tar, asin_num
"""
def random_exchange_order(data, tokenizer, query_max_len):
    #seq = data['ori_product'].x.detach().cpu().tolist()
    #tar = data['product_target'].y.item()
    seq, tar = data['ori_seq']
    i = np.random.randint(len(seq))
    j = np.random.randint(len(seq))
    times = 1
    while j == i:
        j = np.random.randint(len(seq))
        times += 1
        if times == 10:
            break
    tmp = seq[i]
    seq[i] = seq[j]
    seq[j] = tmp
    return sequence_to_graph(0,seq, tar, tokenizer, query_max_len, CFG.ignore_query)

"""
def random_perturbe_node(data, asin_num):
    seq = data['ori_product'].x.detach().cpu().tolist()
    if len(seq) == 1:
        return data
    tar = data['product_target'].y.item()
    i = np.random.randint(len(seq))
    seq[i] = np.random.randint(asin_num)
    return sequence_to_graph(seq, tar, asin_num, CFG.ignore_query)

def random_mask_node(data, asin_num):
    seq = data['ori_product'].x.detach().cpu().tolist()
    if len(seq) == 1:
        return data
    tar = data['product_target'].y.item()
    i = np.random.randint(len(seq))
    seq[i] = 0 # 0 means masked 
    return sequence_to_graph(seq, tar, asin_num)
"""



class FilteredAmazonDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data_list = data_list
        self.data, self.slices = self.collate(self.data_list)

def next_text_embedding_loss(graph_embedding, data, header, q_or_p, target_query_embedder, device):
    if header is not None:
        rep = header(graph_embedding)
    else:
        rep = graph_embedding
    
    target_type = q_or_p + "_target"
    target = target_query_embedder(data[target_type].input_ids, data[target_type].token_type_ids, data[target_type].attention_mask)
    # rep is n * d, target is n * d
    val = torch.sigmoid(rep @ target.T)
    #val = F.normalize(rep,dim=1)@F.normalize(target,dim=1).T
    
    y = torch.diag(data[target_type].mask).to(device)
    loss_mat = -(y*torch.log(val)+(1-y)*torch.log(1-val))
    return torch.mean(loss_mat)
    

def all_text_embedding_loss(graph_embedding, data, header, q_or_p, target_query_embedder, device):
    if header is not None:
        rep = header(graph_embedding)
    else:
        rep = graph_embedding
    
    target_type = q_or_p
    target = target_query_embedder(data[target_type].input_ids, data[target_type].token_type_ids, data[target_type].attention_mask)
    # rep is n * d, target is nq * d
    val = torch.sigmoid(rep @ target.T)
    #val = F.normalize(rep, dim=1) @ F.normalize(target, dim=1).T
    #print(val.isnan().any())
    y = torch.zeros_like(val)
    line_mask = torch.arange(val.size(0)).repeat(val.size(1),1).T
    batch_mask = data[target_type].batch.repeat(val.size(0),1)
    mask = (line_mask == batch_mask.cpu())
    y[mask] = 1
    y = y.to(device)
    #print(y.isnan().any())
    loss_mask = torch.ones_like(val)
    loss_mask[mask] = data[target_type].mask
    loss_mask = loss_mask.bool().to(device)
#    y = torch.diag(data['query_target'].mask).to(device)
    loss_mat = -(y*torch.log(val)+(1-y)*torch.log(1-val))
    
    return torch.mean(loss_mat[loss_mask])

def main():
    if not os.path.exists(CFG.savedir):
        os.makedirs(CFG.savedir)
    shutil.copy(__file__, CFG.savedir)
    shutil.copy("./config.py", CFG.savedir)
    logging.basicConfig(filename =  CFG.savedir+"/pretrain.log",
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    asin_num = 391572
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)
    np.set_printoptions(precision=3,linewidth=100)
    tokenizer = AutoTokenizer.from_pretrained('SavedModel/QAEA')
    #print(tokenizer.model_max_length)
    #assert(tokenizer.model_max_length==20)
    
    

    load_start = time.perf_counter()
    train_data = pickle.load(open('/local2/data-1/raw/us-filtered-split-train-data.pkl', 'rb'))
    #train_data = [(a,b) for a,b in zip(train_data[0], train_data[1])]
    #train_data = random.sample(train_data, 100000)
    short_train_data = [train_data[0][:3000000], train_data[1][:3000000]]
    #del train_data
    train_data = short_train_data
    #print("converting...")
    data_list = [sequence_to_graph(i, train_data[0][i], train_data[1][i],tokenizer, CFG.query_max_len, CFG.ignore_query) for i in tqdm(range(len(train_data[0])))]
    #data_list = pickle.load(open("data/filtered_amazon_data_list_2M.pkl", "rb"))
    print(time.perf_counter()-load_start)
    #print(len(data_list))
    #print('dump')
    #print(len(data_list))
    
    #data_list = [expand(data) for data in tqdm(data_list)]
    
    #pickle.dump(data_list, open("data/filtered_amazon_data_list_2M.pkl", "wb"), protocol=4)
    #data_list = data_list[:1000000]
    #print(len(data_list))
    #print(process.memory_info().rss)
    #del train_data
    #train_list, valid_list = torch.utils.data.random_split(data_list, [len(data_list)-50000, 50000])
    random.shuffle(data_list)
    if len(data_list) == 10000:
        valid_num = 5000
    else:
        valid_num = 50000
    train_list, valid_list = data_list[:-valid_num], data_list[-valid_num:]
    print('split')
    del data_list
    #train_list = valid_list = data_list
    dataset = FilteredAmazonDataset(root="data",
                        data_list = train_list,
                        #transform=lambda x: (x, random_exchange_order(x, tokenizer, CFG.query_max_len)))
                        transform=lambda x: (x, x))
    print(len(dataset))
    valid_set = FilteredAmazonDataset(root="data",
                        data_list = valid_list,
                        #transform=lambda x: (x, random_exchange_order(x, tokenizer, CFG.query_max_len)))
                        transform=lambda x: (x, x))
    #print(process.memory_info().rss)
    #test_data = pickle.load(open('/home/ec2-user/SR-GNN/datasets/yoochoose1_64/test.txt', 'rb'))
    #test_list = [sequence_to_graph(seq,tar,asin_num) for seq,tar in zip(test_data[0],test_data[1])]
    #test_set = YoochooseDataset(root="data", data_list=test_list, transform=lambda x: (x,random_mask_node(x, asin_num), random_exchange_order(x, asin_num)))
    #test_loader = DataLoader(test_set, batch_size=CFG.batch_size, shuffle=True)
    #train_set = valid_set = dataset
    train_loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=CFG.batch_size, shuffle=False)
    #print(process.memory_info().rss)
    
    target_asin_embedding = torch.nn.Embedding(asin_num, CFG.emb_len)
    target_text_embedder = PretrainedQAEAEncoder(None)
    #query_node_embedder = PretrainedQAEAEncoder(CFG.emb_len)
    query_node_embedder = PretrainedQAEAEncoder(None)
    product_node_embedder = NodeAsinEmbedding(nproducts=asin_num, ninp=CFG.emb_len)
    gnn = HeteroGGNN(CFG.gnn_nout, CFG.gnn_nlayers, dataset[0][0])
    if False:
        node_out_dim = CFG.gnn_nlayers*CFG.gnn_nout+768
        product_pooling = SRGNN_Pooling(CFG.gnn_nlayers*CFG.gnn_nout+768, CFG.gnn_nout)
        #product_pooling = AttentionPooling(CFG.gnn_nlayers*CFG.gnn_nout+768, CFG.gnn_nout)
        #query_pooling2 = GraphPooling('mean', CFG.gnn_nout*(CFG.gnn_nlayers+1), CFG.gnn_nout, CFG.gnn_dropout)
        #query_pooling = SRGNN_Pooling((CFG.gnn_nlayers+1)*CFG.gnn_nout, CFG.gnn_nout)
        query_pooling = AttentionPooling(CFG.gnn_nlayers*CFG.gnn_nout+768, CFG.gnn_nout)
        subsession_encoder = GraphLevelEncoder(query_node_embedder=query_node_embedder,
                        product_node_embedder=product_node_embedder,
                        gnn=gnn, product_pooling=product_pooling, query_pooling=query_pooling, use_id_embedding=False)
    else:
        node_out_dim = CFG.gnn_nlayers*CFG.gnn_nout+768
        #node_out_dim = CFG.gnn_nlayers*CFG.gnn_nout+768
        pooling = PositionalAttentionPooling(node_out_dim, node_out_dim, CFG.gnn_nout*2, CFG.max_seq_len)
        cross_attention_transformer = CrossAttentionTransformer(3, 2, CFG.gnn_nlayers*CFG.gnn_nout+768, 768, 8, 1200, 0)
        subsession_encoder = UnifyPoolingGraphLevelEncoder(query_node_embedder=query_node_embedder,
                        product_node_embedder=product_node_embedder,
                        gnn=gnn, pooling=pooling, 
                        cross_attention_transformer=cross_attention_transformer,
                        use_id_embedding=False)
    #gnn_out = (CFG.gnn_nlayers+1)*CFG.gnn_nout*2
    gnn_out = CFG.gnn_nout * 2
    next_product_head = MLP(gnn_out, CFG.emb_len, CFG.ph_nhid, CFG.ph_nlayers, dropout=CFG.ph_dropout)
    all_product_head = MLP(gnn_out, CFG.emb_len, CFG.ph_nhid, CFG.ph_nlayers, dropout=CFG.ph_dropout)
    next_query_head = MLP(gnn_out, 768, CFG.qh_nhid, CFG.qh_nlayers, dropout=CFG.qh_dropout)
    all_query_head = MLP(gnn_out, 768, CFG.qh_nhid, CFG.qh_nlayers, dropout=CFG.qh_dropout)
    next_title_head = MLP(gnn_out, 768, 768, 2, dropout=CFG.qh_dropout)
    all_title_head = MLP(gnn_out, 768, 768, 2, dropout=CFG.qh_dropout)
    qaea_head = MLP(gnn_out, 768, 2000,2,dropout=0)
    query_node_head = MLP(node_out_dim, 768, 768, 2,0)
    product_node_head = MLP(node_out_dim, 768, 768, 2,0)
    token_electra_head = nn.Linear(768, 1)
    #subsession_encoder, next_product_head, all_product_head = torch.load("SavedModel/Yoochoose-next-all-HGGNN-SrGNNPooling/pretrain_model_cont.pth")
    #session_encoder, all_product_head, session_sim_head = torch.load("SavedModel/Yoochoose-HGGNN-SrGNNPooling-1/session_model_cont_2.pth")
    #target_asin_embedding = torch.load("SavedModel/Yoochoose-next-all-HGGNN-SrGNNPooling/target_embedding.pth")
    #subsession_encoder, next_product_head, all_product_head, next_query_head, cur_query_head, next_title_head, cur_title_head = torch.load(CFG.savedir+"pretrain_model.pth")
    #target_asin_embedding, target_text_embedder = torch.load(CFG.savedir+"target_embedding.pth")
    
    subsession_encoder.to(device)
    next_product_head.to(device)
    #session_encoder.to(device)
    all_product_head.to(device)
    target_asin_embedding.to(device)
    target_text_embedder.to(device)
    next_query_head.to(device)
    all_query_head.to(device)
    next_title_head.to(device)
    all_title_head.to(device)
    qaea_head.to(device)
    query_node_head.to(device)
    product_node_head.to(device)
    token_electra_head.to(device)
    #session_sim_head.to(device)
    
      
    best_valid_loss = 1000
    #params = list(session_encoder.parameters()) + list(all_product_head.parameters()) + list(session_sim_head.parameters())
    
    #all_params = params
    #optimizer = torch.optim.Adam(params, lr=CFG.lr)
    optimizer2 = torch.optim.Adam(list(target_asin_embedding.parameters()), lr=CFG.lr)#+list(target_text_embedder.parameters()), lr=CFG.lr)
    all_params = list(target_asin_embedding.parameters())#+list(target_text_embedder.parameters())
    
    params = list(subsession_encoder.parameters())
    params += list(next_product_head.parameters())
    params += list(next_query_head.parameters())
    params += list(all_product_head.parameters())
    params += list(all_query_head.parameters())
    params += list(all_title_head.parameters())
    params += list(next_title_head.parameters())
    params += list(qaea_head.parameters())
    params += list(query_node_head.parameters())
    params += list(product_node_head.parameters())
    params += list(token_electra_head.parameters())
    all_params += params
    optimizer3 = torch.optim.Adam(params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    torch.autograd.set_detect_anomaly(True)
    max_iter = len(train_loader) // 10

    qaea = AutoModel.from_pretrained("SavedModel/QAEA", add_pooling_layer=False)
    qaea = qaea.to(device)
    qaea_MLM_head = nn.Linear(768, tokenizer.vocab_size)
    qaea_MLM_head.to(device)
    qaea_optimizer = torch.optim.Adam(list(qaea.parameters())+list(qaea_MLM_head.parameters()), lr=CFG.lr)

    for epoch in range(CFG.max_epoch):
        print(CFG.savedir)
        # training
        training_loss = []
        mlm_loss = []
        train_next_precision = []
        train_next_recall = []
        train_all_precision = []
        train_all_recall = []
                
        subsession_encoder.train()
        next_product_head.train()
        target_asin_embedding.train()
        #session_encoder.train()
        all_product_head.train()
        target_text_embedder.train()
        next_query_head.train()
        all_query_head.train()
        next_title_head.train()
        all_title_head.train()
        qaea_head.train()
        query_node_head.train()
        product_node_head.train()
        #session_sim_head.train()
       
        #with torch.no_grad():
        #torch.autograd.set_detect_anomaly(True)
        for i, data in enumerate(tqdm(train_loader)):
            if i == max_iter:
                break
            if i % 2000 == 1:
                print(np.mean(training_loss,axis=0), np.mean(mlm_loss))
                #print(np.mean(training_loss), np.mean(train_next_recall), np.mean(train_all_recall))
        #for i, data in enumerate(train_loader):
            #subsession = next(iter(subsession_loader))
            subsession = data[0].to(device)
            #session = subsession
            if data[1] is not None:
                view1 = data[1].to(device)

            #optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            #print(1)
            """Preprocess"""
            #subsession = randomly_mask_tokens(subsession, CFG.mask_token_ratio, tokenizer)
            # use qaea to generate predicted words
            # product
            #qaea_optimizer.zero_grad()
            #product_logits = torch.sigmoid(qaea_MLM_head(qaea(input_ids=subsession['product'].input_ids, 
            #                    token_type_ids = subsession['product'].token_type_ids, 
            #                    attention_mask=subsession['product'].attention_mask).last_hidden_state))
            #query_logits = torch.sigmoid(qaea_MLM_head(qaea(input_ids=subsession['query'].input_ids, 
            #                    token_type_ids = subsession['query'].token_type_ids, 
            #                    attention_mask=subsession['query'].attention_mask).last_hidden_state))
            #loss = MLMLoss(subsession, 'product', product_logits) + MLMLoss(subsession, 'query', query_logits)
            #loss.backward()
            #mlm_loss.append(loss.item())
            #TODO
            #qaea_optimizer.step()
            #subsession = sample_predict_word(subsession, 'product', product_logits)
            #subsession = sample_predict_word(subsession, 'query', query_logits)
            

            try:
                query_node_mask = (torch.rand(subsession['query'].num_nodes)>CFG.node_mask_prob).float().to(device)
                product_node_mask = (torch.rand(subsession['product'].num_nodes)>CFG.node_mask_prob).float().to(device)
                subsession_embedding, node_embedding, token_embedding = subsession_encoder(subsession, query_node_mask, product_node_mask, True, True)
                #subsession_embedding, node_embedding, token_embedding = subsession_encoder(subsession, None, None, True, True)
                if CFG.token_w > 0:
                    
                    query_token_pred = torch.sigmoid(token_electra_head(token_embedding['query'])).squeeze()
                    product_token_pred = torch.sigmoid(token_electra_head(token_embedding['product'])).squeeze()
                    token_loss = ElectraLoss(subsession, 'product', product_token_pred) + ElectraLoss(subsession, 'query', query_token_pred)
                else:
                    token_loss = 0 * torch.mean(query_node_mask)


                query_node_pred = query_node_head(node_embedding['query'])
                query_node_feat = target_text_embedder(subsession['query'].x, subsession['query'].token_type_ids, subsession['query'].attention_mask)

                query_node_loss = torch.sum((1-query_node_mask)*((1-F.cosine_similarity(query_node_pred, query_node_feat))**2))/(torch.sum(1-query_node_mask)+1e-3)

                product_node_pred = product_node_head(node_embedding['product'])
                product_node_feat = target_text_embedder(subsession['product'].input_ids, subsession['product'].token_type_ids, subsession['product'].attention_mask)
                product_node_loss = torch.sum((1-product_node_mask)*((1-F.cosine_similarity(product_node_pred, product_node_feat))**2))/(torch.sum(1-product_node_mask)+1e-3)
                #print(subsession_embedding.shape)
                #next_product_loss = get_next_product_asin_loss(subsession_embedding, next_product_head, subsession, target_asin_embedding, device=device)
                next_product_loss = get_next_product_asin_loss(subsession_embedding, next_product_head, subsession, target_asin_embedding, device=device)
                all_product_loss = get_all_product_asin_loss(subsession_embedding, all_product_head, subsession, target_asin_embedding, device)
                next_query_loss = all_text_embedding_loss(subsession_embedding, subsession, next_query_head, 'query', target_text_embedder, device=device)
                cur_query_loss = all_text_embedding_loss(subsession_embedding, subsession, all_query_head, 'query', target_text_embedder, device=device)
                next_title_loss = all_text_embedding_loss(subsession_embedding, subsession, next_title_head, 'product_target', target_text_embedder, device=device)
                cur_title_loss = all_text_embedding_loss(subsession_embedding, subsession, all_title_head, 'product', target_text_embedder, device=device)
                #print(next_product_loss)
                
                qaea_pred = qaea_head(subsession_embedding)
                qaea_label = qaea(input_ids=subsession['text'].input_ids, 
                                token_type_ids = subsession['text'].token_type_ids, 
                                attention_mask=subsession['text'].attention_mask).last_hidden_state.detach()
                #print(subsession['text'].input_ids.shape)
                #print(qaea_label.shape)
                #print(qaea_pred.shape)
                qaea_label = torch.mean(qaea_label,dim=1)
                qaea_label = global_mean_pool(qaea_label, subsession['text'].batch)
                qaea_loss = torch.mean(1 - F.cosine_similarity(qaea_label,qaea_pred))
                
                if data[1] is not None:
                    view1_emb = subsession_encoder(view1)
                    #view2_emb = subsession_encoder(view2)
                    ctv_loss = ContrastiveLoss(subsession_embedding, view1_emb)

            except Exception as e:
                print(e)
                raise RuntimeError("bug")
                   
            
            #loss = last_query_loss
            #print(next_product_loss.item(), cur_query_loss.item())
            #loss = CFG.ph_w * all_product_loss + CFG.qh_w * last_query_loss + CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
            loss = next_product_loss
            #print(loss, next_product_loss)
            """
            loss += CFG.ph_w * next_product_loss + 2 * CFG.ph_w * all_product_loss
            #print(loss)

            loss += CFG.qh_w * next_query_loss + CFG.qh_w * cur_query_loss
            #print(loss)
            loss += CFG.pt_w * next_title_loss + CFG.pt_w * cur_title_loss
            #print(loss)
            loss += CFG.ctv_w * ctv_loss
            loss += CFG.qaea_w * qaea_loss
            #print(loss)
            loss += CFG.node_w * query_node_loss + CFG.node_w * product_node_loss
            loss += CFG.token_w * token_loss
            loss = next_product_loss
            print(loss)
            """
            #loss += CFG.ctv_w * contrastive_loss
            try:
                assert(torch.sum(loss.isnan())==0)
            except:
                print(next_product_loss.item(), all_product_loss.item(), next_query_loss.item(), cur_query_loss.item(), next_title_loss.item(),
                                    cur_title_loss.item(), ctv_loss.item(), query_node_loss.item(), product_node_loss.item(), qaea_loss.item(), token_loss.item())
                raise RuntimeError("Nan in Loss")
            """
            training_loss.append((next_product_loss.item(), all_product_loss.item(), next_query_loss.item(), cur_query_loss.item(), next_title_loss.item(),
                                    cur_title_loss.item(), ctv_loss.item(), query_node_loss.item(), product_node_loss.item(), qaea_loss.item(), token_loss.item()))
            """
            training_loss.append((next_product_loss.item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.)
            #optimizer.step()
            optimizer2.step()
            optimizer3.step()
            """
            with torch.no_grad():
                precision, recall = get_next_product_asin_accuracy(subsession_embedding, next_product_head, subsession, target_asin_embedding, 20)
                train_next_precision.append(precision)
                train_next_recall.append(recall)

                precision, recall = get_all_product_asin_accuracy(subsession_embedding, all_product_head, subsession, target_asin_embedding, 20)
                train_all_precision.append(precision)
                train_all_recall.append(recall)
            """ 
            if i == len(train_loader)-1 or i == max_iter - 1:#i % CFG.ckpt_iter == 0:
        #if epoch % 10 == 0:
                valid_loss = []
                
                subsession_encoder.eval()
                next_product_head.eval()
                target_asin_embedding.eval()
                target_text_embedder.eval()
                all_product_head.eval()
                next_query_head.eval()
                all_query_head.eval()
                next_title_head.eval()
                all_title_head.eval()
                query_node_head.eval()
                product_node_head.eval()
                
                with torch.no_grad():
                    
                    for data in valid_loader:
                        
                        subsession = data[0]
                        subsession = subsession.to(device)
                        view1 = data[1].to(device)
                        view1_emb = subsession_encoder(view1)

                        """Preprocess"""
                        
                        #subsession = randomly_mask_tokens(subsession, CFG.mask_token_ratio, tokenizer)
                        # use qaea to generate predicted words
                        #product_logits = torch.sigmoid(qaea_MLM_head(qaea(input_ids=subsession['product'].input_ids, 
                        #        token_type_ids = subsession['product'].token_type_ids, 
                        #        attention_mask=subsession['product'].attention_mask).last_hidden_state))
                        #query_logits = torch.sigmoid(qaea_MLM_head(qaea(input_ids=subsession['query'].input_ids, 
                        #        token_type_ids = subsession['query'].token_type_ids, 
                        #        attention_mask=subsession['query'].attention_mask).last_hidden_state))
                        #subsession = sample_predict_word(subsession, 'product', product_logits)
                        #subsession = sample_predict_word(subsession, 'query', query_logits)
                        
                        query_node_mask = (torch.rand(subsession['query'].num_nodes)>CFG.node_mask_prob).float().to(device)
                        product_node_mask = (torch.rand(subsession['product'].num_nodes)>CFG.node_mask_prob).float().to(device)
                        emb, node_embedding, token_embedding = subsession_encoder(subsession, query_node_mask, product_node_mask, True, True)
                        #emb, node_embedding, token_embedding = subsession_encoder(subsession, None, None, True, True)
                        if CFG.token_w > 0:
                            query_token_pred = torch.sigmoid(token_electra_head(token_embedding['query'])).squeeze()
                            product_token_pred = torch.sigmoid(token_electra_head(token_embedding['product'])).squeeze()
                            token_loss = ElectraLoss(subsession, 'product', product_token_pred) + ElectraLoss(subsession, 'query', query_token_pred)
                        else:
                            token_loss = 0*torch.mean(emb)    
                        query_node_pred = query_node_head(node_embedding['query'])
                        query_node_feat = target_text_embedder(subsession['query'].x, subsession['query'].token_type_ids, subsession['query'].attention_mask)
                        query_node_loss = torch.sum((1-query_node_mask)*((1-F.cosine_similarity(query_node_pred, query_node_feat))**2))/(torch.sum(1-query_node_mask)+1e-3)

                        product_node_pred = product_node_head(node_embedding['product'])
                        product_node_feat = target_text_embedder(subsession['product'].input_ids, subsession['product'].token_type_ids, subsession['product'].attention_mask)
                        product_node_loss = torch.sum((1-product_node_mask)*((1-F.cosine_similarity(product_node_pred, product_node_feat))**2))/(torch.sum(1-product_node_mask)+1e-3)
                
                        
                        #emb = subsession_encoder(subsession)
                        ctv_loss = ContrastiveLoss(emb, view1_emb)
                        

                        
                        next_ploss = get_next_product_asin_loss(emb, next_product_head, subsession, target_asin_embedding, device=device)
                        all_ploss = get_all_product_asin_loss(emb, all_product_head, subsession, target_asin_embedding, device=device)
                        next_qloss = all_text_embedding_loss(emb, subsession, next_query_head, 'query', target_text_embedder, device=device)
                        cur_qloss = all_text_embedding_loss(emb, subsession, all_query_head, 'query', target_text_embedder, device=device)
                        next_title_loss = all_text_embedding_loss(subsession_embedding, subsession, next_title_head, 'product_target', target_text_embedder, device=device)
                        cur_title_loss = all_text_embedding_loss(subsession_embedding, subsession, all_title_head, 'product', target_text_embedder, device=device)
                        
                        qaea_label = qaea(input_ids=subsession['text'].input_ids, 
                                token_type_ids = subsession['text'].token_type_ids, 
                                attention_mask=subsession['text'].attention_mask).last_hidden_state.detach()
                        qaea_label = torch.mean(qaea_label,dim=1)
                        qaea_label = global_mean_pool(qaea_label, subsession['text'].batch)
                        
                        qaea_pred = qaea_head(subsession_embedding)
                        qaea_loss = torch.mean(1 - F.cosine_similarity(qaea_label,qaea_pred))
                        valid_loss.append([next_ploss.item()])
                        """
                        valid_loss.append([ctv_loss.item()*CFG.ctv_w, next_ploss.item()*CFG.ph_w, all_ploss.item()*CFG.ph_w, next_qloss.item()*CFG.qh_w, cur_qloss.item()*CFG.qh_w,
                        next_title_loss.item()*CFG.pt_w, cur_title_loss.item()*CFG.pt_w,
                        query_node_loss.item()*CFG.node_w, product_node_loss.item()*CFG.node_w,
                        qaea_loss.item()*CFG.qaea_w,
                        token_loss.item()*CFG.token_w])
                        """
                ave_valid_loss = np.mean(valid_loss)
                print(np.mean(valid_loss,axis=0))
                if ave_valid_loss < best_valid_loss:
                    torch.save((subsession_encoder, next_product_head, all_product_head, next_query_head, all_query_head, 
                        next_title_head, all_title_head, query_node_head, product_node_head, qaea_head), CFG.savedir+"pretrain_model.pth")
                    #torch.save((session_encoder, all_product_head, session_sim_head), CFG.savedir+"session_model_cont_3.pth")
                    torch.save((target_asin_embedding, target_text_embedder), CFG.savedir+"target_embedding.pth")
                    best_valid_loss = ave_valid_loss
               
                # validation
                logging.info("Epoch %d, Iter %d, average training loss: %.3f, average valid loss %.3f"%(epoch, i, np.mean(training_loss), ave_valid_loss))
                logging.info("Detaild Valid Loss"+np.array2string(np.mean(valid_loss,axis=0)))
            
                
                subsession_encoder.train()
                next_product_head.train()
                target_asin_embedding.train()
                target_text_embedder.train()
                next_query_head.train()
                #session_encoder.train()
                all_product_head.train()
                all_query_head.train()
                #next_query_head.train()
                next_title_head.train()
                all_title_head.train()
                query_node_head.train()
                product_node_head.train()
                    
                
                #print(precision, recall)
                training_loss = []
                train_next_precision = []
                train_next_recall = []
                train_all_precision = []
                train_all_recall = []
        

#    all_session_loader = DataLoader(session_dataset, batch_size=CFG.batch_size, shuffle=False)
#    with torch.no_grad():
##        for data in tqdm(all_session_loader):
 #           data = data.to(device)
 #           embedding = 

if __name__ == '__main__':
    main()
