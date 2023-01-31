from Dataset import ProductAsinSessionTestDataset, ProductAsinSessionTrainDataset
from model.model import GraphLevelEncoder, MLP, MyTransformerDecoder
from model.NodeEmbedding import NodeAsinEmbedding, NodeTextTransformer
from model.gnn import get_hetero_GNN, GraphPooling, HGT
from Dataset import pretransform_QueryTokenProductAsin, SmallProductAsinSessionTrainDataset
#from torch_geometric.loader import DataLoader
from DataLoader import MyDataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import numpy as np
from transformers import BertTokenizer
from config import CFG
import pickle
from train_subsession_embedding import to_subsession, get_next_product_asin_loss, get_next_query_loss, get_next_product_asin_accuracy, get_edge_index_max
from train_session_embedding import get_target, get_all_product_asin_loss, get_last_query_loss, get_all_product_asin_accuracy
from train_subsession_embedding import get_next_query_mlm_loss, get_next_query_electra_loss
from train_session_embedding import get_last_query_electra_loss, get_last_query_mlm_loss
import logging
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='2'

def get_asin_onehot(dataset, asin_num):
    one_hot = torch.zeros(len(asin_num))
    for data in dataset:
        one_hot[data['product'].x] = 1
    return one_hot

def overlap_with_onehot(data, one_hot):
    x = torch.zeros_like(one_hot)
    x[data['product'].x] = 1
    overlap = torch.sum(x * one_hot)
    return overlap > 0

def iterate_filter_data(target_set, original_set, asin_num, iter_num):
    onehot = get_asin_onehot(target_set, asin_num)
    #new_target_set = [data for data in target_set]
    for iter in range(iter_num):
        new_added_dataset = [data for data in original_set if overlap_with_onehot(data, onehot)]
        new_target_set = [data for data in target_set] + new_added_dataset
        onehot = get_asin_onehot(new_target_set)
    return new_added_dataset


    
def ContrastiveLoss(session_rep, subsession_rep):
    try:
        assert (session_rep.shape == subsession_rep.shape)
    except Exception as e:
        print(e)
        print(session_rep.shape)
        print(subsession_rep.shape)
        raise RuntimeError("size run")
    score = torch.sigmoid(torch.matmul(session_rep, subsession_rep.T))
    score = torch.clip(score, min=1e-4, max=0.9999)
    JS_est = torch.log(1-score)
    JS_est.fill_diagonal_(0)
    pos_est = torch.diag(torch.diag(torch.log(score)))
    JS_est = JS_est + pos_est
    return torch.mean(JS_est)

def main():
    #dataset = SmallProductAsinSessionTrainDataset(root="data", 
    #    transform=None,
    #    pre_filter = lambda data : (data['product'].num_nodes + data['query'].num_nodes)>2,
    #    pre_transform=pretransform_QueryTokenProductAsin)
    #for i, data in enumerate(dataset):
    #    if i % 10000 == 0:
    #        print(i)
    #    edge_index = torch.cat((data['query', 'clicks', 'product'].edge_index,
    #                        data['query', 'adds', 'product'].edge_index,
    #                        data['query', 'purchases', 'product'].edge_index),1)
    #    if edge_index[1,:].max() >= data['product'].x.size(0):
    #        print(edge_index)
    #        print(data)
    #        print(i)
    #        raise RuntimeError("illegal data")
    #print("pass it")
    logging.basicConfig(filename =  CFG.log_file,
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #print(tokenizer.model_max_length)
    #assert(tokenizer.model_max_length==20)
    with open("data/us-20220401-asin2id-brand2id-small.pkl", "rb") as f:
        asin2id, _ = pickle.load(f)

    """
    session_dataset = SmallProductAsinSessionTrainDataset(root="data", 
        transform=lambda x: get_target(x, tokenizer),
        pre_filter = lambda data : (data['product'].num_nodes + data['query'].num_nodes)>2,
        pre_transform=pretransform_QueryTokenProductAsin)
    train_sessions, valid_sessions = torch.utils.data.random_split(session_dataset, [len(session_dataset)-20000, 20000])
    #if len(train_sessions) > CFG.max_train_num:
    #    train_sessions = train_sessions[:CFG.max_train_num]
    print(len(train_sessions), len(valid_sessions))
    session_loader = DataLoader(train_sessions, batch_size=CFG.batch_size, shuffle=False)
    valid_session_loader = DataLoader(valid_sessions, batch_size=CFG.batch_size, shuffle=False)

    subsession_dataset = SmallProductAsinSessionTrainDataset(root="data", 
        transform=lambda x: to_subsession(x, tokenizer),
        pre_filter = lambda data : (data['product'].num_nodes + data['query'].num_nodes)>2,
        pre_transform=pretransform_QueryTokenProductAsin)
    train_subsessions, valid_subsessions = torch.utils.data.random_split(subsession_dataset, [len(subsession_dataset)-20000, 20000])
    print(len(train_subsessions), len(valid_subsessions))
    #if len(train_subsessions) > CFG.max_train_num:
    #    train_subsessions = train_subsessions[:CFG.max_train_num]

    
    subsession_loader = DataLoader(train_subsessions, batch_size=CFG.batch_size, shuffle=False)
    valid_subsession_loader = DataLoader(valid_subsessions, batch_size=CFG.batch_size, shuffle=False)
    """
    dataset = SmallProductAsinSessionTrainDataset(root="data",
    transform=lambda x: (get_target(x,tokenizer, len(asin2id)), to_subsession(x, tokenizer, len(asin2id))),
    pre_filter = lambda data : (data['product'].num_nodes + data['query'].num_nodes)>2,
        pre_transform=pretransform_QueryTokenProductAsin)
    train_set, valid_set = torch.utils.data.random_split(dataset, [len(dataset)-20000, 20000])
    #train_set = train_set[:10] 
    train_loader = MyDataLoader(train_set, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = MyDataLoader(valid_set, batch_size=CFG.batch_size, shuffle=True)
    

    query_node_embedder = NodeTextTransformer(ntoken=tokenizer.vocab_size, 
                        ninp=CFG.emb_len, 
                        nhead=CFG.query_embedder_nhead, 
                        nhid=CFG.query_embedder_nhid, 
                        nlayers=CFG.query_embedder_nlayers,
                        dropout=CFG.query_embedder_dropout,
                        batch_first=True)
    product_node_embedder = NodeAsinEmbedding(nproducts=len(asin2id.keys()), ninp=CFG.emb_len)
    target_asin_embedding = torch.nn.Embedding(len(asin2id.keys()), CFG.emb_len)
    target_token_embedding = torch.nn.Embedding(tokenizer.vocab_size, CFG.emb_len)
    #gnn1 = get_hetero_GNN('GAT', CFG.emb_len, CFG.gnn_nhid, CFG.gnn_nout, 
    #                        CFG.gnn_nhead, train_sessions[0], CFG.gnn_aggr, CFG.gnn_dropout)
    gnn1 = HGT(CFG.gnn_nout, CFG.gnn_nhead, CFG.gnn_nlayers, dataset[0][0])
    graph_pooling1 = GraphPooling('mean', CFG.gnn_nout*(CFG.gnn_nlayers+1), CFG.gnn_pooling_out, CFG.gnn_dropout)
    session_encoder = GraphLevelEncoder(query_node_embedder=query_node_embedder,
                        product_node_embedder=product_node_embedder,
                        gnn=gnn1, graph_pooling=graph_pooling1)

    query_node_embedder2 = NodeTextTransformer(ntoken=tokenizer.vocab_size, 
                        ninp=CFG.emb_len, 
                        nhead=CFG.query_embedder_nhead, 
                        nhid=CFG.query_embedder_nhid, 
                        nlayers=CFG.query_embedder_nlayers,
                        dropout=CFG.query_embedder_dropout,
                        batch_first=True)
    product_node_embedder2 = NodeAsinEmbedding(nproducts=len(asin2id.keys()), ninp=CFG.emb_len)
    #gnn2 = get_hetero_GNN('GAT', CFG.emb_len, CFG.gnn_nhid, CFG.gnn_nout, 
    #                        CFG.gnn_nhead, train_sessions[0], CFG.gnn_aggr, CFG.gnn_dropout)
    
    gnn2 = HGT(CFG.gnn_nout, CFG.gnn_nhead, CFG.gnn_nlayers, dataset[0][0])
    graph_pooling2 = GraphPooling('mean', (CFG.gnn_nlayers+1)*CFG.gnn_nout, CFG.gnn_pooling_out, CFG.gnn_dropout)
    subsession_encoder = GraphLevelEncoder(query_node_embedder=query_node_embedder2,
                        product_node_embedder=product_node_embedder2,
                        gnn=gnn2, graph_pooling=graph_pooling2)
    
    next_product_head = MLP(CFG.gnn_pooling_out*2, CFG.emb_len, CFG.ph_nhid, CFG.ph_nlayers, dropout=CFG.ph_dropout)

    next_query_decoder = MyTransformerDecoder(CFG.emb_len, 
                            CFG.emb_len,
                            CFG.qh_nhead, 
                            CFG.qh_nhid, 
                            CFG.qh_nlayers, 
                            CFG.qh_dropout, 
                            True)
    next_query_decoder_electra = MyTransformerDecoder(CFG.emb_len, 
                            2,
                            CFG.qh_nhead, 
                            CFG.qh_nhid, 
                            CFG.qh_nlayers, 
                            CFG.qh_dropout, 
                            True)
    
    all_product_head = MLP(CFG.gnn_pooling_out*2, CFG.emb_len, CFG.ph_nhid, CFG.ph_nlayers, dropout=CFG.ph_dropout)
    last_query_decoder = MyTransformerDecoder(CFG.emb_len, 
                            CFG.emb_len,
                            CFG.qh_nhead, 
                            CFG.qh_nhid, 
                            CFG.qh_nlayers, 
                            CFG.qh_dropout, 
                            True)
    last_query_decoder_electra = MyTransformerDecoder(CFG.emb_len, 
                            2,
                            CFG.qh_nhead, 
                            CFG.qh_nhid, 
                            CFG.qh_nlayers, 
                            CFG.qh_dropout, 
                            True)
    subsession_encoder.to(device)
    next_product_head.to(device)
    next_query_decoder.to(device)

    session_encoder.to(device)
    all_product_head.to(device)
    last_query_decoder.to(device)

    target_asin_embedding.to(device)
    target_token_embedding.to(device)

    next_query_decoder_electra.to(device)
    last_query_decoder_electra.to(device)

    params = list(session_encoder.parameters())
    params += list(last_query_decoder.parameters()) + list(all_product_head.parameters())
    #params += list(target_asin_embedding.parameters()) + list(target_token_embedding.parameters())
    #params += 
    params += list(last_query_decoder_electra.parameters())
    
    
    best_valid_loss = 1000
    
    optimizer = torch.optim.Adam(params, lr=CFG.lr)
    optimizer2 = torch.optim.Adam(list(target_asin_embedding.parameters())+list(target_token_embedding.parameters()), lr=0.01)
    params = list(subsession_encoder.parameters())
    params += list(next_product_head.parameters()) + list(next_query_decoder.parameters())
    params += list(next_query_decoder_electra.parameters())
    optimizer3 = torch.optim.Adam(params, lr=CFG.lr)
    for epoch in range(CFG.max_epoch):
        # training
        training_loss = []
        train_next_precision = []
        train_next_recall = []
        train_all_precision = []
        train_all_recall = []
                
        subsession_encoder.train()
        next_product_head.train()
        target_asin_embedding.train()
        next_query_decoder.train()
        target_token_embedding.train()
        session_encoder.train()
        all_product_head.train()
        last_query_decoder.train()
        next_query_decoder_electra.train()
        last_query_decoder_electra.train()
    
        #with torch.no_grad():
        #torch.autograd.set_detect_anomaly(True)
        for i, data in enumerate(tqdm(train_loader)):
        #for i, data in enumerate(train_loader):
            #subsession = next(iter(subsession_loader))
            session = data[0]
            subsession = data[1]
            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            #print(1)
            subsession = subsession.to(device)
            try:
                subsession_embedding = subsession_encoder(subsession)
                next_product_loss = get_next_product_asin_loss(subsession_embedding, next_product_head, subsession, target_asin_embedding, device=device)
                #next_query_loss = get_next_query_loss(subsession_embedding, next_query_decoder, subsession, target_token_embedding, neg_k=CFG.neg_k, device=device)
                next_mlm_loss, output = get_next_query_mlm_loss(subsession_embedding, next_query_decoder, subsession, target_token_embedding, device)
                next_electra_loss = get_next_query_electra_loss(subsession_embedding, next_query_decoder_electra, subsession, output, target_token_embedding, device)
                next_query_loss = next_mlm_loss + next_electra_loss
            except Exception as e:
                print(e)
                """
                print(subsession)
                print(subsession['query'].x.size(0))
                print(subsession['product'].x.size(0))
                print(subsession['query'].batch)
                print(subsession['product'].batch)
                print(get_edge_index_max(subsession['query', 'follows', 'query'].edge_index))
                print(get_edge_index_max(subsession['query', 'clicks', 'product'].edge_index))
                print(get_edge_index_max(subsession['query', 'adds', 'product'].edge_index))
                print(get_edge_index_max(subsession['query', 'purchases', 'product'].edge_index))
                print(get_edge_index_max(subsession['product', 'clicked by', 'query'].edge_index))
                print(get_edge_index_max(subsession['product', 'added by', 'query'].edge_index))
                print(get_edge_index_max(subsession['product', 'purchased by', 'query'].edge_index))
                    
                print(subsession['query', 'follows', 'query'].edge_index)
                print(subsession['query', 'clicks', 'product'].edge_index)
                print(subsession['query', 'adds', 'product'].edge_index)
                print(subsession['query', 'purchases', 'product'].edge_index)
                print(subsession['product', 'clicked by', 'query'].edge_index)
                print(subsession['product', 'added by', 'query'].edge_index)
                print(subsession['product', 'purchased by', 'query'].edge_index)
                """
                raise RuntimeError("bug")
                   
            session = session.to(device)
            session_embedding = session_encoder(session)
            all_product_loss = get_all_product_asin_loss(session_embedding, all_product_head, session, target_asin_embedding, device=device)
            #get_all_product_asin_accuracy(session_embedding, all_product_head, session, target_asin_embedding, 100)
            #last_query_loss = get_last_query_loss(session_embedding, last_query_decoder, session, target_token_embedding, neg_k=CFG.neg_k, device=device)
            last_mlm_loss, output = get_last_query_mlm_loss(session_embedding, last_query_decoder, session, target_token_embedding, device)
            last_electra_loss = get_last_query_electra_loss(session_embedding, last_query_decoder_electra, session, output, target_token_embedding, device)
            last_query_loss = last_mlm_loss + last_electra_loss

            contrastive_loss = ContrastiveLoss(session_embedding, subsession_embedding)
            
            #loss = last_query_loss

            #loss = CFG.ph_w * all_product_loss + CFG.qh_w * last_query_loss + CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
            loss = CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
            #loss += CFG.ctv_w * contrastive_loss
            try:
                assert(torch.sum(loss.isnan())==0)
            except:
                print(next_product_loss, last_query_loss)
                print(next_mlm_loss, next_electra_loss, last_mlm_loss, last_electra_loss)
                """
                print(subsession_embedding)
                print(subsession)
                print(torch.sum(session_embedding.isnan()))
                idx = 0
                for i in range(session.num_graphs):
                    if session_embedding[i,:].isnan().any():
                        idx = i
                        print(session_embedding[i,:])
                        break
                print(session['product'].x[session['product'].batch == idx])
                print(session['query'].x[session['query'].batch == idx])
                #for i in range(session_embedding.size(0)):
                #    for j in range(session_embedding.size(1)):
                #        print(session_embedding[i,j].item(), end=' ')
                #    print()
                for i in range(session['product'].batch.size(0)):
                    print(session['product'].batch[i].item(),end=' ')
                """
                raise RuntimeError("Nan in Loss")
            training_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.)
            optimizer.step()
            optimizer2.step()
            optimizer3.step()
            with torch.no_grad():
                precision, recall = get_next_product_asin_accuracy(subsession_embedding, next_product_head, subsession, target_asin_embedding, 100)
                train_next_precision.append(precision)
                train_next_recall.append(recall)

                precision, recall = get_all_product_asin_accuracy(session_embedding, all_product_head, session, target_asin_embedding, 100)
                train_all_precision.append(precision)
                train_all_recall.append(recall)
            if i % CFG.ckpt_iter == 0:
        #if epoch % 10 == 0:
                valid_loss = []
                valid_next_precision = []
                valid_next_recall = []
                valid_all_precision = []
                valid_all_recall = []
                
                subsession_encoder.eval()
                next_product_head.eval()
                target_asin_embedding.eval()
                next_query_decoder.eval()
                target_token_embedding.eval()
                session_encoder.eval()
                all_product_head.eval()
                last_query_decoder.eval()
                next_query_decoder_electra.eval()
                last_query_decoder_electra.eval()
                
                with torch.no_grad():
                    
                    for data in valid_loader:
                        session = data[0]
                        subsession = data[1]
                        session = session.to(device)
                        subsession = subsession.to(device)
                        graph_embedding = subsession_encoder(subsession)
                        next_product_loss = get_next_product_asin_loss(graph_embedding, next_product_head, subsession, target_asin_embedding, device=device)
                        next_mlm_loss, output = get_next_query_mlm_loss(graph_embedding, next_query_decoder, subsession, target_token_embedding, device)
                        next_electra_loss = get_next_query_electra_loss(graph_embedding, next_query_decoder_electra, subsession, output, target_token_embedding, device)
                        next_query_loss = next_mlm_loss + next_electra_loss
                        loss = CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
                        valid_loss.append(loss.item())
                        precision, recall = get_next_product_asin_accuracy(graph_embedding, next_product_head, subsession, target_asin_embedding, 100)
                        valid_next_precision.append(precision)
                        valid_next_recall.append(recall)
                        
                        graph_embedding = session_encoder(session)
                        all_product_loss = get_all_product_asin_loss(graph_embedding, all_product_head, session, target_asin_embedding, device=device)
                        #last_query_loss = get_last_query_loss(graph_embedding, last_query_decoder, session, target_token_embedding, neg_k=CFG.neg_k, device=device)
                        last_mlm_loss, output = get_last_query_mlm_loss(graph_embedding, last_query_decoder, session, target_token_embedding, device)
                        last_electra_loss = get_last_query_electra_loss(graph_embedding, last_query_decoder_electra, session, output, target_token_embedding, device)
                        last_query_loss = last_mlm_loss + last_electra_loss
                        loss = CFG.ph_w * all_product_loss + CFG.qh_w * last_query_loss
                        valid_loss.append(loss.item())
                        precision, recall = get_all_product_asin_accuracy(graph_embedding, all_product_head, session, target_asin_embedding, 100)
                        valid_all_precision.append(precision)
                        valid_all_recall.append(recall)
                        
                    """
                    for data in valid_session_loader:
                        data = data.to(device)
                        graph_embedding = session_encoder(data)
                        all_product_loss = get_all_product_asin_loss(session_embedding, all_product_head, data, target_asin_embedding, device=device)
                        last_query_loss = get_last_query_loss(graph_embedding, last_query_decoder, data, target_token_embedding, neg_k=CFG.neg_k, device=device)
                        loss = CFG.ph_w * all_product_loss + CFG.qh_w * last_query_loss
                        valid_loss.append(loss.item())
                        precision, recall = get_all_product_asin_accuracy(graph_embedding, next_product_head, data, target_token_embedding, 10)
                        valid_all_precision.append(precision)
                        valid_all_recall.append(recall)
                    """
                ave_valid_loss = np.mean(valid_loss)
                #if ave_valid_loss < best_valid_loss:
                #    torch.save((subsession_encoder, next_product_head, next_query_decoder), CFG.savedir+"subsession_model.pth")
                #    torch.save((session_encoder, all_product_head, last_query_decoder), CFG.savedir+"session_model.pth")
                #    torch.save((target_asin_embedding, target_token_embedding), CFG.savedir+"target_embedding.pth")
            
                # validation
                logging.info("Epoch %d, Iter %d, average training loss: %.3f, average valid loss: %.3f"%(epoch, i, np.mean(training_loss), ave_valid_loss))
                logging.info("Average Valid Next Precision: %.3f, Average Valid Next Recall: %.3f, Average Valid All Precision: %.3f, Average Valid All Recall: %.3f"%(np.mean(valid_next_precision), 
                np.mean(valid_next_recall), np.mean(valid_all_precision), np.mean(valid_all_recall)))
                
                subsession_encoder.train()
                next_product_head.train()
                target_asin_embedding.train()
                next_query_decoder.train()
                target_token_embedding.train()
                session_encoder.train()
                all_product_head.train()
                last_query_decoder.train()
                next_query_decoder_electra.train()
                last_query_decoder_electra.train()
                    
                logging.info("Average Train Next Precision: %.3f, Average Train Next Recall: %.3f, Average Train All Precision: %.3f, Average Train All Recall: %.3f"%(np.mean(train_next_precision), 
                    np.mean(train_next_recall), np.mean(train_all_precision), np.mean(train_all_recall)))
                
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