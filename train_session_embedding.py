#from pickletools import optimize
from itertools import product
#from Dataset import ProductAsinSessionTestDataset, ProductAsinSessionTrainDataset
from model.model import GraphLevelEncoder, MLP, MyTransformerDecoder
from model.NodeEmbedding import NodeAsinEmbedding, NodeTextTransformer
from model.gnn import get_hetero_GNN, GraphPooling
#from Dataset import pretransform_QueryTokenProductAsin
from torch_geometric.loader import DataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import numpy as np
from transformers import BertTokenizer
from config import CFG
import pickle
from torch.distributions.bernoulli import Bernoulli

def filter_edge_index(edge_index, max_from_node, max_to_node):
    #print(edge_index, max_from_node, max_to_node)
    mask = (edge_index[0] < max_from_node).int() + (edge_index[1] < max_to_node).int()
    #print('mask')
    #print(edge_index)
    #print(mask)
    #print(edge_index.shape)
    return edge_index[:, mask==2]

def get_target(data, tokenizer, asin_num):
    data['product'].y = data['product'].x 
    data['product'].y_type = data['product'].click_type
    data['product'].y_cnt = torch.bincount(data['product'].y, minlength=asin_num).view(1,-1)
    
    data['query'].y = data['query'].x[-1].view(1,-1) if data['query'].num_nodes > 0 else None
    data['query'].y_mask = data['query'].attention_mask[-1].view(1,-1) if data['query'].num_nodes > 0 else None
    
    if data['query'].y is not None:
        # generate masked y
        m = Bernoulli(CFG.mask_prob)
        while True:
            mask = m.sample(data['query'].y.shape)
            mask = mask.bool()
            all_mask = (mask + (data['query'].y_mask==0))>0
            num = torch.sum(all_mask)
            if num < data['query'].y.size(1):
                break
        #mask = m.sample(data['query'].y.shape)
        #mask = mask.bool()
        data['query'].masked_y = data['query'].y
        data['query'].masked_y[mask] == tokenizer.mask_token
        data['query'].pred_target = mask

    return data



def get_last_query_mlm_loss(graph_embedding, decoder, data, embedder, device):
    tgt = embedder(data['query'].masked_y) # num_graphs * seq_len * e
    mask = (data['query'].pred_target + (data['query'].y_mask==0))>0 # num_graphs * seq_len
    memory = graph_embedding.unsqueeze(dim=1)
    output = decoder(tgt=tgt, memory=memory, tgt_mask=None, tgt_key_padding_mask=mask) # num_graphs * seq_len * e
    tgt_emb = embedder.weight.clone() # vocab_size * e
    val = torch.matmul(output, tgt_emb.T) # num_graphs * seq_len * e
    
    val = torch.transpose(val, 1, 2) # num_graphs * e * seq_len
    #print(val.shape)
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    loss_mat = criterion(val, data['query'].y) # num_graphs * seq_len
    loss = torch.sum(loss_mat * data['query'].pred_target) / torch.sum(data['query'].pred_target)
    
    pred = torch.argmax(val, dim=1).detach() # num_graphs * seq_len
    output = torch.where(data['query'].pred_target>0, pred, data['query'].masked_y)
    output = output.detach()
    
    return loss, output

def get_last_query_electra_loss(graph_embedding, decoder, data, output, embedder, device):
    tgt = embedder(output)
    memory = graph_embedding.unsqueeze(dim=1)
    out = decoder(tgt=tgt, memory=memory, tgt_mask=None, tgt_key_padding_mask=data['query'].y_mask==0) # num_graphs * seq_len * 2
    out = torch.transpose(out, 1,2) # num_graphs * 2 * seq_len

    label = (output == data['query'].y).long() # 1 means same, 0 means not same
    criterion=torch.nn.CrossEntropyLoss()
    loss = criterion(out, label)
    return loss


def get_all_product_asin_accuracy(graph_embedding, product_head, data, asin_embedding, K):
    if product_head is not None:
        rep = product_head(graph_embedding)
    else:
        rep = graph_embedding 
    #rep = graph_embedding
    val = torch.sigmoid(rep @ asin_embedding.weight.clone().T) #num_asin * num_graphs
    assert(val.isnan().any()==False)
    val = torch.clip(val, min=1e-4, max=0.9999)
    #print(val.shape)
    #print(K)
    _, pred = torch.topk(val, K, dim=1)
    #print(pred.shape)
    #pred = pred.T # num_graphs * K
    y = data['product'].y.long()
    precision = []
    recall = []
    #a = set(y.detach().cpu().tolist())
    #b = set(pred.flatten().detach().cpu().tolist())
    #print(a)
    #print(b)
    #print(a&b)
    #pp = pred.flatten()
    #out = len(set(pp.tolist()) - set(y.tolist()))
    
    for i in range(data.num_graphs):
        gt = set(y[data['product'].batch==i].detach().cpu().tolist())
        p = set(pred[i,:].detach().cpu().tolist())
        hit = float(len(gt & p))
        precision.append(hit / K)
        recall.append(hit / len(gt))
        #print(hit, len(gt))
    return np.mean(precision), np.mean(recall)


def get_all_product_asin_loss(graph_embedding, product_head, data, asin_embedding, device):
    #y =(data['product'].y_cnt != 0).float()
    #data['product_target'].y_cnt = torch.zeros(asin_num).view(1,-1)
    #data['product_target'].y_cnt[0, data['product_target'].y] = 1
    #y_cnt = torch.bincount(data['product'].y, minlength=asin_embedding.num_embeddings).view(1,-1)
    #y = (y_cnt != 0).float()
    y = torch.zeros(data.num_graphs, asin_embedding.num_embeddings).float()
    y = y.to(device)
    y[data['product'].batch, data['product'].y] = 1.
    if product_head is not None:
        rep = product_head(graph_embedding)
    else:
        rep = graph_embedding
    assert(rep.isnan().any()==False)
    val = torch.sigmoid(rep @ asin_embedding.weight.clone().T)
    try:
        assert(val.isnan().any()==False)
    except Exception as e:
        print(e)
        val[val.isnan()] = 0
        print(torch.max(val))
        raise RuntimeError("nan")
    val = torch.clip(val, min=1e-4, max=0.9999)
    loss_mat = -(y*torch.log(val)+(1-y)*torch.log(1-val))
    neg_mask = (torch.rand(loss_mat.shape) < (1000/loss_mat.size(1))).to(device).detach()
    loss_mask = torch.logical_or(neg_mask, y)
    diff_mask = (torch.abs(val-y)>0.5).detach()
    if torch.mean(torch.abs(val[loss_mask]-y[loss_mask])) < 1e-2 and False:
        loss_mask = torch.logical_or(y, diff_mask).detach()
        #print(torch.sum(diff_mask))
    #print(torch.sum(loss_mask))
    #print(loss_mat[neg_mask])
    #print(val[neg_mask])
    loss  = torch.mean(loss_mat[loss_mask])
    #loss_mat = loss_mat/torch.sum(loss_mask)
    #loss = torch.sum(loss_mat[loss_mask])
    #print(val[loss_mask])
    #print(y[loss_mask])
    #print(torch.max(torch.abs(val[loss_mask]-y[loss_mask])))
    #print(torch.sum(val>0.5), torch.sum(y),loss)

    #_, pred = torch.topk(val, 100, dim=1)
    #label = data['product'].y.long()
    #y_hat = torch.zeros_like(y)
    #y_hat[data['product'].batch, label]=1
    #print(torch.sum(torch.abs(y-y_hat)), pred.shape)
    #precision = []
    #recall = []
    #print(set(label.detach().cpu().tolist()))
    #print(set(pred.flatten().detach().cpu().tolist()))
    #print(set(label.detach().cpu().tolist()) & set(pred.flatten().detach().cpu().tolist()))
    
    return loss

"""
def get_all_product_asin_loss(graph_embedding, product_head, data, asin_embedding, device):
    y = data['product'].y_cnt / torch.sum(data['product'].y_cnt, dim=1).unsqueeze(-1)
    if product_head is None:
        rep = product_head(graph_embedding)
    else:
        rep = graph_embedding
    val = rep @ asin_embedding.weight.clone().T 
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(val, y)
"""
"""
def get_all_product_asin_loss(graph_embedding, product_head, data, asin_embedding, device):
    #print('loss')
    if product_head is not None:
        rep = product_head(graph_embedding) # num_graphs * d
    # normalize rep
    #rep = rep / torch.sum(rep**2, dim=1)
    #print(rep)
    sample = data['product'].y
    sample_emb = asin_embedding(sample) # n* d
    assert(torch.sum(sample_emb.isnan())==0)
    #print(sample_emb)
    #print(rep)
    #print(sample_emb)
    val = torch.matmul(rep, sample_emb.T)
    assert(torch.sum(sample_emb.isnan())==0)

    mask1 = data['product'].batch.repeat((data.num_graphs,1))
    mask2 = torch.arange(data.num_graphs).repeat((data['product'].batch.shape[0],1)).T
    mask2 = mask2.to(device)

    sample_type = data['product'].y_type # c: 0, ca: 1, p: 2
    sample_w = 1 + sample_type.float()/5
    sample_w = sample_w.repeat((data.num_graphs, 1))
    sample_w = torch.where(mask1 == mask2, sample_w.float(), torch.ones_like(mask1).float())
    #print(torch.mean(sample_w))
    # print(sample_w)
    #pos_sample_mask = mask1 == mask2
    #print(torch.where(mask1 == mask2, val, -val))
    val = torch.sigmoid(torch.where(mask1 == mask2, val, -val)) 
    #print(torch.mean(val))
    val = val * sample_w
    #print(val)
    #print('end loss')
    
    return -torch.mean(val)
"""

def get_last_query_loss(graph_embedding, decoder, data, embedder, neg_k, device):
    # memory is graph_embedding
    if data['query'].num_nodes < 2:
        return 0
    y = data['query'].y # num_graphs * max_len
    y_mask = data['query'].y_mask == 0
    
    size = y.shape[1]
    nopeak_mask = np.triu(np.ones((size, size)), k=1).astype('uint8')
    loss_mask = nopeak_mask-np.triu(np.ones((size, size)), k=2).astype('uint8')
    loss_mask = torch.from_numpy(loss_mask) == 1
    nopeak_mask = torch.from_numpy(nopeak_mask)==1
    loss_mask = loss_mask.to(device)
    nopeak_mask = nopeak_mask.to(device)

    emb_len = graph_embedding.shape[1]
    memory = torch.reshape(graph_embedding.repeat(1,size), (-1,1,emb_len))
    
    tgt_sequence = torch.reshape(y.repeat((1, size)), (-1, size))
    tgt_sequence = embedder(tgt_sequence) # S*S*E
    
    y_mask = torch.reshape(y_mask.repeat((1,size)), (-1,size))
   
    output = decoder(tgt=tgt_sequence, memory=memory, tgt_mask=nopeak_mask, tgt_key_padding_mask=y_mask) # batch * size * e
    #print(output.shape)
    mask = (y_mask==0) & loss_mask.repeat((data.num_graphs,1))
    #print(mask.shape)
    sample_mask = torch.sum(mask, dim=1)
    #print(sample_mask)
    sample_cnt = torch.sum(sample_mask)
    mask = mask.repeat((emb_len,1)).reshape(output.shape)
    #print(mask.shape)

    rep = torch.sum(mask*output,dim=1) # batch * e

    pos_sample = torch.reshape(y, (-1,))
    pos_sample[:-1] = pos_sample[1:].clone()
    #print(pos_sample)
    pos_sample_emb = embedder(pos_sample)

    #print(pos_sample.shape[0], neg_k)
    neg_sample_emb = embedder(torch.randint(embedder.num_embeddings, (pos_sample.shape[0], neg_k)).to(device))

    pos_sample_val = torch.sigmoid(torch.sum(rep*pos_sample_emb,dim=1)) # batch
    neg_sample_val = torch.sigmoid(-torch.bmm(neg_sample_emb, rep.unsqueeze(-1)).squeeze()) # batch * k
    neg_sample_val = torch.sum(neg_sample_val, dim=1) # batch
    loss = torch.sum(pos_sample_val*sample_mask)/sample_cnt + torch.sum(neg_sample_val*sample_mask)/sample_cnt

    return -loss/ (1+neg_k)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ProductAsinSessionTrainDataset(root="data", 
        transform=get_target,
        pre_filter = lambda data : (data['product'].num_nodes + data['query'].num_nodes)>2,
        pre_transform=pretransform_QueryTokenProductAsin)
    train_set, valid_set = torch.utils.data.random_split(dataset, [len(dataset)-100000, 100000])
    loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=CFG.batch_size, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with open("data/us-20220401-asin2id-brand2id.pkl", "rb") as f:
        asin2id, _ = pickle.load(f)

    query_node_embedder = NodeTextTransformer(ntoken=tokenizer.vocab_size, 
                        ninp=CFG.emb_len, 
                        nhead=CFG.query_embedder_nhead, 
                        nhid=CFG.query_embedder_nhid, 
                        nlayers=CFG.query_embedder_nlayers,
                        batch_first=True)
    product_node_embedder = NodeAsinEmbedding(nproducts=len(asin2id.keys()), ninp=CFG.emb_len)
    gnn = get_hetero_GNN('GAT', CFG.emb_len, CFG.gnn_nhid, CFG.gnn_nout, 
                            CFG.gnn_nhead, dataset[0], CFG.gnn_aggr, CFG.gnn_dropout)
    graph_pooling = GraphPooling('mean', CFG.gnn_nhead*CFG.gnn_nout, CFG.gnn_pooling_out, CFG.gnn_dropout)
    graph_encoder = GraphLevelEncoder(query_node_embedder=query_node_embedder,
                        product_node_embedder=product_node_embedder,
                        gnn=gnn, graph_pooling=graph_pooling)
    
    all_product_head = MLP(CFG.gnn_pooling_out, CFG.emb_len, CFG.ph_nhid, CFG.ph_nlayers, dropout=CFG.ph_dropout)

    last_query_decoder = MyTransformerDecoder(CFG.gnn_pooling_out, 
                            CFG.emb_len,
                            CFG.qh_nhead, 
                            CFG.qh_nhid, 
                            CFG.qh_nlayers, 
                            CFG.qh_dropout, 
                            True)

    graph_encoder.to(device)
    all_product_head.to(device)
    last_query_decoder.to(device)

    params = list(graph_encoder.parameters()) + list(all_product_head.parameters()) + list(last_query_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=CFG.lr)

    best_valid_loss = 1000
    for epoch in range(CFG.max_epoch):
        # training
        training_loss = []
        for data in loader:
            optimizer.zero_grad()
            data = data.to(device)
            graph_embedding = graph_encoder(data)
            next_product_loss = get_all_product_asin_loss(graph_embedding, all_product_head, data, graph_encoder.product_node_embedder)
            next_query_loss = get_last_query_loss(graph_embedding,last_query_decoder, data, query_node_embedder.embedding, neg_k=CFG.neg_k)
            loss = CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        valid_loss = []
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                graph_embedding = graph_encoder(data)
                all_product_loss = get_all_product_asin_loss(graph_embedding, all_product_head, data, graph_encoder.product_node_embedder)
                last_query_loss = get_last_query_loss(graph_embedding, last_query_decoder, data, query_node_embedder.embedding, neg_k=CFG.neg_k)
                loss = CFG.ph_w * all_product_loss + CFG.qh_w * last_query_loss
                valid_loss.append(loss.item())
        ave_valid_loss = np.mean(valid_loss)
        if ave_valid_loss < best_valid_loss:
            torch.save((graph_encoder, all_product_head, last_query_decoder), CFG.savedir+"session_model.pth")
            
        # validation
        print("Epoch %d, average training loss: %.3f, average valid loss: %.3f"%(epoch, np.mean(training_loss), ave_valid_loss))
        

if __name__ == '__main__':
    main()
