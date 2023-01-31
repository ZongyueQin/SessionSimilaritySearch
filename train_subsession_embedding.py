from os import supports_bytes_environ
from pickletools import optimize
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

def get_edge_index_max(edge_index):
    #print(edge_index.shape)
    if edge_index.size(-1) == 0:
        return -1
    else:
        return edge_index.max()

def to_subsession(data, tokenizer, asin_num):
    #print('to_session')
    # select a action as the last action, the rest of the action will be treated as targets (labels)
    subsession = HeteroData()
    # randomly sample a product to be the next product
    num_products = data['product'].x.shape[0]
    next_product = np.random.randint(num_products)
    # randomly decide if we should include the linked query node
    edge_index = torch.cat((data['query', 'clicks', 'product'].edge_index,
                            data['query', 'adds', 'product'].edge_index,
                            data['query', 'purchases', 'product'].edge_index),1)
    #print(edge_index)
    #print(next_product)
    #print(num_products)
    row, col = edge_index
    query_node = row[col==next_product]
    try:
        assert(query_node.shape[0]==1)
    except Exception as e:
        print(e)
        print(edge_index)
        print(next_product)
        print(data)
        raise RuntimeError("assert error")
    query_node = query_node[0].item()

    if next_product == 0 and query_node == 0:
        next_product = 1
        query_node = row[col==next_product]
        #print(edge_index)
        if len(query_node) != 0:
            query_node = query_node[0].item()
        else:
            next_product = 0
            query_node = 1

    keep_query = np.random.randint(2)

    if query_node <= 1 or keep_query == 1:
        next_query = query_node+1
    else:
        next_query = query_node

    

    #print(next_query)
    subsession['product'].x = data['product'].x[:next_product]
    subsession['query'].x = data['query'].x[:next_query]
    subsession['query'].token_type_ids = data['query'].token_type_ids[:next_query]
    subsession['query'].attention_mask = data['query'].attention_mask[:next_query]

    subsession['query', 'follows', 'query'].edge_index = filter_edge_index(data['query', 'follows', 'query'].edge_index,
                                                        next_query, next_query)
    subsession['query', 'clicks', 'product'].edge_index = filter_edge_index(data['query', 'clicks', 'product'].edge_index,
                                                        next_query, next_product)
    subsession['query', 'adds', 'product'].edge_index = filter_edge_index(data['query', 'adds', 'product'].edge_index,
                                                        next_query, next_product)
    subsession['query', 'purchases', 'product'].edge_index = filter_edge_index(data['query', 'purchases', 'product'].edge_index,
                                                        next_query, next_product)
        
    subsession['product', 'clicked by', 'query'].edge_index = filter_edge_index(data['product', 'clicked by', 'query'].edge_index,
                                                        next_product, next_query)
    subsession['product', 'added by', 'query'].edge_index = filter_edge_index(data['product', 'added by', 'query'].edge_index,
                                                        next_product, next_query)
    subsession['product', 'purchased by', 'query'].edge_index = filter_edge_index(data['product', 'purchased by', 'query'].edge_index,
                                                        next_product, next_query)
        
    
    # Filter Isolated Product Nodes
    edge_index = torch.cat((subsession['query', 'clicks', 'product'].edge_index,
                            subsession['query', 'adds', 'product'].edge_index,
                            subsession['query', 'purchases', 'product'].edge_index),1)
    cnt = 0
    #print(edge_index)
    for i in range(next_product):
        product = next_product-i-1
        if torch.sum(edge_index[1]==product) == 0:
            cnt += 1
        else:
            break
    subsession['product'].x = subsession['product'].x[:next_product-cnt]
    subsession['product'].num_nodes = next_product-cnt
    #print(';;;')
    #print(subsession['product'].x.shape)
    #print(subsession['product'].num_nodes)
    if subsession['product'].num_nodes == 0:
        subsession['product'].x = torch.zeros_like(data['product'].x[:1])
        subsession['product'].num_nodes = 1
       # print(subsession['product'].x)
    subsession['query'].num_nodes = next_query

    tmp_data = data.to_homogeneous()
    assert(tmp_data.has_isolated_nodes() == False)
    tmp_data = subsession.to_homogeneous()
    #print(subsession)
    #print(tmp_data.edge_index)
    #print(tmp_data.num_nodes)
    try:
        assert(tmp_data.has_isolated_nodes() == False or subsession['product'].num_nodes == 1)
    except Exception as e:
        print(subsession)
        print(data)
        print(next_product)
        print(next_query)
        raise RuntimeError("assert error")
    subsession['product_target'].y = data['product'].x[next_product:]
    subsession['product_target'].y_cnt = torch.bincount(subsession['product_target'].y, minlength=asin_num).view(1,-1)
    subsession['product_target'].y_type = data['product'].click_type[next_product:]
    subsession['product_target'].num_nodes = subsession['product_target'].y.shape[0]
    
    if next_query == data['query'].num_nodes:
        subsession['query_target'].y = data['query'].x[next_query-1].view(1,-1)
        subsession['query_target'].y_mask = data['query'].attention_mask[next_query-1].view(1,-1)
        subsession['query_target'].num_nodes = 1
    else:        
        subsession['query_target'].y = data['query'].x[next_query].view(1,-1)
        subsession['query_target'].y_mask = data['query'].attention_mask[next_query].view(1,-1)
        subsession['query_target'].num_nodes = 1
    
    m = Bernoulli(CFG.mask_prob)
    subsession['query_target'].masked_y = subsession['query_target'].y
    while True:
        mask = m.sample(subsession['query_target'].masked_y.shape)
        mask = mask.bool()
        all_mask = (mask + (subsession['query_target'].y_mask==0))>0
        num = torch.sum(all_mask)
        if num < subsession['query_target'].y.size(1):
            break
    subsession['query_target'].masked_y[mask] == tokenizer.mask_token
    subsession['query_target'].pred_target = mask

    try:
        assert(get_edge_index_max(subsession['query', 'follows', 'query'].edge_index) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['query', 'clicks', 'product'].edge_index[0,:]) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['query', 'clicks', 'product'].edge_index[1,:]) < subsession['product'].x.size(0))
        assert(get_edge_index_max(subsession['query', 'adds', 'product'].edge_index[0,:]) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['query', 'adds', 'product'].edge_index[1,:]) < subsession['product'].x.size(0))
        assert(get_edge_index_max(subsession['query', 'purchases', 'product'].edge_index[0,:]) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['query', 'purchases', 'product'].edge_index[1,:]) < subsession['product'].x.size(0))

        assert(get_edge_index_max(subsession['product', 'clicked by', 'query'].edge_index[1,:]) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['product', 'clicked by', 'query'].edge_index[0,:]) < subsession['product'].x.size(0))
        assert(get_edge_index_max(subsession['product', 'added by', 'query'].edge_index[1,:]) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['product', 'added by', 'query'].edge_index[0,:]) < subsession['product'].x.size(0))
        assert(get_edge_index_max(subsession['product', 'purchased by', 'query'].edge_index[1,:]) < subsession['query'].x.size(0))
        assert(get_edge_index_max(subsession['product', 'purchased by', 'query'].edge_index[0,:]) < subsession['product'].x.size(0))
    except:
        print(subsession['query'].x.size(0))
        print(subsession['product'].x.size(0))
        print(subsession['query', 'follows', 'query'].edge_index)
        print(subsession['query', 'clicks', 'product'].edge_index)
        print(subsession['query', 'adds', 'product'].edge_index)
        print(subsession['query', 'purchases', 'product'].edge_index)
        print(subsession['product', 'clicked by', 'query'].edge_index)
        print(subsession['product', 'added by', 'query'].edge_index)
        print(subsession['product', 'purchased by', 'query'].edge_index)

        print(next_product)
        print(next_query)
        print(data)
        print(data['query', 'follows', 'query'].edge_index)
        print(data['query', 'clicks', 'product'].edge_index)
        print(data['query', 'adds', 'product'].edge_index)
        print(data['query', 'purchases', 'product'].edge_index)
        print(data['product', 'clicked by', 'query'].edge_index)
        print(data['product', 'added by', 'query'].edge_index)
        print(data['product', 'purchased by', 'query'].edge_index)
        raise Exception("subsession error")
    return subsession

def get_next_query_mlm_loss(graph_embedding, decoder, data, embedder, device):
    tgt = embedder(data['query_target'].masked_y) # num_graphs * seq_len * e
    mask = (data['query_target'].pred_target + (data['query_target'].y_mask==0))>0 # num_graphs * seq_len
    memory = graph_embedding.unsqueeze(dim=1)
    #print(torch.sum(mask,dim=1))
    #print(memory.isnan().any())
    #print(tgt.isnan().any())
    output = decoder(tgt=tgt, memory=memory, tgt_mask=None, tgt_key_padding_mask=mask) # num_graphs * seq_len * e
    tgt_emb = embedder.weight.clone() # vocab_size * e
    #print(output.isnan().any())
    #print(tgt_emb.isnan().any())
    val = torch.matmul(output, tgt_emb.T) # num_graphs * seq_len * e
    #print(val.shape)
    val = torch.transpose(val, 1, 2) # num_graphs * e * seq_len
    #print(val.shape)
    #print(val.isnan().any())
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    loss_mat = criterion(val, data['query_target'].y) # num_graphs * seq_len
    #print(loss_mat.isnan().any())
    loss = torch.sum(loss_mat * data['query_target'].pred_target) / torch.sum(data['query_target'].pred_target)
    
    pred = torch.argmax(val, dim=1).detach() # num_graphs * seq_len
    output = torch.where(data['query_target'].pred_target>0, pred, data['query_target'].masked_y)
    output = output.detach()
    
    return loss, output

def get_next_query_electra_loss(graph_embedding, decoder, data, output, embedder, device):
    tgt = embedder(output)
    memory = graph_embedding.unsqueeze(dim=1)
    out = decoder(tgt=tgt, memory=memory, tgt_mask=None, tgt_key_padding_mask=data['query_target'].y_mask==0) # num_graphs * seq_len * 2
    out = torch.transpose(out, 1,2) # num_graphs * 2 * seq_len

    label = (output == data['query_target'].y).long() # 1 means same, 0 means not same
    criterion=torch.nn.CrossEntropyLoss()
    loss = criterion(out, label)
    return loss

"""
def get_next_product_asin_loss(graph_embedding, next_product_head, data, asin_embedding, device):
    #print(graph_embedding)
    if next_product_head is not None:
        rep = next_product_head(graph_embedding) # num_graphs * d
    # normalize rep
    #rep = rep / torch.sum(rep**2, dim=1)

    sample = data['product_target'].y
    sample_emb = asin_embedding(sample) # n* d
    val = torch.matmul(rep, sample_emb.T)

    mask1 = data['product_target'].batch.repeat((data.num_graphs,1))
    mask2 = torch.arange(data.num_graphs).repeat((data['product_target'].batch.shape[0],1)).T
    mask2 = mask2.to(device)
    #pos_sample_mask = mask1 == mask2

    sample_type = data['product_target'].y_type # c: 0, ca: 1, p: 2
    sample_w = 1 + sample_type.float()/5
    sample_w = sample_w.repeat((data.num_graphs, 1))
    sample_w = torch.where(mask1 == mask2, sample_w.float(), torch.ones_like(mask1).float())

    val = torch.sigmoid(torch.where(mask1 == mask2, val, -val))
    val = sample_w * val
    
    return -torch.mean(val)
"""

def get_next_product_asin_loss(graph_embedding, product_head, data, asin_embedding, device):
    #y =(data['product_target'].y_cnt != 0).float()
    y = torch.zeros(data.num_graphs, asin_embedding.num_embeddings).float()
    y = y.to(device)
    y[data['product_target'].batch, data['product_target'].y] = 1.
    if product_head is not None:
        rep = product_head(graph_embedding)
    else:
        rep = graph_embedding
    assert(rep.isnan().any()==False)
    #val = rep @ asin_embedding.weight.clone().T 
    val = torch.sigmoid(rep @ asin_embedding.weight.clone().T)
    try:
        assert(val.isnan().any()==False)
    except Exception as e:
        print(e)
        val[val.isnan()] = 0
        print(torch.max(val))
    val = torch.clip(val, min=1e-4, max=0.9999)
    loss_mat = -(y*torch.log(val)+(1-y)*torch.log(1-val))
    assert(loss_mat.isnan().any()==False)
    neg_mask = (torch.rand(loss_mat.shape) < (1000/loss_mat.size(1))).to(device).detach()
    loss_mask = torch.logical_or(neg_mask, y)
    assert(torch.sum(loss_mask)>0)
    #print(torch.sum(loss_mask))
    diff_mask = (torch.abs(val-y)>0.5).detach()
    #if torch.mean(torch.abs(val[loss_mask]-y[loss_mask])) < 1e-2 and False:
    #    loss_mask = torch.logical_or(y, diff_mask).detach()
        
    loss  = torch.mean(loss_mat[loss_mask])
    #print(loss)
    return loss
    #return -torch.mean(y*torch.log(val+1e-4)+(1-y)*torch.log(1-val+1e-4))


"""
def get_next_product_asin_loss(graph_embedding, product_head, data, asin_embedding, device):
    y = data['product_target'].y_cnt / torch.sum(data['product_target'].y_cnt, dim=1).view(-1,1)

    if product_head is None:
        rep = product_head(graph_embedding)
    else:
        rep = graph_embedding
    val = rep @ asin_embedding.weight.clone().T 
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(val, y)
"""
def get_next_product_asin_accuracy(graph_embedding, product_head, data, asin_embedding, K):
    if product_head is not None:
        rep = product_head(graph_embedding)
    else:
        rep = graph_embedding 
    val = torch.sigmoid(rep @ asin_embedding.weight.clone().T)
    val = torch.clip(val, min=1e-4, max=0.9999)
    _, pred = torch.topk(val, K, dim=1)
    #pred = pred.T # num_graphs * K
    
    y = data['product_target'].y.long()
    precision = []
    recall = []
    for i in range(data.num_graphs):
        gt = set(y[data['product_target'].batch==i].detach().cpu().tolist())
        if len(gt) == 0:
            continue
        p = set(pred[i,:].detach().cpu().tolist())
        hit = float(len(gt & p))
        precision.append(hit / K)
        recall.append(hit / len(gt))
    return np.mean(precision), np.mean(recall)



def get_next_query_loss(graph_embedding, decoder, data, embedder, neg_k, device):
    # memory is graph_embedding
    y = data['query_target'].y # num_graphs * max_len
    if y is None:
        return 0
    if data['query'].num_nodes < 2:
        return 0
    y_mask = data['query_target'].y_mask == 0

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
    
    mask = (y_mask==0) & loss_mask.repeat((data.num_graphs,1))
    sample_mask = torch.sum(mask, dim=1)
    sample_cnt = torch.sum(sample_mask)
    mask = mask.repeat((emb_len,1)).reshape(output.shape)

    rep = torch.sum(mask*output,dim=1) # batch * e

    pos_sample = torch.reshape(y, (-1,))
    pos_sample_emb = embedder(pos_sample)
    #print(pos_sample.shape[0], neg_k)
    neg_sample_emb = embedder(torch.randint(embedder.num_embeddings, (pos_sample.shape[0], neg_k)).to(device))

    pos_sample_val = torch.sigmoid(torch.sum(rep*pos_sample_emb,dim=1)) # batch
    neg_sample_val = torch.sigmoid(-torch.bmm(neg_sample_emb, rep.unsqueeze(-1)).squeeze()) # batch * k
    neg_sample_val = torch.sum(neg_sample_val, dim=1) # batch
    loss = torch.sum(pos_sample_val*sample_mask)/sample_cnt + torch.sum(neg_sample_val*sample_mask)/sample_cnt

    return -loss/(1+neg_k)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ProductAsinSessionTrainDataset(root="data", 
        transform=to_subsession,
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
    
    next_product_head = MLP(CFG.gnn_pooling_out, CFG.emb_len, CFG.ph_nhid, CFG.ph_nlayers, dropout=CFG.ph_dropout)

    next_query_decoder = MyTransformerDecoder(CFG.gnn_pooling_out, 
                            CFG.emb_len,
                            CFG.qh_nhead, 
                            CFG.qh_nhid, 
                            CFG.qh_nlayers, 
                            CFG.qh_dropout, 
                            True)

    graph_encoder.to(device)
    next_product_head.to(device)
    next_query_decoder.to(device)

    params = list(graph_encoder.parameters()) + list(next_product_head.parameters()) + list(next_query_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=CFG.lr)

    best_valid_loss = 1000
    for epoch in range(CFG.max_epoch):
        # training
        training_loss = []
        for data in loader:
            optimizer.zero_grad()
            data = data.to(device)
            graph_embedding = graph_encoder(data)
            next_product_loss = get_next_product_asin_loss(graph_embedding, next_product_head, data, graph_encoder.product_node_embedder)
            next_query_loss = get_next_query_loss(graph_embedding, next_query_decoder, data, query_node_embedder.embedding, neg_k=CFG.neg_k)
            loss = CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()

        valid_loss = []
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                graph_embedding = graph_encoder(data)
                next_product_loss = get_next_product_asin_loss(graph_embedding, next_product_head, data, graph_encoder.product_node_embedder)
                next_query_loss = get_next_query_loss(graph_embedding, next_query_decoder, data, query_node_embedder.embedding, neg_k=CFG.neg_k)
                loss = CFG.ph_w * next_product_loss + CFG.qh_w * next_query_loss
                valid_loss.append(loss.item())
                
        ave_valid_loss = np.mean(valid_loss)
        if ave_valid_loss < best_valid_loss:
            torch.save((graph_encoder, next_product_head, next_query_decoder), CFG.savedir+"subsession_model.pth")
            
        # validation
        print("Epoch %d, average training loss: %.3f, average valid loss: %.3f"%(epoch, np.mean(training_loss), ave_valid_loss))
        

if __name__ == '__main__':
    main()
