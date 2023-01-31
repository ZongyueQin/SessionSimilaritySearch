#from Dataset import ProductAsinSessionTestDataset, ProductAsinSessionTrainDataset
from model.model import GraphLevelEncoder, MLP, MyTransformerDecoder
from model.NodeEmbedding import NodeAsinEmbedding, NodeTextTransformer
from model.gnn import get_hetero_GNN, GraphPooling, HGT
#from Dataset import pretransform_QueryTokenProductAsin, SmallProductAsinSessionTrainDataset
from torch_geometric.loader import DataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import numpy as np
from transformers import AutoModel, AutoTokenizer
from config import CFG
import pickle
from train_subsession_embedding import to_subsession, get_next_product_asin_loss, get_next_query_loss, get_next_product_asin_accuracy, get_edge_index_max
from train_session_embedding import get_target, get_all_product_asin_loss, get_last_query_loss, get_all_product_asin_accuracy
from train_subsession_embedding import get_next_query_mlm_loss, get_next_query_electra_loss
from train_session_embedding import get_last_query_electra_loss, get_last_query_mlm_loss
import logging
from tqdm import tqdm
import os
import faiss
#from pretrain_filtered_amazon import sequence_to_graph
from util_amazon_filtered import sequence_to_graph
from collections import defaultdict
import time
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix, vstack
import Levenshtein
from util_amazon_filtered import normalize, get_item, sequence_to_graph, get_query, get_string_match, session_to_text
from torch_geometric.nn import global_mean_pool
from fine_tune_filtered_amazon import get_score, get_ave_score
import sys
os.environ['CUDA_VISIBLE_DEVICES']='2'


def sequence_to_stan_vec(seq, asin_num, lammy):
    vec = np.zeros(asin_num)
    item_seq = [action for action in seq if action[1] != 's']
    if len(item_seq) == 0:
        return vec
    for i in range(len(item_seq)):
        w = np.exp((i-len(item_seq))/lammy)
        vec[item_seq[i][-1]] += w
        
    return vec/np.sqrt(np.sum(vec**2))

def sequence_to_binary_vec(seq, asin_num):
    item_seq = [action for action in seq if action[1] != 's']
    vec = np.zeros(asin_num)
    
    if len(item_seq) == 0:
        return vec
    for i in range(len(item_seq)):
        vec[item_seq[i][-1]] = 1
    
    return vec/np.sqrt(np.sum(vec**2))

def get_prediction_by_knn(emb, index, dataset, sample_size, K):
    emb = emb.detach().cpu().numpy()
    D, I = index.search(emb, sample_size)
    D = D.squeeze()
    I = I.squeeze()

    # retrieve all asins in I, weighted by sum of D, sort and return top K
    session_list = [dataset[i] for i in I]
    #print("3-nn")
    #print(session_list[0]['product'].x)
    #print(session_list[1]['product'].x)
    #print(session_list[2]['product'].x)
    asins = np.concatenate([session['product'].x for session in session_list],axis=0)
    weights = np.concatenate([np.ones_like(session['product'].x)*D[i] for i, session in enumerate(session_list)])
    aw = defaultdict(lambda : 0)
    for i in range(asins.shape[0]):
        aw[asins[i]] += weights[i]
    sorted_aw = sorted(aw.items(), key=lambda x: x[1], reverse=True)
    pred_items = sorted_aw[:K]
    return [pred[0] for pred in pred_items]

def get_p_r(gt, pred, K):
    pred = pred[:K]
    hit = float(len(gt & set(pred)))
    precision = hit/K
    recall = hit/len(gt)
    return precision, recall

def main():
    
    #logging.basicConfig(filename =  CFG.savedir+"test.log",
    #                level = logging.INFO,
    #                format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    asin_num = 43097
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #session_index = faiss.read_index("yoochoose_session_emb.faiss_index")
    #subsession_index = faiss.read_index("yoochoose_subsession_emb.faiss_index")
    
    
    #print(tokenizer.model_max_length)
    #assert(tokenizer.model_max_length==20)
    test_data = pickle.load(open('/home/ec2-user/SR-GNN/datasets/yoochoose1_64/test.txt', 'rb'))
    dataset = [sequence_to_graph(seq,tar,asin_num, True) for seq,tar in zip(test_data[0],test_data[1])]
    print(len(dataset))
    #with open("yoochoose-test.pkl", "wb") as f:
    #    pickle.dump(dataset, f)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    subsession_encoder, next_product_head, _ = torch.load(CFG.savedir+"subsession_model_cont.pth")
    session_encoder, all_product_head, _ = torch.load(CFG.savedir+"session_model_cont.pth")
    target_asin_embedding = torch.load(CFG.savedir+"target_embedding_cont.pth")
   
    subsession_encoder.to(device)
    #next_product_head.to(device)
    session_encoder.to(device)
    #all_product_head.to(device)
    target_asin_embedding.to(device)
    
    subsession_encoder.eval()
    next_product_head.eval()
    session_encoder.eval()
    all_product_head.eval()
    target_asin_embedding.eval()
    # use model to predict
    """
    recall_list = []
    with torch.no_grad():
        for data in tqdm(loader):
            subsession = data.to(device)
            subsession_embedding = subsession_encoder(subsession)
            _, recall = get_next_product_asin_accuracy(subsession_embedding, next_product_head, subsession, target_asin_embedding, 20)
            recall_list.append(recall)
    logging.info("Model Predict, Recall: %.3f"%(np.mean(recall_list)))
    """
    # use knn to predict (subsession - session)
    train_data = pickle.load(open('/home/ec2-user/SR-GNN/datasets/yoochoose1_64/all_train_seq.txt', 'rb'))
    #train_set = [sequence_to_graph(seq,tar,asin_num) for seq,tar in zip(train_data[0],train_data[1])]
    train_set = [sequence_to_graph(seq,seq[-1],asin_num) for seq in train_data]
    #with open("yoochoose-train.pkl", "wb") as f:
    #    pickle.dump(train_set, f)
    
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=False)
    
    print("building session embedding index")
    emb_list = []
    with torch.no_grad():
        for data in tqdm(train_loader):
            data = data.to(device)
            emb = all_product_head(session_encoder(data))
            #emb = session_encoder(data)
            emb = emb.detach().cpu().numpy()
            emb_list.append(emb)
    print('Finish computing embeddings')
    emb = np.concatenate(emb_list, axis=0)
    print(emb.shape)
    print('Building index')
    session_index = faiss.IndexFlatIP(emb.shape[1])
    session_index.add(normalize(emb))
    #faiss.write_index(session_index, "yoochoose_session_emb.faiss_index")

    print("building subsession embedding index")
    emb_list = []
    with torch.no_grad():
        for data in tqdm(train_loader):
            data = data.to(device)
            emb = next_product_head(subsession_encoder(data))
            #emb = subsession_encoder(data)
            emb = emb.detach().cpu().numpy()
            emb_list.append(emb)
    print('Finish computing embeddings')
    emb = np.concatenate(emb_list, axis=0)
    print(emb.shape)
    print('Building index')
    subsession_index = faiss.IndexFlatIP(emb.shape[1])
    subsession_index.add(normalize(emb))
    #faiss.write_index(subsession_index, "yoochoose_subsession_emb.faiss_index")
    
    print("querying by knn")
    ss_s_r = []
    ss_ss_r = []
    s_s_r = []
    with torch.no_grad():
        for data in tqdm(loader):
            #print(data['product'].x)
            #print(data['product_target'].y)
            data = data.to(device)
            emb = next_product_head(subsession_encoder(data))
            #emb = subsession_encoder(data)
            pred = get_prediction_by_knn(emb, session_index, train_set, 500, 20)
            _, r = get_p_r(set(data['product_target'].y.detach().cpu().tolist()), pred, 20)
            ss_s_r.append(r)

            pred = get_prediction_by_knn(emb, subsession_index, train_set, 500, 20)
            _, r = get_p_r(set(data['product_target'].y.detach().cpu().tolist()), pred, 20)
            ss_ss_r.append(r)

            emb = all_product_head(session_encoder(data))
            #emb = session_encoder(data)
            pred = get_prediction_by_knn(emb, session_index, train_set, 500, 20)
            _, r = get_p_r(set(data['product_target'].y.detach().cpu().tolist()), pred, 20)
            s_s_r.append(r)
           # break
    logging.info("Q: subsession, D: session, recall: %.3f"%(np.mean(ss_s_r)))
    logging.info("Q: subsession, D: subsession, recall: %.3f"%(np.mean(ss_ss_r)))
    logging.info("Q: session, D: session, recall: %.3f"%(np.mean(s_s_r)))

def build_index(emb, metric):
    print(emb.shape)
    #print('Building index')
    #res = faiss.StandardGpuResources()
    if metric == 'cos':
        index = faiss.IndexFlatIP(emb.shape[1])
        #index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(normalize(emb))
    elif metric == 'l2':
        index = faiss.IndexFlatL2(emb.shape[1])
        index.add(emb)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
    else:
        raise RuntimeError("Unregnozed metric", metric)
    return index


def get_future_map(I, test_data, train_data):
    map_list = []
    K = I.shape[1]
    for i in range(I.shape[0]):
        future = get_item(test_data[1][i])
        y_true = np.zeros(K)
        for j in range(K):
            session = get_item(train_data[I[i,j]])
            if len(set(session) & future) > 0:
                y_true[j] = 1
        if np.sum(y_true) == 0:
            map = 0
        else:
            y_scores = np.linspace(1,0,K)
            map = average_precision_score(y_true, y_scores)
        if map != map:
            map = 0
        map_list.append(map)
    return np.mean(map_list)

def get_all_map(I, test_data, train_data):
    map_list = []
    K = I.shape[1]
    for i in range(I.shape[0]):
        future = set(test_data[1][i]+test_data[0][i])
        y_true = np.zeros(K)
        for j in range(K):
            session = train_data[0][I[i,j]]
            if len(set(session) & future) > 0:
                y_true[j] = 1
        if np.sum(y_true) == 0:
            map = 0
        else:
            y_scores = np.linspace(1,0,K)
            map = average_precision_score(y_true, y_scores)
        if map != map:
            map = 0
        map_list.append(map)
    return np.mean(map_list)

def get_cur_map(I, test_data, train_data):
    map_list = []
    K = I.shape[1]
    for i in range(I.shape[0]):
        future = set(test_data[0][i])
        y_true = np.zeros(K)
        for j in range(K):
            session = train_data[0][I[i,j]]
            if len(set(session) & future) > 0:
                y_true[j] = 1
        if np.sum(y_true) == 0:
            map = 0
        else:
            y_scores = np.linspace(1,0,K)
            map = average_precision_score(y_true, y_scores)
        if map != map:
            map = 0
        map_list.append(map)
    return np.mean(map_list)

def get_cur_jaccard(I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):
        query = get_item(test_data[0][i])
        if len(query) == 0:
            continue
        for j in range(K):
            session = get_item(train_data[I[i,j]])
            jaccard = len(session&query) / len(session|query)
            jaccard_list.append(jaccard)
    return np.mean(jaccard_list)

def get_all_jaccard(I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):

        #query = get_item(test_data[0][i]+test_data[1][i])
        #if len(query) == 0:
        #    continue
        for j in range(K):
            jaccard_list.append(get_score((test_data[0][i], test_data[1][i]), (train_data[I[i,j]],[]), "all_jaccard"))
        #    session = get_item(train_data[I[i,j]])
        #    jaccard = len(session&query) / len(session|query)
        #    jaccard_list.append(jaccard)
    return np.mean(jaccard_list)

def get_all_jaccard_mse(D, I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):
        #query = get_item(test_data[0][i]+test_data[1][i])
        #if len(query) == 0:
        #    continue
        for j in range(K):
         #   session = get_item(train_data[I[i,j]])
         #   jaccard = len(session&query) / len(session|query)
         #   if jaccard >= 0.1 and jaccard < 0.7:
         #       jaccard = 0.7
         #   jaccard_list.append(jaccard)
            jaccard_list.append(get_score((test_data[0][i], test_data[1][i]), (train_data[I[i,j]],[]), "all_jaccard"))
    pickle.dump((D,jaccard_list), open("tmp_result.pkl", "wb"), protocol=4)
    return np.mean(np.abs(D.flatten()-np.array(jaccard_list)))

def get_future_jaccard(I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):
        query = get_item(test_data[1][i])
        if len(query) == 0:
            continue
        
        for j in range(K):
            session = get_item(train_data[I[i,j]])
            jaccard = len(session&query) / len(session|query)
            jaccard_list.append(jaccard)
    return np.mean(jaccard_list)

def get_cur_recall(I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):
        query = get_item(test_data[0][i])
        if len(query) == 0:
            continue
        for j in range(K):
            session = get_item(train_data[I[i,j]])
            jaccard = len(session&query) / len(query)
            jaccard_list.append(jaccard)
    return np.mean(jaccard_list)

def get_all_recall(I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):
        query = get_item(test_data[0][i]+test_data[1][i])
        if len(query) == 0:
            continue
        for j in range(K):
            session = get_item(train_data[I[i,j]])
            jaccard = len(session&query) / len(query)
            jaccard_list.append(jaccard)
    return np.mean(jaccard_list)

def get_future_recall(I, test_data, train_data):
    K = I.shape[1]
    jaccard_list = []
    for i in range(I.shape[0]):
        query = get_item(test_data[1][i])
        if len(query) == 0:
            continue
        for j in range(K):
            session = get_item(train_data[I[i,j]])
            jaccard = len(session&query) / len(query)
            jaccard_list.append(jaccard)
    return np.mean(jaccard_list)


def get_STAN_score(I, test_data, train_data, asin_num):
    K = I.shape[1]
    score_list = []
    for i in range(I.shape[0]):
        query = sequence_to_stan_vec(test_data[0][i], asin_num, CFG.STAN_lammy)
        norm = np.sqrt(len(test_data[0][i]))
        if norm == 0:
            continue
        query = query/norm
        query = csr_matrix(query)
        for j in range(K):
            session = sequence_to_binary_vec(train_data[I[i,j]], asin_num)
            session = session / np.sqrt(np.sum(session**2)+1e-6)
            score = query.dot(session)
            score_list.append(score)
    return np.mean(score_list)


def find_K_sparse_dense(sparse_data, dense_query, K):
    I = np.zeros((dense_query.shape[0],K),dtype=np.int32)
    D = np.zeros((dense_query.shape[0],K))
    for i in range(dense_query.shape[0]):
        query = dense_query[i,:]
        val = sparse_data.dot(query)
        val = np.squeeze(val)
        I[i,:] = np.argsort(val)[-K:][::-1]
        D[i,:] = np.sort(val)[-K:][::-1]
    return D,I



def get_query_metric(I, test_data, train_data, mode, metric):
    K = I.shape[1]
    val_list = []
    for i in range(I.shape[0]):
        if mode == 'all':
            query = get_query(test_data[0][i]+test_data[1][i], False)
        elif mode == 'cur':
            query = get_query(test_data[0][i], False)
        elif mode == 'future':
            query = get_query(test_data[1][i], False)
        else:
            raise RuntimeError("unrecognized mode", mode)
        if len(query) == 0:
            continue
        for j in range(K):
            session = get_query(train_data[I[i,j]], False)
            query_match_cnt, sess_match_cnt = get_string_match(query, session)
            if metric == 'score':
                if len(query)+len(session) == 0:
                    val = 0
                else:
                    val = float(query_match_cnt+sess_match_cnt)/ (len(query)+len(session))
            elif metric == 'recall':
                val = float(query_match_cnt) / len(query)
            val_list.append(val)
    return np.mean(val_list)

def get_recall(test_data, train_data, I, sim_type, thres):
    gt = np.zeros_like(I,dtype=np.float32)
    for i,t in enumerate(test_data):
        for j,d in enumerate(I[i,:]):
            r = train_data[d]
            score = get_score(t, (r,[]), sim_type)
            gt[i,j] = score
    return np.mean(np.sum(gt>thres,axis=1))/float(I.shape[1])

def main2(encoding_type, model_path=None):
    print(encoding_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device('cpu')
    print(device)
    asin_num = 391572
    lammy = 1.04
    K = 100
    #dir = "SavedModel/Yoochoose-STAN-HGGNN-SrGNNPooling-1/"
    train_data = pickle.load(open('/local2/data-1/raw/us-filtered-asin-train-data.pkl', 'rb'))
    #train_data = pickle.load(open('data/raw/us-filtered-split-train-data.pkl', 'rb'))
    #train_data = [a+b for a,b in zip(train_data[0], train_data[1])]
    #print(train_data[0])
    #print(train_data[1])
    test_data = pickle.load(open('/local2/data-1/raw/us-filtered-split-test-data.pkl', 'rb'))
    
    if encoding_type == 'model':
        
        print(model_path)
        #data_encoder = torch.load(model_path)[1]
        data_encoder = torch.load(model_path)[0]
        query_encoder = torch.load(model_path)[0]
        #subsession_encoder, head = torch.load(model_path)
        data_encoder.to(device)
        query_encoder.eval()
        data_encoder.to(device)
        query_encoder.eval()
        #head.to(device)
        #head.eval()
        #data = [sequence_to_stan_vec(seq ,asin_num, lammy) for seq in train_data[0]]
        tokenizer = AutoTokenizer.from_pretrained('./SavedModel/QAEA')
#        train_data = train_data[:100]    
        #train_list = [sequence_to_graph(0, seq , seq[-1:], tokenizer, CFG.query_max_len, False) for seq in train_data]
        train_list = [sequence_to_graph(0, seq+tar, tar, tokenizer, CFG.query_max_len, False) for seq, tar in zip(test_data[0], test_data[1])]

        #train_list = pickle.load(open("data/filtered_amazon_data_list.pkl", "rb"))
        train_loader = DataLoader(train_list, batch_size=200, shuffle=False)

        print("building subsession embedding index")
        emb_list = []
        print("Building Index...")
        with torch.no_grad():
            for data in tqdm(train_loader):
                data = data.to(device)
                #emb = next_product_head(subsession_encoder(data))
                #emb = subsession_encoder(data)
                emb = data_encoder(data)
                #emb = head(subsession_encoder(data))
                emb = emb.detach().cpu().numpy()
                emb_list.append(emb)
        print('Finish computing embeddings')
        data = np.concatenate(emb_list, axis=0)
        pickle.dump(data, open('full_query_emb.pkl','wb'), protocol=4)
        
        #pickle.dump(data, open('tmp_data.pkl', 'wb'), protocol=4)
        #data = pickle.load(open('tmp_data.pkl', 'rb'))
        """
        tokenizer = AutoTokenizer.from_pretrained("SavedModel/QAEA")
        model = AutoModel.from_pretrained("SavedModel/QAEA", add_pooling_layer=False)
        model = model.to(device)
        print('get tokenizer and model')
        #text = train_list[0]
        #input = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
        #print(input)
        #model(**input)
        def tokenize(text, tokenizer):
            tokens = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
            data = HeteroData()
            data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
            return data
        
        train_list = [tokenize(session_to_text(session), tokenizer) for session in tqdm(train_data)]
        train_loader = DataLoader(train_list, batch_size=CFG.batch_size, shuffle=False)

        emb_list = []
        with torch.no_grad():
            for data in tqdm(train_loader):
                data = data.to(device)
                input, type_id, mask = data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x
                emb = model(input_ids=input, token_type_ids = type_id, attention_mask=mask).last_hidden_state
                emb = torch.mean(emb,dim=1)
                emb = global_mean_pool(emb, data['input_ids'].batch)
                emb = emb.detach().cpu().numpy()
                emb_list.append(emb)
        print('Finish computing embeddings')
        emb_qaea = np.concatenate(emb_list, axis=0)
        data = np.concatenate([emb_qaea, emb_model],axis=1)
        print(data.shape)
        """
        subsession_index = build_index(data, 'cos')
        print('done index')
        print('memory cost: ', sys.getsizeof(data))
        # subsession_index = faiss.IndexFlatL2(emb.shape[1])
        # subsession_index.add(emb)
        query_list = [sequence_to_graph(0, seq, tar, tokenizer, CFG.query_max_len, False) for seq, tar in zip(test_data[0], test_data[1])]
        query_loader = DataLoader(query_list, batch_size=200, shuffle=False)
        emb_list = []
        with torch.no_grad():
            for data in tqdm(query_loader):
                data = data.to(device)
                #emb = head(subsession_encoder(data))
                emb = query_encoder(data)
                emb = emb.detach().cpu().numpy()
                emb_list.append(emb)
        print('Finish computing embeddings')
        emb = np.concatenate(emb_list, axis=0)
        pickle.dump(emb, open('query_emb.pkl','wb'), protocol=4)
        os._exit(1)

        """
        query_list = [tokenize(session_to_text(session), tokenizer) for session in test_data[0]]
        query_loader = DataLoader(query_list, batch_size=CFG.batch_size, shuffle=False)
        emb_list = []
        with torch.no_grad():
            for data in tqdm(query_loader):
                data = data.to(device)
                input, type_id, mask = data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x
                emb = model(input_ids=input, token_type_ids = type_id, attention_mask=mask).last_hidden_state
                emb = torch.mean(emb,dim=1)
                emb = global_mean_pool(emb, data['input_ids'].batch)
                emb = emb.detach().cpu().numpy()
                emb_list.append(emb)
        emb_qaea = np.concatenate(emb_list, axis=0)
        emb = np.concatenate((emb_qaea, emb_model),axis=1)
        """
        start = time.perf_counter()
        D, I = subsession_index.search(normalize(emb), K)
        print('search time: %d'%(time.perf_counter()-start))  
        pickle.dump((D,I), open('result.pkl','wb'),protocol=4)

    elif encoding_type == 'STAN' or encoding_type == 'SKNN':
        print('sknn...')
        data = [csr_matrix(normalize(sequence_to_binary_vec(seq ,asin_num).astype('float32'))) for seq in train_data]
        #data = data.astype('float32')
        #data = normalize(data)
        #data = csr_matrix(data)
        data = vstack(data)
        
        print('memory cost: ', data.data.nbytes)
        os._exit(0)
        print(data.shape)
        print('finish data')
        if encoding_type == 'STAN':
            emb = np.array([sequence_to_stan_vec(seq, asin_num, CFG.STAN_lammy) for seq in test_data[0]]).astype('float32')
        else:
            emb = np.array([sequence_to_binary_vec(seq, asin_num) for seq in test_data[0]]).astype('float32')
        emb = normalize(emb)
        print('start search')
        start = time.perf_counter()
        D, I = find_K_sparse_dense(data, emb, K)
        print('search time: %d'%(time.perf_counter()-start)) 
        print('finish search')
        
    elif encoding_type == 'load':
        path = model_path
        D,I = pickle.load(open(path,"rb"))
    elif encoding_type == 'QAEA':
        tokenizer = AutoTokenizer.from_pretrained("SavedModel/QAEA")
        model = AutoModel.from_pretrained("SavedModel/QAEA", add_pooling_layer=False)
        model = model.to(device)
        print('get tokenizer and model')
        #text = train_list[0]
        #input = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
        #print(input)
        #model(**input)
        def tokenize(text, tokenizer):
            tokens = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
            data = HeteroData()
            data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
            return data
        
        train_list = [tokenize(session_to_text(session), tokenizer) for session in tqdm(train_data)]
        train_loader = DataLoader(train_list, batch_size=CFG.batch_size, shuffle=False)

        emb_list = []
        with torch.no_grad():
            for data in tqdm(train_loader):
                data = data.to(device)
                input, type_id, mask = data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x
                emb = model(input_ids=input, token_type_ids = type_id, attention_mask=mask).last_hidden_state
                emb = torch.mean(emb,dim=1)
                emb = global_mean_pool(emb, data['input_ids'].batch)
                emb = emb.detach().cpu().numpy()
                emb_list.append(emb)
        print('Finish computing embeddings')
        data = np.concatenate(emb_list, axis=0)
        print(data.shape)
        #print('start building index')
        #data = np.concatenate(data,axis=0)
        subsession_index = build_index(data, 'cos')
        print('memory cost: ', sys.getsizeof(data))
        print('done')

        query_list = [tokenize(session_to_text(session), tokenizer) for session in test_data[0]]
        query_loader = DataLoader(query_list, batch_size=CFG.batch_size, shuffle=False)
        emb_list = []
        with torch.no_grad():
            for data in tqdm(query_loader):
                data = data.to(device)
                input, type_id, mask = data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x
                emb = model(input_ids=input, token_type_ids = type_id, attention_mask=mask).last_hidden_state
                emb = torch.mean(emb,dim=1)
                emb = global_mean_pool(emb, data['input_ids'].batch)
                emb = emb.detach().cpu().numpy()
                emb_list.append(emb)
        emb = np.concatenate(emb_list, axis=0)
        print(emb.shape)
        print('search')
        start = time.perf_counter()
        D, I = subsession_index.search(emb, K)
        print('search time: %d'%(time.perf_counter()-start)) 
        print('done')
        
    else:
        raise RuntimeError("unrecognized encoding type", encoding_type)
    pickle.dump((D,I), open(encoding_type+'_test_DI.pkl', 'wb'), protocol=4)
    
    test_data = [(a,b) for a,b in zip(test_data[0], test_data[1])]
    print(get_ave_score(I, test_data, train_data, 'all_product_type_score'))
    print(get_ave_score(I, test_data, train_data, 'all_jaccard'))
    print(get_ave_score(I, test_data, train_data, 'all_product_title_score'))
    print(get_ave_score(I, test_data, train_data, 'all_query_score'))
    """
    gt = np.zeros_like(I,dtype=np.float32)
    for i,t in enumerate(test_data):
        for j,d in enumerate(I[i,:]):
            r = train_data[d]
            score = get_score(t, (r,[]), 'all_product_type_score')
            gt[i,j] = score
    print('mean',np.mean(gt))
        
        
    gt = np.zeros_like(I,dtype=np.float32)
    for i,t in enumerate(test_data):
        for j,d in enumerate(I[i,:]):
            r = train_data[d]
            score = get_score(t, (r,[]), 'all_query_score')
            gt[i,j] = score
    print('mean',np.mean(gt))
    """
     
#    logging.info("Subsession Embedding")
    # get query embeddings
    """
    print(get_all_jaccard_mse(D, I, test_data, train_data))
    #os._exit(0)
    #D,I = pickle.load(open("yoochoose_"+encoding_type+".result","rb"))
    #map = get_cur_map(I, test_data, train_data)
    
    #logging.info('cur map: %.5f'%(map))
    #print('cur map: %.5f'%(map))
    
    map = get_future_map(I, test_data, train_data)
    
    logging.info('future map: %.5f'%(map))
    print('future map: %.5f'%(map))
    
    #map = get_all_map(I, test_data, train_data)
    #logging.info('all map: %.5f'%(map))
    #print('all map: %.5f'%(map))

    stan_score = get_STAN_score(I, test_data, train_data, asin_num)
    logging.info('STAN score: %.5f'%(stan_score))
    print('STAN score: %.5f'%(stan_score))

    cur_jaccard = get_cur_jaccard(I, test_data, train_data)
    logging.info('cur jaccard: %.5f'%(cur_jaccard))
    print('cur jaccard: %.5f'%(cur_jaccard))

    all_jaccard = get_all_jaccard(I, test_data, train_data)
    logging.info('all jaccard: %.5f'%(all_jaccard))
    print('all jaccard: %.5f'%(all_jaccard))

    future_jaccard = get_future_jaccard(I, test_data, train_data)
    logging.info('future jaccard: %.5f'%(future_jaccard))
    print('future jaccard: %.5f'%(future_jaccard))

    cur_jaccard = get_cur_recall(I, test_data, train_data)
    logging.info('cur recall: %.5f'%(cur_jaccard))
    print('cur recall: %.5f'%(cur_jaccard))

    all_jaccard = get_all_recall(I, test_data, train_data)
    logging.info('all recall: %.5f'%(all_jaccard))
    print('all recall: %.5f'%(all_jaccard))

    future_jaccard = get_future_recall(I, test_data, train_data)
    logging.info('future recall: %.5f'%(future_jaccard))
    print('future recall: %.5f'%(future_jaccard))

    score = get_query_metric(I, test_data, train_data, "all", "score")
    print('all query score:', score)

    score = get_query_metric(I, test_data, train_data, "cur", "score")
    print('cur query score:', score)

    score = get_query_metric(I, test_data, train_data, "future", "score")
    print('future query score:', score)

    score = get_query_metric(I, test_data, train_data, "all", "recall")
    print('all query recall:', score)

    score = get_query_metric(I, test_data, train_data, "cur", "recall")
    print('cur query recall:', score)

    score = get_query_metric(I, test_data, train_data, "future", "recall")
    print('future query recall:', score)
    """
    """
    map_list = []
    for i in range(emb.shape[0]):
        future = set(test_data[1][i]+test_data[0][i])
        #future = set(test_data[0][i])
        y_true = np.zeros(K)
        
        for j in range(K):
            session = train_data[0][I[i,j]]
            if len(set(session) & future) > 0:
                y_true[j] = 1
        if np.sum(y_true) == 0:
            map = 0
        else:
            y_scores = np.linspace(1,0,K)
            map = average_precision_score(y_true, y_scores)
        #map = 0
        if map != map:
            map = 0
        map_list.append(map)
        
    logging.info('IP all: %.5f'%(np.mean(map_list)))
    print('all: %.5f'%(np.mean(map_list)))

    logging.info("Session Embedding")
    print("building session embedding index")
    emb_list = []
    with torch.no_grad():
        for data in tqdm(train_loader):
            data = data.to(device)
            emb = all_product_head(session_encoder(data))
            #emb = session_encoder(data)
            emb = emb.detach().cpu().numpy()
            emb_list.append(emb)
    print('Finish computing embeddings')
    emb = np.concatenate(emb_list, axis=0)
    print(emb.shape)
    print('Building index')
    session_index = faiss.IndexFlatIP(emb.shape[1])
    session_index.add(normalize(emb))
    #session_index = faiss.IndexFlatL2(emb.shape[1])
    #session_index.add(emb)
    #faiss.write_index(session_index, "yoochoose_session_emb.faiss_index")
    start = time.time()
    emb_list = []
    with torch.no_grad():
        for data in tqdm(query_loader):
            data = data.to(device)
            emb = all_product_head(session_encoder(data))
            #emb = session_encoder(data)
            emb = emb.detach().cpu().numpy()
            emb_list.append(emb)
    print('Finish computing embeddings')
    emb = np.concatenate(emb_list, axis=0)
    
    print('encode time', time.time()-start)
    start = time.time()
    D, I = session_index.search(emb, K)
    
    print('search time', time.time()-start)
    map_list = []
    
    for i in range(emb.shape[0]):
        future = set(test_data[1][i])
        y_true = np.zeros(K)
        for j in range(K):
            session = train_data[0][I[i,j]]
            if len(set(session) & future) > 0:
                y_true[j] = 1
        if np.sum(y_true) == 0:
            map = 0
        else:
            y_scores = np.linspace(1,0,K)
            map = average_precision_score(y_true, y_scores)
        if map != map:
            map = 0
        map_list.append(map)
    logging.info('IP future: %.5f'%(np.mean(map_list)))
    print('future: %.5f'%(np.mean(map_list)))
    
    map_list = []
    for i in range(emb.shape[0]):
        future = set(test_data[1][i]+test_data[0][i])
        #future = set(test_data[0][i])
        y_true = np.zeros(K)
        
        for j in range(K):
            session = train_data[0][I[i,j]]
            if len(set(session) & future) > 0:
                y_true[j] = 1
        if np.sum(y_true) == 0:
            map = 0
        else:
            y_scores = np.linspace(1,0,K)
            map = average_precision_score(y_true, y_scores)
        #map = 0
        if map != map:
            map = 0
        map_list.append(map)
        
    logging.info('IP all: %.5f'%(np.mean(map_list)))
    print('all: %.5f'%(np.mean(map_list)))
    
    """
if __name__ == '__main__':
    #dir = "SavedModel/Yoochoose-STAN-HGGNN-SrGNNPooling-1/"
    dir = CFG.savedir
    print(normalize(np.ones(4)))
    logging.basicConfig(filename =  dir+"test.log",
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    #main()
    #main2('model', 'SavedModel/FilteredAmazon-CL-02-next-all-HGGNN-SrGNNPooling-pt-pid-fq-ctv/pretrain_model.pth')
    main2('model', 'SavedModel/debug-no-cross-transformer-with-node/pretrain_model.pth')
    #main2("SKNN")
    
    #main2("QAEA")
    #main2("load", 'SKNN_test_DI.pkl')
