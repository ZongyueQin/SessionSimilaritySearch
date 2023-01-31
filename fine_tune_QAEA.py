from asyncio.proactor_events import _ProactorBasePipeTransport
from collections import defaultdict
from tarfile import ExFileObject
from tkinter import OptionMenu
from model.model import MLP, BinarizeHead, QAEA_Linear
from model.NodeEmbedding import NodeAsinEmbedding, NodeTextTransformer, PretrainedQAEAEncoder
from model.gnn import get_hetero_GNN, GraphPooling, HGT, HeteroGGNN, AttentionPooling, SRGNN_Pooling
from torch_geometric.loader import DataLoader
from DataLoader import MyDataLoader
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import numpy as np
from transformers import AutoTokenizer, AutoModel
from config import CFG
import pickle
import logging
from tqdm import tqdm
import torch.nn.functional as F
import os
import shutil
from torch_geometric.data import InMemoryDataset
import torch.nn as nn
import psutil
from util_amazon_filtered import get_item, normalize, sequence_to_graph, get_query, get_string_match, get_session_item_title, get_item_type, binary_regularize, session_to_text
from pretrain_filtered_amazon import FilteredAmazonDataset
import random
import copy
import faiss
from multiprocessing import Pool
import Levenshtein
import time
import sys
from fine_tune_filtered_amazon import get_ave_score

os.environ['CUDA_VISIBLE_DEVICES']='2'


def get_score(data_a, data_b, sim_type):
    #print(data_b)
    if sim_type == 'all_jaccard':
        a_item = get_item(data_a[0]+data_a[1])
        b_item = get_item(data_b[0]+data_b[1])
        score = len(a_item & b_item) / len(a_item | b_item)
    elif sim_type == 'cur_jaccard':
        a_item = get_item(data_a[0])
        b_item = get_item(data_b[0])
        c = len(a_item | b_item)
        if c == 0:
            score = 0
        else:
            score = len(a_item & b_item) / c
    elif sim_type == 'all_query_score':
        a_query = get_query(data_a[0]+data_a[1], pad=False)
        b_query = get_query(data_b[0]+data_b[1], pad=False)
        if len(a_query) == 0 or len(b_query) == 0:
            return 0
        score =  Levenshtein.seqratio(a_query, b_query)
    elif sim_type =='all_product_title_score':
        a_item = get_session_item_title(data_a[0]+data_a[1])
        b_item = get_session_item_title(data_b[0]+data_b[1])
        score = Levenshtein.seqratio(a_item, b_item)
    elif sim_type == 'all_product_type_score':
        a_type = get_item_type(data_a[0]+data_a[1])
        b_type = get_item_type(data_b[0]+data_b[1])
        vec_len = len(set(a_type+b_type))
        type_to_id = {}
        a_vec = np.zeros(vec_len)
        b_vec = np.zeros(vec_len)
        for t in a_type:
            if t not in type_to_id:
                type_to_id[t] = len(type_to_id)
            a_vec[type_to_id[t]] += 1
        if len(a_type) > 0:
            a_vec = a_vec / np.linalg.norm(a_vec)
        for t in b_type:
            if t not in type_to_id:
                type_to_id[t] = len(type_to_id)
            b_vec[type_to_id[t]] += 1
        if len(b_type) > 0:
            b_vec = b_vec / np.linalg.norm(b_vec)
        score = np.sum(a_vec*b_vec)
    else:
        raise RuntimeError("unrecognized sim type: $s"%sim_type)
    return score

def get_ave_score(I, test_data, train_data, sim_type):
    gt = np.zeros_like(I,dtype=np.float32)
    for i,t in enumerate(test_data):
        for j,d in enumerate(I[i,:]):
            r = train_data[d]
            score = get_score(t, (r,[]), sim_type)
            gt[i,j] = score
    return np.mean(gt)

def get_loss(out, data, loss_type, sim_type, train_data, device):
    ori_data = [(train_data[0][i], train_data[1][i]) for i in data['idx'].idx]
    norm_out = F.normalize(out)
    cos = norm_out @ norm_out.T
    #pred = (cos + 1) / 2
    pred = cos
    label = torch.zeros_like(cos).to(device)
    
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == "L1":
        criterion = nn.L1Loss()
    else:
        raise RuntimeError("unrecognized loss type "+loss_type)
   
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            label[i,j] = get_score(ori_data[i], ori_data[j], sim_type)
    
    weight = torch.sqrt(torch.where(label > 0, 10, 1).to(device).detach())
    return criterion(pred*weight, label*weight), pred, label

    

def get_pair_loss(out1, out2, lab, loss_type, device, reg=True, mat = False):
    
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    elif loss_type == "L1":
        criterion = nn.L1Loss()
    else:
        raise RuntimeError("unrecognized loss type "+loss_type)
    if reg == True:
        #pred = F.cosine_similarity(out1, out2, dim=1)
        pred = F.normalize(out1) @ F.normalize(out2).T
        tgt = torch.diag(lab).to(device)
        weight = 0.001*torch.ones_like(pred.cpu()) + 0.999 * torch.diag(torch.ones(CFG.batch_size))
        #print(weight)
        weight = torch.sqrt(weight.to(device))
    else:
        pred = F.cosine_similarity(out1, out2, dim=1)
        tgt = lab.float()
        weight = 1
    if mat == False:
        #print(tgt*weight)
        #print(pred*weight)
        return criterion(tgt*weight, pred*weight)
    else:
        return criterion(tgt*weight, pred*weight), pred*weight, tgt*weight

def get_triplet_loss(out, pos_out, neg_out, pos_score, neg_score):
    pos_pred = F.cosine_similarity(out, pos_out)
    neg_pred = F.cosine_similarity(out, neg_out)
    loss = torch.mean(torch.clip(neg_pred - pos_pred + (pos_score-neg_score), min=0))
    return loss
    
def main():
    if not os.path.exists(CFG.savedir):
        os.makedirs(CFG.savedir)
    shutil.copy(__file__, CFG.savedir)
    shutil.copy("./config.py", CFG.savedir)
    logging.basicConfig(filename =  CFG.savedir+"/fine-tune-%s-%s.log"%(CFG.loss_type, CFG.sim_type),
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    asin_num = 391572
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    tokenizer = AutoTokenizer.from_pretrained('./SavedModel/QAEA')
    def tokenize(text, tokenizer):
        tokens = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
        data = HeteroData()
        data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
        return data
    """Sample Fine Tune Data"""
    """
    query_data = pickle.load(open('data/raw/us-filtered-split-train-data.pkl', 'rb'))
    query_data = [(a,b) for a,b in zip(query_data[0], query_data[1])]
    db_data = pickle.load(open('data/raw/us-filtered-asin-train-data.pkl', 'rb'))
    db_data = [(a,[]) for a in db_data]
    #tgt_data = pickle.load(open('data/raw/us-filtered-asin-test-data.pkl', 'rb'))
    #tgt_data = [(a,[]) for a in tgt_data]
    #query_data = pickle.load(open('data/raw/us-filtered-split-test-data.pkl', 'rb'))
    #query_data = [(a,b) for a,b in zip(query_data[0], query_data[1])]
    
    print(len(db_data))
    random.shuffle(query_data)
    random.shuffle(db_data)
    data_tuple = []
    
    #pbar = tqdm(total=len(db_data))
    
    pos = 0
    
    while len(data_tuple) < CFG.fine_tune_data_num and pos < len(query_data):
    #for pos in tqdm(range(len(query_data))):
        #if len(data_tuple) % 10 == 0:
        print(len(data_tuple))
        ori_data = query_data[pos]
        pos += 1
        #score_list = []
        pos_data = None
        half_pos_data = None
        neg_data = None
        cnt = 0
        #pbar = tqdm(total=len(db_data))
        for i, data in enumerate(db_data):
            #pbar.update(1)
            score = get_score(ori_data, data, CFG.sim_type)
            #score_list.append(score)
            # 0.4 for product title
            if score >= 0.8 and pos_data is None:
                pos_data = data
                pos_idx = i
                #half_pos_idx = i
                pos_score = score
                #half_pos_score = score
                cnt += 1
                #print('pos', score,i)
                #half_pos_data = pos_data
            if score < 0.8 and score >= 0.2 and half_pos_data is None:
                half_pos_data = data
                half_pos_score = score
                half_pos_idx = i
                cnt += 1
                #print('h', score,i)
            # 0.1 for product title
            if score < 0.2 and neg_data is None:
                neg_data = data
                cnt += 1
                neg_idx = i
                neg_score = score
                #print('n', score,i)
            if cnt == 3:
                break
        if cnt == 3:
            ori_data_g = sequence_to_graph(0, ori_data[0], ori_data[1], tokenizer, CFG.query_max_len)
            pos_data_g = sequence_to_graph(pos_idx, pos_data[0], pos_data[1], tokenizer, CFG.query_max_len)
            half_pos_data_g = sequence_to_graph(half_pos_idx, half_pos_data[0], half_pos_data[1], tokenizer, CFG.query_max_len)
            neg_data_g = sequence_to_graph(neg_idx, neg_data[0], neg_data[1], tokenizer, CFG.query_max_len)
            
            data_tuple.append((ori_data_g, pos_data_g, half_pos_data_g, neg_data_g, pos_score, half_pos_score, neg_score))
        #pbar.update(1)
        #print(np.min(score_list))
        #if len(data_tuple) == 100:
        #    pickle.dump(data_tuple, open("asym_t_fine_tune_data_tuple_%s_100.pkl"%(CFG.sim_type), 'wb'), protocol=4)
        #if len(data_tuple) == 3000:
        #    pickle.dump(data_tuple, open("asym_fine_tune_data_tuple_%s_3000.pkl"%(CFG.sim_type), 'wb'), protocol=4)
        #if len(data_tuple) == 10000:
        #    pickle.dump(data_tuple, open("asym_fine_tune_data_tuple_%s_10000.pkl"%(CFG.sim_type), 'wb'), protocol=4)
        #    break
        #if len(data_tuple) == 500:
        #    pickle.dump(data_tuple, open("fine_tune_data_tuple_%s_500.pkl"%(CFG.sim_type), 'wb'), protocol=4)
        #if len(data_tuple) == 1000:
        #    pickle.dump(data_tuple, open("fine_tune_data_tuple_%s_1000.pkl"%(CFG.sim_type), 'wb'), protocol=4)
        #if len(data_tuple) == 3000:
        #    pickle.dump(data_tuple, open("fine_tune_data_tuple_%s_3000.pkl"%(CFG.sim_type), "wb"), protocol=4)
    
#        if len(data_tuple) == 10:
#            break
    pickle.dump(data_tuple, open("asym_fine_tune_data_tuple_%s.pkl"%(CFG.sim_type), "wb"), protocol=4)
    print('finish getting pairs', len(data_tuple))
    os._exit(-1)
    """
    query_data = pickle.load(open('data/raw/us-filtered-split-train-data.pkl', 'rb'))
    print('transform')
    query_data = [(a,b) for a,b in zip(query_data[0], query_data[1])]
    print('sample')
    query_data = random.sample(query_data,5000)
    print('transform')
    aux_data_list = [(tokenize(session_to_text(seq), tokenizer),
                        tokenize(session_to_text(seq+tar), tokenizer)) for seq,tar in query_data]
    aux_data_loader = DataLoader(aux_data_list, batch_size=CFG.batch_size, shuffle=True)
    del query_data
    print('load')
    try:
        data_tuple = pickle.load(open("asym_fine_tune_data_tuple_%s_10000.pkl"%CFG.sim_type, "rb"))
    except Exception as e:
        data_tuple = pickle.load(open("asym_fine_tune_data_tuple_%s.pkl"%CFG.sim_type, "rb"))
    
    def tfm(data):
        return tokenize(session_to_text(data['ori_seq'][0]), tokenizer)
    data_tuple = [(tfm(a), tfm(b), tfm(c), tfm(d), e, f, g) for a,b,c,d,e,f,g in data_tuple]
    
    valid_num = len(data_tuple)//4
    train_list, valid_list = torch.utils.data.random_split(data_tuple, [len(data_tuple)-valid_num, valid_num])
    print('split')
    #del data_tuple
    
    train_loader = MyDataLoader(train_list, batch_size=CFG.ft_batch_size, shuffle=True)
    valid_loader = MyDataLoader(valid_list, batch_size=CFG.batch_size, shuffle=False)

    test_tuple = pickle.load(open("asym_test_fine_tune_data_tuple_%s.pkl"%(CFG.sim_type), "rb"))
    test_tuple = [(tfm(a), tfm(b), tfm(c), tfm(d), e, f, g) for a,b,c,d,e,f,g in test_tuple]
    
    test_loader = MyDataLoader(test_tuple, batch_size=CFG.batch_size, shuffle=False)

    model = QAEA_Linear(None)
    query_model = copy.deepcopy(model)
    qaea = QAEA_Linear(None)
    gnn_out = 1000
    if CFG.code_len > 0:
        #bin_mlp = MLP(768, 1200, 1200, 0, dropout=0)
        bin_mlp = nn.Linear(768, 1200)
        model_bin_head = BinarizeHead(1200, CFG.code_len, bin_mlp)
        q_bin_mlp = nn.Linear(768, 1200)#
        #q_bin_mlp = MLP(768, 1200, 1200, 0, dropout=0)
        qmodel_bin_head = BinarizeHead(1200, CFG.code_len, q_bin_mlp)
        model_bin_head.to(device)
        qmodel_bin_head.to(device)
    #sim_head = MLP(gnn_out, gnn_out, gnn_out, 2, 0)
    ##sim_head =  nn.Sequential(nn.Linear(gnn_out, gnn_out),
    #            nn.ReLU(),
    #            nn.Linear(gnn_out, gnn_out))
    decode_n_input = gnn_out
    if CFG.code_len > 0:
        decode_n_input = CFG.code_len
    model_decode_head = MLP(decode_n_input, 768, 1200,1,0, last_act=False)
    q_model_decode_head = MLP(decode_n_input, 768, 1200,1,0, last_act=False)
    model_decode_head.to(device)
    q_model_decode_head.to(device)
    model.to(device)
    qaea.to(device)
    #sim_head.to(device)
    query_model.to(device)
    #model_db.to(device)
    
    #all_params = list(model.parameters())+list(sim_head.parameters())
    all_params = list(model.parameters())+list(query_model.parameters())+list(model_decode_head.parameters())+list(q_model_decode_head.parameters())
    if CFG.code_len > 0:
        all_params += list(model_bin_head.parameters())+list(qmodel_bin_head.parameters())
    #all_params = sim_head.parameters()
    if CFG.code_len > 0:
        optimizer1 = torch.optim.Adam(list(model.parameters())+list(model_bin_head.parameters()), lr=CFG.lr, weight_decay=0)
        optimizer2 = torch.optim.Adam(list(query_model.parameters())+list(qmodel_bin_head.parameters()), lr=CFG.lr, weight_decay=0)
        optimizer3 = torch.optim.Adam(list(model_decode_head.parameters())+list(q_model_decode_head.parameters()))
    else:
        optimizer1 = torch.optim.Adam(list(model.parameters()), lr=CFG.lr, weight_decay=0)
        optimizer2 = torch.optim.Adam(list(query_model.parameters()), lr=CFG.lr, weight_decay=0)
        optimizer3 = torch.optim.Adam(list(model_decode_head.parameters())+list(q_model_decode_head.parameters()))
    #max_iter = len(train_loader) // 10
    best_valid_loss = 1000
    
    for epoch in range(CFG.fine_tune_epoch):
        
        print(CFG.savedir, CFG.sim_type)
        # training
        training_loss = []        
        train_reg_loss = []
        model.train()
        query_model.train()
        model_decode_head.train()
        q_model_decode_head.train()
        if CFG.code_len > 0:
            print('hashing')
            model_bin_head.train()
            qmodel_bin_head.train()
#        sim_head.train()
        reg_mat = torch.eye(CFG.batch_size).float().to(device)
        reg_w = 0.1*torch.ones_like(reg_mat) + 0.9*reg_mat
        for i, data in enumerate(tqdm(train_loader)):
            if i % 2 == 0:
                #optimizer1.step()
                model.train()
                query_model.eval()
                if CFG.code_len > 0:
                    model_bin_head.train()
                    qmodel_bin_head.eval()
            else:
                #optimizer2.step()
                query_model.train()
                model.eval()
                if CFG.code_len > 0:
                    model_bin_head.eval()
                    qmodel_bin_head.train()

            if i == len(train_loader) - 1 or i == 1:
                print(torch.mean(torch.abs(aux_qaea_emb)), torch.mean(torch.abs(aux_sub_qaea_emb)))
                #print(np.mean(training_loss), np.mean(train_reg_loss))
                print(np.mean(training_loss, axis=0))
                #print(torch.mean(pos_score.float()).item(),torch.mean(half_pos_score.float()).item(),torch.mean(neg_score.float()).item())
                #print(reg_loss.item())
                #print(mat)
            ori_data = data[0].to(device)
            pos_data = data[1].to(device)
            half_pos_data = data[2].to(device)
            neg_data = data[3].to(device)
            pos_score = data[4].to(device)
            half_pos_score = data[5].to(device)
            neg_score = data[6].to(device)
            #p_data = data[7].to(device)

            aux_sub_data, aux_data = next(iter(aux_data_loader))
            aux_sub_data.to(device)
            aux_data.to(device)

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()
            try:
                """
                #ori_out = sim_head(model(ori_data))
                #pos_out = sim_head(model(pos_data))
                ori_out = model(ori_data)
                pos_out = tgt_model(pos_data)
                pos_loss = get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device, False)

                half_pos_out = tgt_model(half_pos_data)
                #half_pos_out = sim_head(model(half_pos_data))
                half_pos_loss = get_pair_loss(ori_out, half_pos_out, half_pos_score, CFG.loss_type, device, False)

                #neg_out = sim_head(model(neg_data))
                neg_out = tgt_model(neg_data)
                neg_loss = get_pair_loss(ori_out, neg_out, neg_score, CFG.loss_type, device, False)

                #ori_out_2 = tgt_model(ori_data)
                #mat = F.normalize(ori_out) @ F.normalize(ori_out_2).T
                #reg_loss = torch.mean(torch.abs(mat-reg_tgt))

                loss = pos_loss + half_pos_loss + 0.1*neg_loss 
                """
                ori_out = query_model(ori_data)
                pos_out = model(pos_data)
                
                half_pos_out = model(half_pos_data)
                
                #neg_out = sim_head(model(neg_data))
                neg_out = model(neg_data)
                
                
                
                #p_out = model(p_data)
                #ori_out_2 = tgt_model(ori_data)
                #mat = F.normalize(ori_out) @ F.normalize(ori_out_2).T
                #reg_loss = torch.mean(torch.abs(mat-reg_tgt))

                aux_sub_out = query_model(aux_sub_data)
                aux_out = model(aux_data)

                if CFG.code_len > 0:
                    ori_out = qmodel_bin_head(ori_out)
                    pos_out = model_bin_head(pos_out)
                    half_pos_out = model_bin_head(half_pos_out)
                    neg_out = model_bin_head(neg_out)
                    aux_sub_out = qmodel_bin_head(aux_sub_out)
                    aux_out = model_bin_head(aux_out)
                    reg_loss = binary_regularize(ori_out)+binary_regularize(pos_out)+binary_regularize(half_pos_out)+binary_regularize(neg_out)+binary_regularize(aux_sub_out)+binary_regularize(aux_out)
                else:
                    reg_loss = 0*torch.sum(ori_out)
                aux_pred = F.normalize(aux_sub_out) @ F.normalize(aux_out).T
                aux_loss = torch.sum(reg_w*torch.abs(aux_pred-reg_mat))/torch.sum(reg_w)

                #loss = get_triplet_loss(ori_out, pos_out, half_pos_out, pos_score, half_pos_score) + get_triplet_loss(ori_out, half_pos_out, neg_out, half_pos_score, neg_score)
                loss = 1*get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device, False)+\
                     1*get_pair_loss(ori_out, neg_out, neg_score, CFG.loss_type, device, False)+\
                    get_pair_loss(ori_out, half_pos_out, half_pos_score, CFG.loss_type, device, False)+\
                        CFG.aux_w*aux_loss + CFG.bin_w*reg_loss

                #loss = 0
                rec_aux_sub_qaea = q_model_decode_head(aux_sub_out)
                rec_aux_qaea = model_decode_head(aux_out)
                aux_sub_qaea_emb = qaea(aux_sub_data)
                aux_qaea_emb = qaea(aux_data)
                #gt = F.normalize(aux_sub_qaea_emb) @ F.normalize(aux_qaea_emb).T
                #pred = F.normalize(aux_sub_out) @ F.normalize(aux_out).T
                #reg_loss = binary_regularize(aux_sub_out)+binary_regularize(aux_out)
                #loss = torch.mean((gt-pred)**2) #+ CFG.bin_w*reg_loss
                #loss = torch.mean(torch.abs(gt-pred))
                #loss = 0
                rec_loss = torch.mean(torch.sum((aux_sub_qaea_emb-rec_aux_sub_qaea)**2,dim=1))+torch.mean(torch.sum((aux_qaea_emb-rec_aux_qaea)**2, dim=1))    
                loss += CFG.rec_w * rec_loss
            except Exception as e:
                print(e)
                raise RuntimeError("bug")
                   
              
            try:
                assert(torch.sum(loss.isnan())==0)
            except:
                raise RuntimeError("Nan in Loss")
            training_loss.append([loss.item(), rec_loss.item(), reg_loss.item()])
            #train_reg_loss.append(rec_loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.)
            #if epoch > 0:
            if i % 2 == 0:
                optimizer1.step()
                optimizer3.step()
            else:
                optimizer2.step()
                optimizer3.step()
            
        if True:#i % CFG.ckpt_iter == 0:
        #if epoch % 10 == 0:
            valid_loss = []
            model.eval()
            #sim_head.eval()
            query_model.eval()
            if CFG.code_len > 0:
                model_bin_head.eval()
                qmodel_bin_head.eval()
            with torch.no_grad():
                for data in valid_loader:
                    ori_data = data[0].to(device)
                    pos_data = data[1].to(device)
                    half_pos_data = data[2].to(device)
                    neg_data = data[3].to(device)
                    pos_score = data[4].to(device)
                    half_pos_score = data[5].to(device)
                    neg_score = data[6].to(device)
                    #p_data = data[7].to(device)
                    aux_sub_data, aux_data = next(iter(aux_data_loader))
                    aux_sub_data.to(device)
                    aux_data.to(device)
                    """
                    #ori_out = sim_head(model(ori_data))
                    #pos_out = sim_head(model(pos_data))
                    ori_out = model(ori_data)
                    pos_out = tgt_model(pos_data)
                    pos_loss, p1, p2 = get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device, False, True)

                    half_pos_out = tgt_model(half_pos_data)
                    #half_pos_out = sim_head(model(half_pos_data))
                    half_pos_loss, h1, h2 = get_pair_loss(ori_out, half_pos_out, half_pos_score, CFG.loss_type, device, False, True)

                    neg_out = tgt_model(neg_data)
                    #neg_out = sim_head(model(neg_data))
                    neg_loss, n1, n2 = get_pair_loss(ori_out, neg_out, neg_score, CFG.loss_type, device, False, True)

                    #loss = 0.2*pos_loss + half_pos_loss + neg_loss       
                    valid_loss.append([pos_loss.item(), half_pos_loss.item(), 0.01*neg_loss.item()])
                    """
                    ori_out = query_model(ori_data)
                    pos_out = model(pos_data)
                
                    half_pos_out = model(half_pos_data)
                
                    #neg_out = sim_head(model(neg_data))
                    neg_out = model(neg_data)
                    #p_out = model(p_data)
                

                    #ori_out_2 = tgt_model(ori_data)
                #mat = F.normalize(ori_out) @ F.normalize(ori_out_2).T
                #reg_loss = torch.mean(torch.abs(mat-reg_tgt))
                    aux_sub_out = query_model(aux_sub_data)
                    aux_out = model(aux_data)
                    if CFG.code_len > 0:
                        ori_out = qmodel_bin_head(ori_out)
                        pos_out = model_bin_head(pos_out)
                        half_pos_out = model_bin_head(half_pos_out)
                        neg_out = model_bin_head(neg_out)
                        aux_sub_out = qmodel_bin_head(aux_sub_out)
                        aux_out = model_bin_head(aux_out)
                        reg_loss = binary_regularize(ori_out)+binary_regularize(pos_out)+binary_regularize(half_pos_out)+binary_regularize(neg_out)+binary_regularize(aux_sub_out)+binary_regularize(aux_out)
                    else:
                        reg_loss = 0*torch.sum(ori_out)

                    aux_pred = F.normalize(aux_sub_out) @ F.normalize(aux_out).T
                    aux_loss = torch.sum(reg_w*torch.abs(aux_pred-reg_mat))/torch.sum(reg_w)
                    _, p1, p2 = get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device, False, True)
                    
                    rec_aux_sub_qaea = q_model_decode_head(aux_sub_out)
                    rec_aux_qaea = model_decode_head(aux_out)
                    aux_sub_qaea_emb = qaea(aux_sub_data)
                    aux_qaea_emb = qaea(aux_data)
                
                    rec_loss = torch.mean(torch.sum((aux_sub_qaea_emb-rec_aux_sub_qaea)**2,dim=1))+torch.mean(torch.sum((aux_qaea_emb-rec_aux_qaea)**2, dim=1))    
                
                    valid_loss.append((1*get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device, False).item(),
                     1*get_pair_loss(ori_out, neg_out, neg_score, CFG.loss_type, device, False).item(),
                     1*get_pair_loss(ori_out, half_pos_out, half_pos_score, CFG.loss_type, device, False).item(),
                     CFG.aux_w*aux_loss.item(), CFG.bin_w*reg_loss.item(), CFG.rec_w*rec_loss.item()))
                    #valid_loss.pop()
                    #aux_sub_qaea_emb = qaea(aux_sub_data)
                    #aux_qaea_emb = qaea(aux_data)
                    #gt = F.normalize(aux_sub_qaea_emb) @ F.normalize(aux_qaea_emb).T
                    #pred = F.normalize(aux_sub_out) @ F.normalize(aux_out).T
                    #valid_loss.append([torch.mean((gt-pred)**2).item(), binary_regularize(aux_sub_out).item(), binary_regularize(aux_out).item()])
                    #valid_loss.append([get_triplet_loss(ori_out, pos_out, half_pos_out, pos_score, half_pos_score).item(), 
                    #get_triplet_loss(ori_out, half_pos_out, neg_out, half_pos_score, neg_score).item()])
                #print(np.mean(valid_loss,axis=0))
                #print(p1[:5], p2[:5])
                #print(half_pos_loss, h1[:5], h2[:5])
                #print(neg_loss, n1[:5], n2[:5])
                #pos_pred = F.cosine_similarity(ori_out, pos_out)
                #hf_pred = F.cosine_similarity(ori_out, half_pos_out)
                #neg_pred = F.cosine_similarity(ori_out, neg_out)
                #print(valid_loss[-1],pos_pred[:5], hf_pred[:5], neg_pred[:5])
                test_loss = []
                for data in test_loader:
                    ori_data = data[0].to(device)
                    pos_data = data[1].to(device)
                    half_pos_data = data[2].to(device)
                    neg_data = data[3].to(device)
                    pos_score = data[4].to(device)
                    half_pos_score = data[5].to(device)
                    neg_score = data[6].to(device)
                    #p_data = data[7].to(device)
                    
                    ori_out = query_model(ori_data)
                    pos_out = model(pos_data)
                
                    half_pos_out = model(half_pos_data)
                
                    neg_out = model(neg_data)
                    if CFG.code_len > 0:
                        ori_out = qmodel_bin_head(ori_out)
                        pos_out = model_bin_head(pos_out)
                        half_pos_out = model_bin_head(half_pos_out)
                        neg_out = model_bin_head(neg_out)

                    test_loss.append((1*get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device, False).item(),
                     get_pair_loss(ori_out, neg_out, neg_score, CFG.loss_type, device, False).item(),
                     get_pair_loss(ori_out, half_pos_out, half_pos_score, CFG.loss_type, device, False).item()))
                ave_test_loss = np.mean(test_loss)
            
                """
                valid_loss = []
                for data in valid_loader:
                    ori_data = data[0].to(device)
                    pos_data = data[1].to(device)
                    half_pos_data = data[2].to(device)
                    neg_data = data[3].to(device)
                    pos_score = data[4].to(device)
                    half_pos_score = data[5].to(device)
                    neg_score = data[6].to(device)

                    #ori_out = sim_head(model(ori_data))
                    #pos_out = sim_head(model(pos_data))
                    ori_out = model(ori_data)
                    pos_out = tgt_model(pos_data)
                    pos_loss = get_pair_loss(ori_out, pos_out, pos_score, CFG.loss_type, device)

                    half_pos_out = tgt_model(half_pos_data)
                    #half_pos_out = sim_head(model(half_pos_data))
                    half_pos_loss = get_pair_loss(ori_out, half_pos_out, half_pos_score, CFG.loss_type, device)

                    neg_out = tgt_model(neg_data)
                    #neg_out = sim_head(model(neg_data))
                    neg_loss = get_pair_loss(ori_out, neg_out, neg_score, CFG.loss_type, device)

                    #loss = 0.2*pos_loss + half_pos_loss + neg_loss       
                    valid_loss.append([pos_loss.item(), half_pos_loss.item(), neg_loss.item()])
                """
            ave_valid_loss = np.mean(valid_loss)
            #print(np.mean(valid_loss,axis=0)/np.array([1,1,1,1, 0.001]))
            print(np.mean(valid_loss, axis=0))
            #print(np.mean(test_loss,axis=0)/np.array([1,1,1]))
            
            #print(np.mean(valid_loss,axis=0))
            
            if ave_valid_loss < best_valid_loss:
                print('save')
                if CFG.code_len > 0:
                    torch.save((model, query_model, model_bin_head, qmodel_bin_head, model_decode_head, q_model_decode_head), CFG.savedir+"qaea_fine_tune_%s_%s_hash.pth"%(CFG.loss_type, CFG.sim_type))
                else:
                    torch.save((model, query_model), CFG.savedir+"qaea_fine_tune_%s_%s_aux_debug_2.pth"%(CFG.loss_type, CFG.sim_type))
                best_valid_loss = ave_valid_loss
               
            # validation
            ave_train_loss = np.mean(np.array(training_loss)[:,0])
            logging.info("Epoch %d, Iter %d, average training loss: %.3f, average train reg loss: %.3f, average valid loss %.3f, ave test loss %.6f"%(epoch, i, ave_train_loss, 
            np.mean(train_reg_loss), ave_valid_loss, ave_test_loss))
                
            model.train()
            #sim_head.train()
            query_model.train()
            training_loss = []

    #os._exit(0)
   


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    """training mse"""
    #print("asym_triplet_fine_tune_%s_%s_4.pth"%(CFG.loss_type, CFG.sim_type))
    #model, tgt_model = torch.load(CFG.savedir+"asym_triplet_fine_tune_%s_%s_4.pth"%(CFG.loss_type, CFG.sim_type))
    if CFG.code_len > 0:
        model, query_model, model_bin_head, qmodel_bin_head, model_decode_head, qmodel_decode_head = torch.load(CFG.savedir+'qaea_fine_tune_%s_%s_hash.pth'%(CFG.loss_type, CFG.sim_type))
    else:
        model, query_model = torch.load(CFG.savedir+'qaea_fine_tune_%s_%s_aux_debug_2.pth'%(CFG.loss_type, CFG.sim_type))
    #print('qaea_fine_tune_%s_%s_aux.pth'%(CFG.loss_type, CFG.sim_type))
   # model = QAEA_Linear(768)
   # query_model = QAEA_Linear(768)
    qaea = QAEA_Linear(None)
    qaea.to(device)
    model.to(device)
    query_model.to(device)
    model.eval()
    query_model.eval()    
    
    if CFG.code_len > 0:
        model_bin_head.to(device)
        qmodel_bin_head.to(device)
        model_bin_head.eval()
        qmodel_bin_head.eval()
        
    tokenizer = AutoTokenizer.from_pretrained('./SavedModel/QAEA')
    def tokenize(text, tokenizer):
        tokens = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
        data = HeteroData()
        data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
        return data
    
    print('loading db data')
    train_data = pickle.load(open('data/raw/us-filtered-asin-train-data.pkl', 'rb'))
    #train_data = train_data[:10000]
    train_list = [tokenize(session_to_text(seq), tokenizer) for seq in train_data]
    db_loader = DataLoader(train_list, batch_size=100, shuffle=False)
    """
    pred_list = []
    with torch.no_grad():
        #print(len(query_list))
        for q_data in tqdm(train_loader):
            q_emb = tgt_model(q_data[0].to(device))
            q_emb = F.normalize(q_emb)
            #print(q_emb)
            for d_data in db_loader:
                d_emb = model(d_data.to(device))
                #print(d_emb.shape)
                d_emb = F.normalize(d_emb, dim=1)
                pred = q_emb @ d_emb.T
                pred = pred.flatten().cpu().tolist()
                pred_list += pred
    print(np.mean(pred_list))
    """
    """testing"""
    
    print('loading test data')
    test_data = pickle.load(open('data/raw/us-filtered-split-test-data.pkl', 'rb'))
    query_list = [tokenize(session_to_text(seq), tokenizer) for seq, _ in zip(test_data[0], test_data[1])]
    #query_list = query_list[:100]
    query_loader = DataLoader(query_list, batch_size=100, shuffle=False)
    test_data = [(a,b) for a,b in zip(test_data[0], test_data[1])]
    
    with torch.no_grad():
        d_emb_list = []
        d_qaea_emb_list = []
        decoded_qaea_emb_list = []
        for d_data in tqdm(db_loader):
            d_emb = model(d_data.to(device))
            #d_qaea_emb = F.normalize(qaea(d_data.to(device)))
            d_qaea_emb = qaea(d_data.to(device))
            if CFG.code_len > 0:
                d_emb = model_bin_head(d_emb)
            #decoded_qaea_emb = F.normalize(model_decode_head(d_emb))
            decoded_qaea_emb = model_decode_head(d_emb)
            decoded_qaea_emb_list.append(decoded_qaea_emb.detach().cpu().numpy())
            #d_emb_list.append(F.normalize(d_emb).detach().cpu().numpy())
            d_emb_list.append(d_emb.detach().cpu().numpy())
            d_qaea_emb_list.append(d_qaea_emb.detach().cpu().numpy())
        d_emb = np.concatenate(d_emb_list, axis=0)
        d_qaea_emb = np.concatenate(d_qaea_emb_list, axis=0)
        decoded_qaea_emb = np.concatenate(decoded_qaea_emb_list, axis=0)
        print('d error', np.mean(np.sum((decoded_qaea_emb-d_qaea_emb)**2, axis=1)))
        print('d relative error', np.mean(np.sum((decoded_qaea_emb-d_qaea_emb)**2, axis=1)/(np.sum(d_qaea_emb**2, axis=1)+1e-4)))
        #print('d cos', np.mean(np.sum(decoded_qaea_emb*d_qaea_emb, axis=1)))
        if CFG.code_len > 0:
            d_emb = (d_emb+1)/2
            bin_d_emb = np.packbits(d_emb.astype(int), axis=1)
            index = faiss.IndexBinaryFlat(bin_d_emb.shape[1]*8)
            index.add(bin_d_emb)
        else:
            #index = faiss.IndexFlatIP(d_emb.shape[1])
            #index.add(d_emb)
            #index = faiss.IndexFlatIP(decoded_qaea_emb.shape[1])
            index = faiss.IndexFlatL2(d_emb.shape[1])
            index.add(d_emb)
        q_emb_list = []
        q_qaea_emb_list = []
        decoded_q_qaea_emb_list = []
        for q_data in tqdm(query_loader):
            q_emb = query_model(q_data.to(device))
            #q_qaea_emb = F.normalize(qaea(q_data.to(device)))
            q_qaea_emb = qaea(q_data.to(device))
            if CFG.code_len > 0:
                q_emb = qmodel_bin_head(q_emb)
            #decoded_q_qaea_emb = F.normalize(qmodel_decode_head(q_emb))
            decoded_q_qaea_emb = qmodel_decode_head(q_emb)
            #q_emb_list.append(F.normalize(q_emb).detach().cpu().numpy())
            q_emb_list.append(q_emb.detach().cpu().numpy())
            q_qaea_emb_list.append(q_qaea_emb.detach().cpu().numpy())
            decoded_q_qaea_emb_list.append(decoded_q_qaea_emb.detach().cpu().numpy())
        q_emb = np.concatenate(q_emb_list, axis=0)
        q_qaea_emb = np.concatenate(q_qaea_emb_list, axis=0)
        decoded_q_qaea_emb = np.concatenate(decoded_q_qaea_emb_list, axis=0)
        print('q error', np.mean(np.sum((decoded_q_qaea_emb-q_qaea_emb)**2, axis=1)))
        print('q relative error', np.mean(np.sum((decoded_q_qaea_emb-q_qaea_emb)**2, axis=1)/(np.sum(q_qaea_emb**2, axis=1)+1e-4)))
        #print('q cos', np.mean(np.sum(decoded_q_qaea_emb*q_qaea_emb, axis=1)))
        if CFG.code_len > 0:
            q_emb = (q_emb+1)/2
            bin_q_emb = np.packbits(q_emb.astype(int), axis=1)
            #index = faiss.IndexBinaryFlat(bin_q_emb.shape[1]*8)
            start = time.perf_counter()
            D,I = index.search(bin_q_emb, 100) 
            print('memory cost: ', sys.getsizeof(bin_q_emb))
            print('memory cost: ', sys.getsizeof(index))
            print('search time: %d'%(time.perf_counter()-start))       
        else:
            #D,I = index.search(q_emb, 100)
            D,I = index.search(q_emb, 100)
        gt = np.zeros_like(I,dtype=np.float32)
        for i,t in enumerate(test_data):
            for j,d in enumerate(I[i,:]):
                r = train_data[d]
                score = get_score(t, (r,[]), CFG.sim_type)
                gt[i,j] = score
        print('mean',np.mean(gt))
        print(np.mean(np.sum(gt>0.5,axis=1))/float(I.shape[1]))
#        print('mean',np.mean(gt[1000:]))
#        print(np.mean(np.sum(gt[1000:]>0.5,axis=1))/float(I.shape[1]))
        pickle.dump((D,I,gt), open("test_result_%s_%d.pkl"%(CFG.sim_type, CFG.code_len), "wb"), protocol = 4)
        print(get_ave_score(I, test_data, train_data, 'all_product_type_score'))
        print(get_ave_score(I, test_data, train_data, 'all_jaccard'))
        print(get_ave_score(I, test_data, train_data, 'all_product_title_score'))
        print(get_ave_score(I, test_data, train_data, 'all_query_score'))
    """
    gt = []
    for q in test_data:
        for d in tqdm(train_data):
            score = get_score(q,(d,[]),CFG.sim_type)
            gt.append(score)
    print(len(gt))
    #print(gt[:20])
    gt = np.array(gt)
    print(np.mean(gt), np.max(gt), np.sum(gt>0))
    

    pred_list = []
    with torch.no_grad():
        #print(len(query_list))
        for q_data in tqdm(query_loader):
            q_emb = tgt_model(q_data.to(device))
            q_emb = F.normalize(q_emb)
            #print(q_emb)
            for d_data in db_loader:
                d_emb = model(d_data.to(device))
                #print(d_emb.shape)
                d_emb = F.normalize(d_emb, dim=1)
                pred = q_emb.view(1,-1) @ d_emb.T
                pred = pred.flatten().cpu().tolist()
                pred_list += pred
            #print(len(pred_list))
    pred = np.array(pred_list)
    print(np.mean(np.abs(gt-pred)))
    print(np.mean(np.abs(gt[gt>0]-pred[gt>0])))
    print(np.mean(np.abs(gt[pred>0.2]-pred[pred>0.2])))

    pickle.dump((gt, pred), open("train_result_pt.pkl", 'wb'), protocol=4)
    """    
if __name__ == '__main__':
    main()
    test()