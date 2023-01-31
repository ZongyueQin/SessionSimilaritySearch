import numpy as np
import torch
from torch_geometric.data import HeteroData
import Levenshtein


def get_query_node_tokens(session_details, tokenizer, max_length):
    query_words = [""] # root node
    query_pos = [0]
    for i, action in enumerate(session_details):
        if action[1] != 's': # not search action
            continue
        query_word = action[2]
        if query_word is None:
            query_word = ""
        query_words.append(query_word)
        query_pos.append(i+1)
    #print(query_words)
    tokens = tokenizer(query_words, padding='max_length', 
            max_length=max_length, truncation=True, return_tensors="pt")
    #print(tokens)
    return tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask'], len(session_details)-torch.tensor(query_pos)


def binary_regularize(out):
    return torch.mean(torch.abs(1-torch.abs(out)))

def normalize(vec):
    if len(vec.shape)==1:
        return vec / np.sqrt(np.clip(np.sum(vec**2), 1e-6, None))
    return vec / np.sqrt(np.clip(np.sum(vec**2,axis=1), 1e-6, None)).reshape(-1,1)

def get_item(session):
    return set([action[-1] for action in session if action[1] != 's'])

def get_session_item_title(session):
    return [action[-2] if action[-2] is not None else '' for action in session if action[1] != 's']

def get_next_query(seq):
    next_query = None
    for action in seq:
        if action[1] == 's':
            query_word = action[2]
            if query_word is not None:
                next_query = query_word
                break
    return next_query

def get_all_query(seq):
    all_query = []
    for action in seq:
        if action[1] == 's':
            query_word = action[2]
            if query_word is not None:
                all_query.append(query_word)
    return all_query


def get_item_type(session):
    return [action[4] for action in session if action[1] != 's' if action[4] is not None]

def get_item_title(seq, item_list):
    title_list = []
    for item in item_list:
        for action in seq:
            if action[1] != 's' and action[-1] == item:
                title = action[-2]
                if title is None:
                    title = ""
                title_list.append(title)
                break
    return title_list


def get_item_pos_cnt(seq, item_list):
    pos_emb_id_list = []
    cnt_list = [0 for item in item_list]
    for i, item in enumerate(item_list):
        for j, action in enumerate(seq):
            if action[1] != 's' and action[-1] == item:
                cnt_list[i] += 1
                pos_emb_id_list.append(len(seq)-j)
    return pos_emb_id_list, cnt_list
                
def session_to_text(session):
    text = []
    for action in session:
        if action[1] == 's':
            sentence = action[2]
        else:
            sentence = action[-2]
        if sentence is None:
            sentence = ""
        text.append(sentence)
    return text
    

def sequence_to_graph(idx, seq, tar, tokenizer, query_max_len, ignore_query=False):
    data = HeteroData()
    data['idx'].idx = idx
    if ignore_query == True:
        old_seq = seq
        seq = [action for action in seq if action[1] != 's']

    data['query'].x, data['query'].token_type_ids, data['query'].attention_mask, data['query'].pos_emb_id = get_query_node_tokens(seq, tokenizer, query_max_len)

    data['query'].input_ids = data['query'].x
    data['query'].num_nodes = data['query'].x.shape[0]
    data['query'].mask = torch.ones(data['query'].x.shape[0])
    data['query'].mask[0] = 0

    #next_query = get_next_query(tar)
    future_query = get_all_query(tar)
    if len(future_query) == 0:#next_query is None:
        #next_query = ""
        future_query = [""]
        data['query_target'].mask = torch.zeros(1)
    else:
        data['query_target'].mask = torch.ones(len(future_query))
    tokens = tokenizer(future_query, padding='max_length', 
            max_length=query_max_len, truncation=True, return_tensors="pt")
    data['query_target'].input_ids, data['query_target'].token_type_ids, data['query_target'].attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
    data['query_target'].num_nodes = len(future_query)
    #assert(data['query_target'].input_ids.size(0)==1)
    #print(tar[:1])
    #print(data['query_target'].input_ids.shape)

    distinct_item = list(get_item(seq))
    item_pos_emb_id, item_cnt = get_item_pos_cnt(seq, distinct_item)
    assert(np.sum(item_cnt)==len(item_pos_emb_id))
    assert(len(item_cnt)==len(distinct_item))
    if len(distinct_item) == 0:
        distinct_item = [0] # represent an unknown product
        item_cnt = [1]
        item_pos_emb_id = [0]
    pos = {}
    for i in range(len(distinct_item)):
        pos[distinct_item[i]]=i
    data['product'].x = torch.LongTensor(distinct_item)
    data['product'].num_nodes = len(distinct_item)
    data['product'].cnt = torch.tensor(item_cnt)
    data['product'].pos_emb_id = torch.tensor(item_pos_emb_id)
    title_list = get_item_title(seq, distinct_item)
    if len(title_list) == 0:
        assert(len(distinct_item)==1 and distinct_item[0]==0)
        title_list = ['UNK']
    if len(title_list) != len(distinct_item):
        print(title_list)
        print(distinct_item)
    
    tokens = tokenizer(title_list, padding='max_length', 
            max_length=query_max_len, truncation=True, return_tensors="pt")
    data['product'].input_ids, data['product'].token_type_ids, data['product'].attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
    assert(data['product'].input_ids.size(0)==data['product'].num_nodes)
    data['product'].mask = torch.ones(data['product'].num_nodes)
    item_seq = [action[-1] for action in seq if action[1] != 's']
    if len(item_seq) == 0:
        item_seq = [0]
    data['ori_seq'] = (seq, tar)

    
    data['product_target'].y = torch.LongTensor(list(get_item(tar)))
    data['product_target'].num_nodes = data['product_target'].y.size(0)
    n = data['product_target'].num_nodes
    distinct_item = list(get_item(tar))
    if len(distinct_item) == 0:
        distinct_item = [0]
    title_list = get_item_title(tar, distinct_item)
    if len(title_list) == 0:
        title_list = ['UNK']
    tokens = tokenizer(title_list, padding='max_length', 
            max_length=query_max_len, truncation=True, return_tensors="pt")
    data['product_target'].input_ids, data['product_target'].token_type_ids, data['product_target'].attention_mask = tokens['input_ids'][:n], tokens['token_type_ids'][:n], tokens['attention_mask'][:n]
#    print(data['product_target'].title_input_ids.size(0),data['product_target'].num_nodes)
    data['product_target'].mask = torch.ones(n)
    assert(data['product_target'].input_ids.size(0)==data['product_target'].num_nodes)

    
    # query-item edges, ignore ca, p, c differences
    last_query_node = 0
    from_node = []
    to_node = []
    for action in seq:
        if action[1] == 's': # search action
            last_query_node += 1
            continue
        elif action[3] is None and action[-1] != 0:
            raise RuntimeError("asin is None")     
        item = action[-1]
        from_node.append(last_query_node)
        to_node.append(pos[item])
        

    data['query', 'clicks','product'].edge_index = torch.tensor([from_node, to_node], dtype=torch.long)
    data['product', 'clicked by', 'query'].edge_index = torch.tensor([to_node, from_node], dtype=torch.long)
    data['query', 'clicks','product'].edge_weight = None
    data['product', 'clicked by', 'query'].edge_weight= None
    
    from_node = []
    to_node = []
    weight = []
    edge_to_pos = {}
    last_click_pos = 0
    for i in range(len(item_seq)-1):
        if (pos[item_seq[i]], pos[item_seq[i+1]]) not in edge_to_pos.keys():
            edge_to_pos[(pos[item_seq[i]], pos[item_seq[i+1]])] = len(from_node)
            from_node.append(pos[item_seq[i]])
            to_node.append(pos[item_seq[i+1]])
            weight.append(1)
        else:
            p = edge_to_pos[(pos[item_seq[i]], pos[item_seq[i+1]])] 
            weight[p] += 1
        last_click_pos = pos[item_seq[i+1]]
    
    data['product'].last_click_mask = torch.zeros_like(data['product'].x).float()
    data['product'].last_click_mask[last_click_pos] = 1
    data['product', 'to', 'product'].edge_index = torch.tensor([from_node, to_node], dtype=torch.long)
    data['product', 'to', 'product'].edge_weight = torch.tensor(weight, dtype=torch.float32)
    data['product'].y = data['product'].x
    #data['product'].y_cnt = torch.bincount(data['product'].y, minlength=asin_num).view(1,-1)

    #text = session_to_text(seq+tar)
    text = ['']+session_to_text(seq)
    tokens = tokenizer(text,padding='max_length', max_length=20, truncation=True, return_tensors="pt")
    data['text'].input_ids, data['text'].token_type_ids, data['text'].attention_mask = tokens['input_ids'], tokens['token_type_ids'], tokens['attention_mask']
    data['text'].num_nodes = data['text'].input_ids.shape[0]
    if ignore_query == True:
        seq = old_seq

    return data



def get_query(sess, pad = True):
    if pad == False:
        return [action[2] for action in sess if action[1] == 's' and action[2] is not None]
    else:
        return [""] + [action[2] for action in sess if action[1] == 's' and action[2] is not None]
def get_string_match(a, b):
  #  print(a,b)
    a_match = [0 for item in a]
    b_match = [0 for item in b]
    for i, a_s in enumerate(a):
        for j, b_s in enumerate(b):
            #print(a_s, b_s)
            if Levenshtein.ratio(a_s,b_s) > 0.9:
                a_match[i] = 1
                b_match[j] = 1
    return np.sum(a_match), np.sum(b_match)

