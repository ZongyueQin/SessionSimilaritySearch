import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, to_hetero, GCNConv, SAGEConv, GatedGraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, global_sort_pool
from torch_geometric.nn import HGTConv, HeteroConv

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.lout_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        #self.convs.append(HGTConv(hidden_channels, out_channels, data.metadata(),
        #                   num_heads, group='sum'))

    def forward(self, x_dict, edge_index_dict):
        
        out_dict = x_dict.copy()
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.lin_dict[node_type](x).relu_()

        out_dict_list = [out_dict]
        for conv in self.convs:
            out_dict = conv(out_dict_list[-1], edge_index_dict)
            out_dict_list.append(out_dict)

        for node_type in x_dict.keys():
            out_list = [out_dict[node_type] for out_dict in out_dict_list]
            x_dict[node_type] = torch.cat(out_list, dim=1)
        return x_dict

class HeteroGGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, data):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            d = {}
            for k in data.edge_index_dict.keys():
                #d[k] = GatedGraphConv(hidden_channels, 1)
                src, dst = k[0], k[2]
                if src != dst:
                    d[k] = GATConv((-1,-1),hidden_channels)
                    #d[k] = GatedGraphConv(hidden_channels, 1)
                    #d[k] = SAGEConv((-1,-1), hidden_channels)
                else:
                    d[k] = GatedGraphConv(hidden_channels, 1)
            conv = HeteroConv(d, aggr='sum')
            self.convs.append(conv)

        #self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, add_input_feat=True):
        out_dict = x_dict.copy()
        out_dict_list = [out_dict]
        for conv in self.convs:
            if edge_weight_dict is not None:
                out_dict = conv(out_dict_list[-1], edge_index_dict, edge_weight_dict)
            else:
                out_dict = conv(out_dict_list[-1], edge_index_dict)
            out_dict = {key: x.relu() for key, x in out_dict.items()}
            out_dict_list.append(out_dict)

        for node_type in x_dict.keys():
            out_list = [out_dict[node_type] for out_dict in out_dict_list]
            if add_input_feat == True:
                x_dict[node_type] = torch.cat(out_list, dim=1)
            else:
                x_dict[node_type] = torch.cat(out_list[1:], dim=1)
        return x_dict

def get_hetero_GNN(conv_key, num_features, hidden_channels_list, out_channels, num_head, data, aggr, dropout):
    model = GNN(conv_key, num_features, hidden_channels_list, out_channels, num_head, dropout)
    #print(data.metadata())
    model = to_hetero(model, data.metadata(), aggr=aggr)
    return model

class GNN(nn.Module):
    def __init__(self, conv_key, num_features, hidden_dim, out_dim, num_heads, dropout=0.5):
        super(GNN, self).__init__()
#        dims = [num_features] + hidden_channels_list + [out_channels]
        #self.layers = []
        #self.conv1 = GATConv((-1,-1), hidden_dim, num_heads, dropout=dropout)
        #self.conv2 = GATConv((-1,-1), hidden_dim, num_heads, dropout=dropout)
        #self.conv3 = GATConv((-1,-1), out_dim, num_heads, dropout=dropout)
        self.conv1 = SAGEConv((-1,-1), hidden_dim)
        self.conv2 = SAGEConv((-1,-1), hidden_dim)
        self.conv3 = SAGEConv((-1,-1), out_dim)
        """
        for i in range(len(dims)-1):
            if conv_key == 'GAT':
                conv = GATConv(dims[i], dims[i+1], num_heads, dropout=dropout)
            elif conv_key == 'GCN':
                conv = GCNConv(dims[i], dims[i+1])
            elif conv_key == 'SAGE':
                conv = SAGEConv(dims[i], dims[i+1])
            else:
                raise Exception("ConvKey can only be GAT, GCN or SAGE.")
            self.layers.append(conv)
        """
       
    def forward(self, x, edge_index):
        #for conv in self.layers:
        #    x = conv(x, edge_index)
        #    x = F.relu(x)
        #print(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        return x

class GraphPooling(nn.Module):
    def __init__(self, pooling_key, num_in, num_out, dropout):
        super(GraphPooling, self).__init__()
        self.lin = nn.Linear(num_in, num_out)
        self.pooling_key = pooling_key
        self.dropout = dropout

    def forward(self, x, batch, data):
        if self.pooling_key == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling_key == 'add':
            x = global_add_pool(x, batch)
        elif self.pooling_key == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling_key == 'sort':
            x = global_sort_pool(x, batch)
        else:
            raise Exception('Unrecognized pooling key: '+self.pooling_key)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x    

class AttentionPooling(nn.Module):
    def __init__(self, num_in, num_out):
        super(AttentionPooling, self).__init__()
        self.lin = nn.Linear(num_in, num_out)

    def forward(self, x, batch, data):
        #print(x.shape)
        coarse_rep = global_mean_pool(x, batch)
        att = x @ coarse_rep.T # num_nodes * num_graphs
        idx = torch.arange(att.size(0))
        idx = idx.cuda(att.get_device()) if att.is_cuda else idx
        att = att[idx, batch]
        #print(att.shape)
        x = x * att.view(-1,1)
        #print(x.shape)

        return self.lin(global_mean_pool(x, batch))


class SRGNN_Pooling(nn.Module):
    def __init__(self, num_in, num_out):
        super(SRGNN_Pooling, self).__init__()
        self.lin1 = nn.Linear(num_in, num_in)
        self.lin2 = nn.Linear(num_in, num_in)
        self.lin3 = nn.Linear(num_in, 1, bias=False)
        self.lin4 = nn.Linear(num_in*2, num_out)

    def forward(self, x, batch, data):
        local_rep = global_add_pool(x*data['product'].last_click_mask.view(-1,1), batch)
        local_rep_repeat = local_rep[batch]
        
        #print(num_in)
        att = self.lin3(torch.sigmoid(self.lin1(local_rep_repeat)+self.lin2(x)))
        weighted_x = x * att
        global_rep = global_add_pool(weighted_x, batch)
        rep = torch.cat([local_rep, global_rep], dim=1)
        return self.lin4(rep)

class PositionalAttentionPooling(nn.Module):
    def __init__(self, query_in, product_in, num_out, max_seq_len):
        super(PositionalAttentionPooling, self).__init__()
        self.query_lin = nn.Linear(query_in, num_out-max_seq_len)
        self.product_lin = nn.Linear(product_in, num_out-max_seq_len)
        self.positional_emb = nn.Embedding(max_seq_len, max_seq_len)
        self.node_emb_lin = nn.Linear(num_out, num_out)
        self.coarse_rep_lin = nn.Linear(num_out, num_out, bias=False)
        self.att_lin = nn.Linear(num_out, 1, bias=False)

    def forward(self, input_emb, data):
        # Linear Transformation to unify dimensionalties
        query_emb = self.query_lin(input_emb['query'])
        product_emb = self.product_lin(input_emb['product'])

        # expand and concatenate with positional encoding
        query_pos_emb = self.positional_emb(data['query'].pos_emb_id)
        query_emb = torch.tanh(torch.concat([query_emb, query_pos_emb],dim=1))

        product_emb = torch.repeat_interleave(product_emb, data['product'].cnt, dim=0)
        product_pos_emb = self.positional_emb(data['product'].pos_emb_id)
        product_emb = torch.tanh(torch.concat([product_emb, product_pos_emb], dim=1))

        product_batch=torch.repeat_interleave(data['product'].batch, data['product'].cnt, dim=0)

        node_emb = torch.concat([product_emb, query_emb], dim = 0)
        node_batch = torch.concat([product_batch, data['query'].batch], dim = 0)

        coarse_rep = global_mean_pool(node_emb, node_batch)
        coarse_rep = coarse_rep[node_batch]
        a=self.node_emb_lin(node_emb)
        b=self.coarse_rep_lin(coarse_rep)
        att = self.att_lin(torch.sigmoid(a+b))
        weighted_node_emb = node_emb * att    
        return global_mean_pool(weighted_node_emb, node_batch)
