import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import sys
sys.path.insert(0, "..")
#from Dataset import ProductAsinSessionTestDataset, pretransform_QueryTokenProductAsin
from transformers import BertTokenizer, AutoModel
from . import get_hetero_GNN, GraphPooling, AttentionPooling
from . import NodeAsinEmbedding, NodeTextTransformer, AveragePooling, PositionalEncoding
import numpy as np
from torch_geometric.nn import global_mean_pool


class MyTransformerDecoder(nn.Module):
    def __init__(self, ninp, nout, nhead, nhid, nlayers, dropout=0.5, batch_first=True):
        super(MyTransformerDecoder, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout, batch_first=batch_first)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, nlayers)
        self.ninp = ninp
        self.batch_first = batch_first
        self.lin = nn.Linear(ninp, nout)
        self.dropout = dropout
                
    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        tgt = self.pos_encoder(tgt)
        assert(tgt.isnan().any()==False)
        #print(tgt.shape)
        #print(tgt_key_padding_mask.shape)
        #print(torch.sum(tgt_key_padding_mask,dim=1))
        output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        
        assert(output.isnan().any()==False)
        #output = self.transformer_decoder(tgt=tgt, memory=memory)
        output = self.lin(F.dropout(output, p=self.dropout))
        assert(output.isnan().any()==False)
        return output

class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_hidden_layers, dropout, last_act = True, jump = False):
        super(MLP, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.layers = nn.ModuleList()
        self.dropout = dropout
        #self.layers.append(nn.BatchNorm1d(n_input))
        self.layers.append(nn.Linear(n_input, n_hidden))
        self.layers.append(nn.BatchNorm1d(n_hidden))
        for i in range(n_hidden_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(nn.BatchNorm1d(n_hidden))
        if jump == False:
            self.layers.append(nn.Linear(n_hidden, n_output))
        else:
            self.layers.append(nn.Linear(n_hidden+n_input, n_output))
        self.last_act = last_act
        self.jump = jump
    
    def forward(self, x):
        inp = x
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout)
        if self.jump == True:
            x = torch.concat([inp,x],dim=1)
        if self.last_act == True:
            x = torch.tanh(self.layers[-1](x))
        else:
            x = self.layers[-1](x)
        return x

class QAEA_Linear(nn.Module):
    def __init__(self, n_out):
        super(QAEA_Linear, self).__init__()
        self.qaea = AutoModel.from_pretrained("SavedModel/QAEA", add_pooling_layer=False)
        if n_out is not None:
            self.lin = nn.Linear(768, n_out)
            #self.lin.weight.data = torch.eye(768)
            #self.lin.bias.data = torch.zeros(768)
        else: 
            self.lin = None
        for param in self.qaea.parameters():
            param.requires_grad = False

    def forward(self, data):
        input, type_id, mask = data['input_ids'].x, data['token_type_ids'].x, data['attention_mask'].x
        emb = self.qaea(input_ids=input, token_type_ids = type_id, attention_mask=mask).last_hidden_state
        #emb = torch.sum(emb,dim=1)
        emb = torch.sum(emb*mask.unsqueeze(-1), dim=1)
        emb = emb / torch.sum(mask, dim=1).view(-1,1)
        emb = global_mean_pool(emb, data['input_ids'].batch)
        if self.lin is not None:
            #out = self.lin(emb)
            #print(torch.sum(torch.abs(self.lin.weight.data.cpu()-torch.eye(768))))
            #print(torch.sum(torch.abs(self.lin.bias.data)))
            #print(torch.sum(torch.abs(out-emb)))
            #assert torch.sum(torch.abs(out-emb)) < 1e-4
            return self.lin(emb)
        else:
            return emb
            
class BinarizeHead(nn.Module):
    def __init__(self, n_input, n_output, mlp, jump=False):
        super(BinarizeHead, self).__init__()
        self.n_input = n_input
        self.mlp = mlp
        self.n_output = n_output
        self.lammy = nn.parameter.Parameter(torch.tensor(1.00))
        self.bn = nn.BatchNorm1d(n_input, affine=False)
        self.lin1 = nn.Linear(n_input, self.n_output)
        self.jump = jump
        #self.lin2 = nn.Linear(self.n_hid, n_output)

    def forward(self, x):
        #out = self.lin2(F.relu(self.lin1(x)))

        if self.mlp is None:
            out = x
            out = self.lin1(out)
        else:
            out = F.tanh(self.mlp(x))
            if self.jump == True:
                out = torch.concat([out,x],dim=1)
            out = self.lin1(out)
       # out = self.lin1(out)
        #print(self.train)
        if self.training == True:
            #print(self.lammy)
            #print(out)
            return F.tanh(out)
            #return out + (F.tanh(out) - out).detach()
            return F.tanh(out/torch.clip(self.lammy,min=0.01))
        else:
            return (torch.sign(out) - F.tanh(out)).detach() + F.tanh(out)
            return (torch.sign(out) - F.tanh(out/torch.clip(self.lammy,min=0.01))).detach() + F.tanh(out/torch.clip(self.lammy,min=0.01))


class CrossAttentionTransformer(nn.Module):
    def __init__(self, nlayers, node_emb_K, node_dim, token_dim, nhead, nhid, dropout=0.5):
        super(CrossAttentionTransformer, self).__init__()
        #self.pos_encoder = PositionalEncoding(token_dim, 0)
        encoder_layer = nn.TransformerEncoderLayer(token_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        self.node_lin = nn.Linear(node_dim, node_emb_K * token_dim)
        self.token_dim = token_dim
        self.mask = None
        self.K  = node_emb_K

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.zeros(sz, sz)
        mask[:self.K, self.K:] = float('-inf')
        return mask

    def forward(self, node_emb, token_emb, token_mask):
        # node_emb shape batch * node_dim
        # token_emb shape batch * seq * token_dim
        node_emb = self.node_lin(node_emb)
        node_emb = torch.stack(torch.split(node_emb, self.token_dim, dim=1), dim=1) # batch * K * token_dim
        token_emb = torch.concat((node_emb, token_emb), dim=1) # batck * (K+seq) * token_dim

        if self.mask is None or self.mask.size(0) != token_emb.size(1):
            self.mask = self._generate_square_subsequent_mask(token_emb.size(1)).to(node_emb.device)

        #out = self.pos_encoder(token_emb)
        latent_token_mask = torch.zeros_like(node_emb[:,:,0]).bool()
        token_mask = torch.concat((latent_token_mask, token_mask), dim=1)
        out = self.transformer(token_emb, mask=self.mask, src_key_padding_mask=token_mask)
        # only return updated token embeddings
        return out[:, self.K:, :]

class NodeLevelEncoder(nn.Module):
    def __init__(self, query_node_embedder, product_node_embedder, gnn):
        super(NodeLevelEncoder, self).__init__()
        self.query_node_embedder = query_node_embedder
        self.product_node_embedder = product_node_embedder
        self.gnn = gnn

    def forward(self, data):
        embedding = {}
        embedding['query'] = self.query_node_embedder(data['query'].x, data['query'].attention_mask==0)
        embedding['product'] = self.product_node_embedder(data['product'].x)
        #print(data.edge_index_dict)
        #print(embedding['query'].shape)
        #print(embedding['product'].shape)
        
        node_embedding = self.gnn(embedding, data.edge_index_dict, data.edge_weight_dict)
        return node_embedding

class GraphLevelEncoder(nn.Module):
    def __init__(self, query_node_embedder, product_node_embedder, gnn, product_pooling, query_pooling, use_id_embedding=True):
        super(GraphLevelEncoder, self).__init__()
        self.query_node_embedder = query_node_embedder
        self.product_node_embedder = product_node_embedder
        self.gnn = gnn
        self.query_pooling = query_pooling
        self.product_pooling = product_pooling
        self.use_id_embedding = use_id_embedding

    def forward(self, data, query_node_mask = None, product_node_mask=None, get_node=False, get_token=False):
        embedding = {}
        #print(data['query'].x.shape)
        embedding['query'] = self.query_node_embedder(data['query'].x, data['query'].token_type_ids, data['query'].attention_mask)
        #print(embedding['query'].shape)
        a = self.product_node_embedder(data['product'].x)
        b = self.query_node_embedder(data['product'].input_ids, data['product'].token_type_ids, data['product'].attention_mask)
        #print(a.shape, b.shape)
        if hasattr(self, 'use_id_embedding') == False or self.use_id_embedding == True:
            embedding['product'] = torch.concat((a,b),dim=1)
        else:
            embedding['product'] = b

        if query_node_mask is not None:
            embedding['query'] = embedding['query']*query_node_mask.view(-1,1)
        if product_node_mask is not None:
            embedding['product'] = embedding['product']*product_node_mask.view(-1,1)
        
        #embedding['product'] = b
        #print(data.edge_index_dict)
        #print(data)
        assert(data['query'].x.isnan().any()==False)
        try:
            assert(embedding['query'].isnan().any()==False)
        except Exception as e:
            for i in range(embedding['query'].size(0)):
                if embedding['query'][i,:].isnan().any():
                    print(i)
                    print(embedding['query'][i,:])
                    print(data['query'].x[i,:])
                    print(data['query'].attention_mask[i,:])
                    
            raise RuntimeError("nan in embedding[query]")

        assert(embedding['product'].isnan().any()==False)
        
        node_embedding = self.gnn(embedding, data.edge_index_dict)
        assert(node_embedding['query'].isnan().any()==False)
        assert(node_embedding['product'].isnan().any()==False)
        #print(node_embedding['query'].shape)


        query_embedding = self.query_pooling(node_embedding['query'], data['query'].batch, data)
        product_embedding = self.product_pooling(node_embedding['product'], data['product'].batch, data)
        assert(product_embedding.isnan().any()==False)
        assert(query_embedding.isnan().any()==False)
        #print(data['query'].batch)
        #print(data['product'].batch)
        #print(query_embedding.shape)
        #product_embedding = self.graph_pooling(embedding['product'], data['product'].batch)
        #query_embedding = self.graph_pooling(embedding['query'], data['query'].batch)
        
        graph_embedding = torch.cat((query_embedding, product_embedding), 1)
        
        #print(graph_embedding.shape)
        if get_node == False:
            return graph_embedding
        else:
            return graph_embedding, node_embedding, None


class UnifyPoolingGraphLevelEncoder(nn.Module):
    def __init__(self, query_node_embedder, product_node_embedder, gnn, pooling, cross_attention_transformer, use_id_embedding=True):
        super(UnifyPoolingGraphLevelEncoder, self).__init__()
        self.query_node_embedder = query_node_embedder
        self.product_node_embedder = product_node_embedder
        self.gnn = gnn
        self.pooling = pooling
        self.use_id_embedding = use_id_embedding
        self.cross_attention_transformer = cross_attention_transformer
        gnn_nlayers = 3
        gnn_nout = 800
        self.query_pooling = AttentionPooling(gnn_nlayers*gnn_nout+768, gnn_nout)
        self.product_pooling = AttentionPooling(gnn_nlayers*gnn_nout+768, gnn_nout)
        self.last_lin = nn.Linear(1600,768)


    def forward(self, data, query_node_mask = None, product_node_mask=None, get_node=False, get_token=False):
        embedding = {}
        token_emb = {}
        #print(data['query'].x.shape)
        embedding['query'], token_emb['query'] = self.query_node_embedder(data['query'].x, data['query'].token_type_ids, data['query'].attention_mask, True)
        #print(embedding['query'].shape)
        a = self.product_node_embedder(data['product'].x)
        b, token_emb['product'] = self.query_node_embedder(data['product'].input_ids, data['product'].token_type_ids, data['product'].attention_mask, True)
        #print(a.shape, b.shape)
        if hasattr(self, 'use_id_embedding') == False or self.use_id_embedding == True:
            embedding['product'] = torch.concat((a,b),dim=1)
        else:
            embedding['product'] = b

        if query_node_mask is not None:
            embedding['query'] = embedding['query']*query_node_mask.view(-1,1)
        if product_node_mask is not None:
            embedding['product'] = embedding['product']*product_node_mask.view(-1,1)
        
        #embedding['product'] = b
        #print(data.edge_index_dict)
        #print(data)
        assert(data['query'].x.isnan().any()==False)
        try:
            assert(embedding['query'].isnan().any()==False)
        except Exception as e:
            for i in range(embedding['query'].size(0)):
                if embedding['query'][i,:].isnan().any():
                    print(i)
                    print(embedding['query'][i,:])
                    print(data['query'].x[i,:])
                    print(data['query'].attention_mask[i,:])
                    
            raise RuntimeError("nan in embedding[query]")

        assert(embedding['product'].isnan().any()==False)
        
        #node_embedding = self.gnn(embedding, data.edge_index_dict, add_input_feat=False)
        node_embedding = self.gnn(embedding, data.edge_index_dict, add_input_feat=True)
        
        #print(node_embedding['query'].shape)
        session_level_token_emb = {}
        #TODO
        """
        session_level_token_emb['product'] = self.cross_attention_transformer(node_embedding['product'], token_emb['product'], data['product'].attention_mask==0)
        session_level_token_emb['query'] = self.cross_attention_transformer(node_embedding['query'], token_emb['query'], data['query'].attention_mask==0)

        mean_query_token_emb = torch.sum(session_level_token_emb['query']*data['query'].attention_mask.unsqueeze(-1), dim=1)/torch.sum(data['query'].attention_mask,dim=1).view(-1,1)
        node_embedding['query'] = torch.concat([node_embedding['query'], mean_query_token_emb], dim=1)
        #print(node_embedding['query'].shape)
        mean_product_token_emb = torch.sum(session_level_token_emb['product']*data['product'].attention_mask.unsqueeze(-1), dim=1)/torch.sum(data['product'].attention_mask,dim=1).view(-1,1)
        node_embedding['product'] = torch.concat([node_embedding['product'], mean_product_token_emb], dim=1)
        #assert(node_embedding['query'].isnan().any()==False)
        #assert(node_embedding['product'].isnan().any()==False)
        """
        
        graph_embedding = self.pooling(node_embedding, data)
        #graph_embedding = self.last_lin(graph_embedding)
        #TODO
        #query_embedding = self.query_pooling(node_embedding['query'], data['query'].batch, data)
        #product_embedding = self.product_pooling(node_embedding['product'], data['product'].batch, data)
        #assert(product_embedding.isnan().any()==False)
        #assert(query_embedding.isnan().any()==False)
        #graph_embedding = torch.cat((query_embedding, product_embedding), 1)
        #print(graph_embedding.shape)
        if get_node == False and get_token == False:
            return graph_embedding
        elif get_node == True and get_token == False:
            return graph_embedding, node_embedding
        elif get_node == False and get_token == True:
            return graph_embedding, session_level_token_emb
        else:
            return graph_embedding, node_embedding, session_level_token_emb

def debug():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = ProductAsinSessionTestDataset(root=".", pre_transform=pretransform_QueryTokenProductAsin)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    query_node_embedder = NodeTextTransformer(ntoken=tokenizer.vocab_size, ninp=100, nhead=4, nhid=200, 
            nlayers=3,batch_first=True)
    product_node_embedder = NodeAsinEmbedding(nproducts=100000, ninp=100)
    gnn = get_hetero_GNN('GAT', 100, 200, 100, 4, dataset[0], 'sum', 0.25)
    node_encoder = NodeLevelEncoder(query_node_embedder=query_node_embedder,
                        product_node_embedder=product_node_embedder,
                        gnn=gnn)
    graph_pooling = GraphPooling('mean', 400, 100, 0.5)
    graph_encoder = GraphLevelEncoder(query_node_embedder=query_node_embedder,
                        product_node_embedder=product_node_embedder,
                        gnn=gnn, graph_pooling=graph_pooling)
    for data in loader:
        print(data['query'].x.shape)
        print(data['product'].x.shape)
        #print(data.x_dict)
        out = graph_encoder(data)
        print(out)
        print(out.shape)
        break
if __name__ == '__main__':
    debug()
