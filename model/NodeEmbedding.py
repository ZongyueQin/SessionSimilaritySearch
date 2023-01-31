import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #print(pe.shape)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, seq_len, embed dim]
            output: [batch size, seq_len embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class AveragePooling(nn.Module):
    """
    Conduct Average Pooling over dim-th dimension
    """
    def __init__(self, dim):
        super(AveragePooling, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.mean(x, dim=self.dim)

class NodeTextTransformer(nn.Module):
    """ Use Transformer model to get node embeeding from text attributes 
        According to pytorch documents, the input shape should be (sentence_len, batch_size)
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, pooling_layer=None, dropout=0.5, batch_first=True):
        super(NodeTextTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.batch_first = batch_first
        if pooling_layer is not None:
            self.pooling = pooling_layer
        else:
            if self.batch_first == False:
                self.pooling = AveragePooling(0)
            else:
                self.pooling = AveragePooling(1)
                
    def forward(self, src, mask):
        #print("HERE")
        #print(src)
        assert(src.isnan().any()==False)
        src = self.embedding(src) 
        assert(src.isnan().any()==False)
        src = src * math.sqrt(self.ninp)
        assert(src.isnan().any()==False)
        src = self.pos_encoder(src)
        assert(src.isnan().any()==False)
        output = self.transformer_encoder(src, src_key_padding_mask=mask)
        #print(mask)
        assert(output.isnan().any()==False)
        output = self.pooling(output)
        assert(output.isnan().any()==False)
        return output

class PretrainedQAEAEncoder(nn.Module):

    def __init__(self, nout):
        super(PretrainedQAEAEncoder, self).__init__()
        self.model = AutoModel.from_pretrained("/home/qinzongyue/Amazon/code/SavedModel/QAEA", add_pooling_layer = False)

        #self.model = AutoModel.from_pretrained("bert-base-uncased", add_pooling_layer = False)
        electra_dim = 768
        if nout is not None:
            self.lin = nn.Linear(electra_dim, nout)
        else:
            self.lin = None  
    def __call__(self, input_ids, token_type_ids, attention_mask, get_token=False):
        token_emb = self.model(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask).last_hidden_state
        out = torch.sum(token_emb*attention_mask.unsqueeze(-1), dim=1)/torch.sum(attention_mask, dim=1).view(-1,1)
        out = out.detach()
        if self.lin is not None:
            if get_token == True:
                return self.lin(out), token_emb
            else:
                return self.lin(out)
        else:
            if get_token == True:
                return out, token_emb
            else:
                return out
            
            
class NodeAsinEmbedding(nn.Module):
    """
    Treat each product (identified by ASIN) as a separate label 
    """
    def __init__(self, nproducts, ninp):
        super(NodeAsinEmbedding, self).__init__()
        self.nproducts = nproducts
        self.encoder = nn.Embedding(nproducts, ninp)
       # self.encoder.weight = nn.Parameter(0.001 * self.encoder.weight)
    def forward(self, input):
        return self.encoder(input)


def test_main():
    input = torch.Tensor([[1,2,3,4],[2,3,4,1]]).long()
    encoder = NodeTextTransformer(5,10,2,2,2)
    output = encoder(input)
    print(output)
    print(output.shape)


if __name__ == '__main__':
    test_main()
