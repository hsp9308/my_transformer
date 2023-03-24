# Defining Encoder and Decoder Layer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.sublayers import MultiheadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """_summary_
    Single Encoder Layer

    Args:
        embed_dim (int): embedding dimensions
        d_pf (int): PointwiseFeedForward dimension
        n_heads (int): number of heads
        dropout (int): dropout rate in the sublayers. 
        
    """    
    
    def __init__(self, embed_dim, d_pf, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.self_attn = MultiheadAttention(embed_dim, n_heads,
                                            dropout=dropout)
        self.pos_feedforward = PositionwiseFeedForward(embed_dim, d_pf, dropout=dropout)
    
    def forward(self, enc_inp, self_attn_mask=None):
        out, enc_scores = self.self_attn(enc_inp, enc_inp, enc_inp, mask=self_attn_mask)
        out = self.pos_feedforward(out)
        return out, enc_scores
    
class DecoderLayer(nn.Module):
    """_summary_
    Single Decoder Layer

    Args:
        embed_dim (int): embedding dimensions
        d_pf (int): PointwiseFeedForward dimension
        n_heads (int): number of heads
        dropout (int): dropout rate in the sublayers. 
        
    """    
    def __init__(self, embed_dim, d_pf, n_heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, n_heads, 
                                            dropout=dropout)
        self.enc_attn = MultiheadAttention(embed_dim, n_heads,
                                           dropout=dropout)
        self.pos_feedforward = PositionwiseFeedForward(embed_dim, d_pf, dropout=dropout)
        
    def forward(self, dec_inp, enc_out, self_attn_mask=None, enc_attn_mask=None):
        dec_out, dec_self_scores = self.self_attn(dec_inp, dec_inp, dec_inp, mask=self_attn_mask)
        dec_out, dec_enc_scores = self.enc_attn(dec_out, enc_out, enc_out, mask=enc_attn_mask)
        dec_out = self.pos_feedforward(dec_out)
        return dec_out, dec_self_scores, dec_enc_scores
        
        