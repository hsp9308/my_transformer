# Defining transformer model
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.layers import EncoderLayer, DecoderLayer

class PositionalEncoding(nn.Module):
    """_summary_
    Positional encoders in original papers
    One may use it.

    Args:
        embed_dim (int): feature dimensions
        n_positions (int): number of positions. Default is 200 .
    """    
    def __init__(self, embed_dim, n_position=200):
        super().__init__()
        
        self.register_buffer('pos_table', self._get_pos_table(n_position, embed_dim))
        
    def _get_pos_table(self, n_position, embed_dim):
        def position_angle_vec(pos):
            return [pos / (10000.0)**(2.0 * i / embed_dim) for i in range(embed_dim) ]
        table = np.array([position_angle_vec(pos_i) for pos_i in range(n_position)])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return torch.FloatTensor(table).unsqueeze(0)
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    
class Encoder(nn.Module):
    """_summary_
    Encoder module
    
    Args:
        input_dim (int): Input dimensions (=number of vocab.)
        embed_dim (int): Embedding vector dimensions
        n_layers (int): Number of EncoderLayer
        n_heads (int): Number of heads for multi-head attention layer
        d_pf (int): Intermediate dimensions in the 'Position-wise Feed Forward layer'
        device (str): Pytorch device type.
        n_position (int): number of positions in the positional encoding layer.
        preset_pos_encoding (bool): If True, Positional Encoding is performed as in the original paper.
        dropout_ratio (int): Dropout ratio.
    """    
    def __init__(self, input_dim, embed_dim, n_layers, n_heads, d_pf, device,
                 n_position=200, preset_pos_encoding=False, dropout_ratio=0.1):
        super().__init__()
        self.src_to_embedding = nn.Embedding(input_dim, embed_dim)
        if preset_pos_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, n_position)
        else:
            self.pos_encoding = nn.Embedding(n_position, embed_dim)
            
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, d_pf, n_heads, dropout_ratio) 
             for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout_ratio)
        self.device = device
        
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim])).to(self.device)
        
        
    def forward(self, src, src_mask):
        # src: [batch_size, src_len]
        # src_mask: [batch_size, src_len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        x = self.src_to_embedding(src) * self.scale
        
        x = self.dropout(x + self.pos_encoding(pos))
        
        for layer in self.layers:
            x, score = layer(x, src_mask)
        
        # x: [batch_size, src_len, embed_dim]
        return x, score
        
class Decoder(nn.Module):
    """_summary_
    Decoder module

    Args:
        output_dim (int): Output dimensions (=number of vocab.)
        embed_dim (int): Embedding vector dimensions
        n_layers (int): Number of EncoderLayer
        n_heads (int): Number of heads for multi-head attention layer
        d_pf (int): Intermediate dimensions in the 'Position-wise Feed Forward layer'
        device (str): Pytorch device type.
        n_position (int): number of positions in the positional encoding layer.
        preset_pos_encoding (bool): If True, Positional Encoding is performed as in the original paper.
        dropout_ratio (int): Dropout ratio.
    """    
    def __init__(self, output_dim, embed_dim, n_layers, n_heads, d_pf, device, 
                 n_position=200, preset_pos_encoding=False, dropout_ratio=0.1):
        super().__init__()
        self.src_to_embedding = nn.Embedding(output_dim, embed_dim)
        if preset_pos_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, n_position)
        else:
            self.pos_encoding = nn.Embedding(n_position, embed_dim)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, d_pf, n_heads, dropout_ratio)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(embed_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_ratio)
        
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
        
    def forward(self, target, target_mask, enc_out, enc_mask):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        
        pos = torch.arange(0, target_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        target = self.src_to_embedding(target) * self.scale
        
        target = self.dropout(target + self.pos_encoding(pos))
        
        for layer in self.layers:
            target, self_score, attn_score = layer(target, enc_out, target_mask, enc_mask)
        
        output = self.fc_out(target)
        return output, self_score, attn_score
    
    
class Transformer(nn.Module):
    """_summary_
    Transformer class.

    Args:
        encoder (nn.Module): Encoder module. 
            Because no default is set, one should make it and insert here.
        decoder (nn.Module): Decoder module.
            Same as encoder.
        src_pad_idx (int): Padding token index for input text.
        target_pad_idx (int): Padding token index for target text.
        device (str): devices to use.
    """    
    def __init__(self, encoder, decoder, src_pad_idx, target_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.target_pad_idx = target_pad_idx
        self.device = device
    
    # Input 문장의 <pad> 토큰에 대해 mask=0
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # Shape = [batch_size, 1, 1, src_len]
        return src_mask
    
    # Output 문장의 <pad> 토큰에 대해 mask=0
    def make_target_mask(self, target):
        # Shape of mask = [batch_size, 1, 1, target_len]
        target_pad_mask = (target != self.target_pad_idx).unsqueeze(1).unsqueeze(2)
        target_len = target.shape[1]
        
        '''
        subsequent masking example
        ---------
        1 0 0 0 0
        1 1 0 0 0
        1 1 1 0 0 
        1 1 1 1 0
        1 1 1 1 1
        ---------
        NOT TO PREDICT THE RESULT FROM THE FUTURE INFORMATIONS..
        '''
        
        target_sub_mask = torch.tril(torch.ones(
            (target_len, target_len), device=self.device)).bool()
        
        target_mask = target_pad_mask & target_sub_mask
        # size: [batch_size, 1, target_len, target_len]
        return target_mask
        
    def forward(self, src, target):
        
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)
        
        enc_out, enc_score = self.encoder(src, src_mask)
        
        output, dec_self_score, dec_attn_score = self.decoder(target, target_mask, 
                                                              enc_out, src_mask)
        
        return output, dec_attn_score
        
        
        