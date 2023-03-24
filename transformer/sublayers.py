# Sublayers for the encoder and decoder layer

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """_summary_
    Scaled Dot-Product Attention Layer for multi-head attention layer. 

    Args:
        kdim (int): Dimension of Key and Query.
        dropout (float): Dropout rate. Default is '0.1' in the paper. 
    """

    def __init__(self, div_factor, dropout=0.1):
        super().__init__()
        self.div_factor = div_factor
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.div_factor

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e+9)

        scores = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(scores, V)
        return out, scores


class MultiheadAttention(nn.Module):
    """_summary_
    Multihead Attention layer for transformers

    Args:
        embed_dim (int): Dimension of the model.
        n_heads (int): Number of parallel heads. 'embed_dim // num_heads'
            becomes the dimension of Q, K, V. 
        dropout: Dropout rate. Default is '0.0' (NO DROPOUT)
        bias: Bias for protection layers.
    """

    def __init__(self, embed_dim, n_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # In the implementations, W_x is concat of W_x_i * n_heads.
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.fc = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(
            div_factor=self.head_dim ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]
        len_seq = Q.shape[1]
        residual = Q

        # Multi-heads splitting
        # (batch_size, n_seq, n_heads, head_dim) -> (batch_size, n_heads, n_seq, head_dim)
        q = self.W_q(Q).view(batch_size, -1, self.n_heads,
                             self.head_dim).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.n_heads,
                             self.head_dim).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.n_heads,
                             self.head_dim).transpose(1, 2)

        # Attention
        q, scores = self.attention(q, k, v, mask=mask)

        # Reshaping: (batch_size, n_heads, len_seq, head_dim) -> batch_size, len_seq, embed_dims
        # contiguous called for restoring memory sequences.
        q = q.transpose(1, 2).contiguous().view(batch_size, len_seq, -1)
        q = self.dropout(self.fc(q))
        # Residual
        q += residual
        # Layer normalization
        q = self.layer_norm(q)

        return q, scores


class PositionwiseFeedForward(nn.Module):
    """_summary_
    Positionwise FeedForward layers in Transformer
    Args:
        d_in (int): Input dimension = Embedding vector dimension
        d_h (int): Hidden layer dimension

    """

    def __init__(self, d_in, d_h, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.fc2(F.relu(self.fc1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
