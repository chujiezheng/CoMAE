# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
    """
    
    def __init__(self, n_positions, n_embd):
        super().__init__()
        assert n_positions <= 512
        if n_embd % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got n_embd={:d})".format(n_embd))
        pe = torch.zeros(n_positions, n_embd)
        position = torch.arange(0, n_positions).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, n_embd, 2, dtype=torch.float) * -(math.log(10000.0) / n_embd)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        self.embeddings = nn.Embedding.from_pretrained(pe, freeze=True)
    
    def forward(self, position_ids):
        return self.embeddings(position_ids)


class MLP(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super().__init__()
        self.c_fc = nn.Linear(n_in, n_mid)
        self.c_proj = nn.Linear(n_mid, n_out)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


