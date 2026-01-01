import torch
import torch.nn as nn
import torch.nn.functional as F



class MutiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size,hidden_size)

    def apply_rope(self, x, position_idx):
        seq_len = position_idx.size(1)

        inv_freq = 1./(10000 ** torch.arange(0, self.head_dim, 2).float()/self.head_dim)
        freqs = torch.einsum('bi,j->bij',position_idx.float(),inv_freq)
        emb = torch.cat((freqs, freqs),dim=-1)
        cos = torch.cos(emb).view(1,1,seq_len, self.head_dim)
        sin = torch.sin(emb).view(1,1,seq_len,self.head_dim)
        return (x*cos) + (self.rotate_half(x)*sin)
    
    def rotate_half(self,x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1),dim=-1)
    
    

        

