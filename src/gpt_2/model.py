from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F
import torch

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12 
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd), 
            "wpe": nn.Embedding(config.block_size, config.n_embd), # T,d; learned positional embeddings different from BERT's sin cos embeddings; better alternative is RoPE
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd), # d
        })

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # d,V
        
        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)

        tok_emb = self.transformer.wte(idx) # B, T, d
        x = tok_emb + pos_emb # B, T, d
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x) # B, T, d
        logits = self.lm_head(x) # B, T, V
        return logits




class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # d
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd) # d
        self.mlp = MLP(config) # d,d

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # t, d*3
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # t,d

    def forward(self, x):
        B, T, C = x.size()

        kqv = self.c_attn(x) # t, d*3
        k,q,v = kqv.split(self.n_embd, dim=2) # Prev (B, T, d*3) -> Now (B, T, d), (B, T, d), (B, T, d)
        
        
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim) # FROM (B, T, d) -> (B, T, n_head, d/n_head)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)

        q = q.permute(0, 2, 1, 3) # B, n_head, T, d/n_head
        k = k.permute(0, 2, 1, 3) # B, n_head, T, d/n_head
        v = v.permute(0, 2, 1, 3) # B, n_head, T, d/n_head

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # B, n_head, T, d/n_head

        y = y.permute(0, 2, 1, 3) # B, T, n_head, d/n_head
        y = y.reshape(B, T, C) # FROM (B, n_head, T, d/n_head) -> (B, T, d)
        y = self.c_proj(y) # B, T, d
        return y

       
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # t, d*4
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # t, d

    def forward(self, x):
        x = self.c_fc(x) # t, d*4
        x = F.gelu(x, approximate="tanh") # t, d*4
        x = self.c_proj(x) # t, d
        return x
