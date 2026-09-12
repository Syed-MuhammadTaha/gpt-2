from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    seq_len: int = 1024
    n_layer: int = 12
    n_heads: int = 12
    n_embed: int = 768
    vocab_size: int = 40000


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=GPTConfig

        self.arch=nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.seq_len, config.n_embd),
            "n": nn.ModuleList([Transformer(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })

        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.arch.wte.weight = self.head.weight
