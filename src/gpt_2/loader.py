import torch
import numpy as np
import os

class DataLoader:
    def __init__(self, B, T, split="train"):
        self.B = B
        self.T = T

        data_root = "scripts/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s and s.endswith('.bin')]
        self.shards = sorted([os.path.join(data_root, s) for s in shards])

        self.current_shard_idx = 0
        self.current_position = 0

        self._load_shard()

    def _load_shard(self):
        shard_path = self.shards[self.current_shard_idx]
        self.tokens = np.memmap(shard_path, dtype=np.uint16, mode='r')

    def next_batch(self):
        B, T = self.B, self.T

        if self.current_position + (B * T) + 1 > len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self._load_shard()
            self.current_position = 0

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = torch.tensor(buf[:-1].astype(np.int64), dtype=torch.long).view(B, T)
        y = torch.tensor(buf[1:].astype(np.int64), dtype=torch.long).view(B, T)
        
        self.current_position += B * T
        
        return x, y