import torch
import torch.nn.functional as F
from .model import GPT, GPTConfig
from .loader import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import math

MAX_STEPS = 19073
WARMUP_STEPS = 715
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1

def get_lr(it):
    """Cosine learning rate decay with linear warmup."""
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS
    if it > MAX_STEPS:
        return MIN_LR
    
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (MAX_LR - MIN_LR)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    config = GPTConfig()
    model = GPT(config)
    model.to(device)
    
    train_loader = DataLoader(B=4, T=1024, split="train")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    
    max_steps = 50
    
    for step in range(max_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x) # (B, T, V)
        
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        y_flat = y.view(B * T)
        
        loss = F.cross_entropy(logits_flat, y_flat)
        
        loss.backward()
        

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if step % 10 == 0 or step == MAX_STEPS - 1:
            print(f"Step {step:05d} | LR: {lr:.6f} | Loss: {loss.item():.4f}")

        optimizer.step()

if __name__ == "__main__":
    pass