import torch
import torch.nn.functional as F
from .model import GPT, GPTConfig
from .loader import DataLoader
import math

# 500,000 tokens per batch in GPT-2 training / (4 batch size * 1024 context window)
GRAD_ACCUMULATION_STEPS = 122 
MAX_STEPS = 19073      # ~1 Epoch of 10B tokens
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
    
    torch.set_float32_matmul_precision('high')

    config = GPTConfig()
    model = GPT(config)
    model.to(device)
    
    model = torch.compile(model)
    train_loader = DataLoader(B=4, T=1024, split="train")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.0)
    
    for step in range(MAX_STEPS):

        accumulated_loss = 0.0
        
        optimizer.zero_grad()
        
        for micro_step in range(GRAD_ACCUMULATION_STEPS):
            
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                logits = model(x) # (B, T, V)
            
                B, T, C = logits.shape
                logits_flat = logits.view(B * T, C)
                y_flat = y.view(B * T)
                
                loss = F.cross_entropy(logits_flat, y_flat)
                
            loss = loss / GRAD_ACCUMULATION_STEPS

            accumulated_loss += loss.item()
                
            loss.backward()
            

        lr = get_lr(step)
        
        optimizer.param_groups[0]['lr'] = get_lr(step)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()


        if step % 10 == 0 or step == MAX_STEPS - 1:
            print(f"Step {step:05d} | LR: {lr:.6f} | Loss: {accumulated_loss.item():.4f}")

if __name__ == "__main__":
    main()