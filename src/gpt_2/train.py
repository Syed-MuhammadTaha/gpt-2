import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from loader import DataLoader
import math
import os
import wandb
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 500,000 tokens per batch in GPT-2 training / (8 batch size * 1024 context window)
GRAD_ACCUMULATION_STEPS = 61 
MAX_STEPS = 19073      # ~1 Epoch of 10B tokens
WARMUP_STEPS = 715
MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
CHECKPOINT_DIR = "checkpoints"


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

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    torch.set_float32_matmul_precision('high')

    config = GPTConfig()
    model = GPT(config)
    model.to(device)
    
    model = torch.compile(model)
    train_loader = DataLoader(B=8, T=1024, split="train")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    start_step=0
    ckpt_path = os.path.join(CHECKPOINT_DIR, "latest_ckpt.pt")

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        start_step = checkpoint['step'] + 1
        train_loader.load_state_dict(checkpoint['loader_position'])
        print(f"Resuming at Step {start_step}")

    wandb.init(project="gpt2-pretraining", name="run-2 (pc restart)", resume="allow", id="7o0jjh1f")
    
    for step in range(start_step, MAX_STEPS):


        accumulated_loss = 0.0
        
        optimizer.zero_grad()

        t0 = time.time()
        
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

        wandb.log({
            "loss": accumulated_loss,
            "learning_rate": lr,
            "step": step
        })

        if device == 'cuda':
            torch.cuda.synchronize()
            
        t1 = time.time()
        dt = t1 - t0

        if step % 10 == 0 or step == MAX_STEPS - 1:
            print(f"Step {step:05d} | LR: {lr:.6f} | Loss: {accumulated_loss:.4f}")

            if step > 0:
                # 499,712 tokens divided by the seconds it took
                tokens_per_sec = (4 * 1024 * GRAD_ACCUMULATION_STEPS) / dt
                
                # Seconds per step * remaining steps, converted to hours
                remaining_hours = (dt * (MAX_STEPS - step)) / 3600
                
                print(f"   ---> Speed: {tokens_per_sec:.0f} tokens/sec | Time/Step: {dt:.2f}s | Est. Remaining: {remaining_hours:.2f} hours")
                
        if step % 200 == 0 or step == MAX_STEPS:
            checkpoint = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'step': step,
                'loader_position': train_loader.state_dict()
            }
            
            temp_path = os.path.join(CHECKPOINT_DIR, "temp_ckpt.pt")
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, ckpt_path)

            print("saving ckpt....")

if __name__ == "__train__":
    train()