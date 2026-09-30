import torch
import torch.nn.functional as F
import tiktoken
from model import GPT, GPTConfig

def infer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = tiktoken.get_encoding("gpt2")
    
    config = GPTConfig()
    model = GPT(config)
    
    ckpt_path = "checkpoints/uncompiled_model.pt"
    clean_state_dict = torch.load(ckpt_path, map_location=device)
    
    model.load_state_dict(clean_state_dict)
    model.eval()
    model.to(device)

    prompt = "The capital of France is"
    input_ids = enc.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    max_new_tokens = 1000
    eot_token_id = enc.eot_token
    
    print(prompt, end="", flush=True)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x_crop = x if x.size(1) <= config.block_size else x[:, -config.block_size:]
            logits = model(x_crop)

            next_token_logits = logits[:, -1, :]

            next_token_logits = next_token_logits[:, :50257] # stripping end token as they were padded ot support fast tensor operations
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
                        
            if next_token_id.item() == eot_token_id:
                break
                
            print(enc.decode([next_token_id.item()]), end="", flush=True)
            x = torch.cat((x, next_token_id), dim=1)
            
            
    print()

if __name__ == "__main__":
    infer()