import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = "checkpoints/latest_ckpt.pt"
clean_ckpt_path = "checkpoints/uncompiled_model.pt"

print("Loading compiled checkpoint...")
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model_state"]
unc_state_dict = {}

for k, v in state_dict.items():
    # compiling adds _orig_mod to all keys
    new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
    unc_state_dict[new_key] = v

torch.save(unc_state_dict, clean_ckpt_path)
print(f"Clean weights saved to {clean_ckpt_path}")