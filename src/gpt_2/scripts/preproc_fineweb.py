import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
local_dir = "edu_fineweb10B"
shard_size = int(1e8) # 100 Million tokens per .bin file
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train",)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # = 50256

def tokenize(doc):
    """Tokenizes a single document and appends the End of Text token."""
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16) # integers consume less space and besides vocab size is 50257 that encodes into an integer
    return tokens_np

def write_datafile(filename, tokens_np):
    """Writes the numpy array directly to a raw binary file."""
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())


if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() - 2) 
    
    token_count = 0
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    
    print(f"Tokenizing on {nprocs} CPU cores...")
    with mp.Pool(nprocs) as pool:

        for tokens in tqdm(pool.imap(tokenize, fw, chunksize=16), total=len(fw)):
            

            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
            else:

                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.bin")
                

                remainder = shard_size - token_count
                all_tokens_np[token_count : shard_size] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                

                shard_index += 1
                token_count = len(tokens) - remainder
                all_tokens_np[0 : token_count] = tokens[remainder:]

    # partial shard
    if token_count > 0:

        all_tokens_np[token_count] = eot
        token_count += 1

        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])
        