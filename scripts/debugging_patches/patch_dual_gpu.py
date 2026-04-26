import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

"""
Patch script for Dual GPU (2x A100) support.
Updates titanragv1_rarity.ipynb to load embedder on cuda:0 and reranker on cuda:1.
"""
import json

nb_path = str(BASE_DIR / 'titanragv1_rarity.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for c in nb['cells']:
    if c['cell_type'] == 'code':
        source = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        
        # Look for the model loading cell (embedder and reranker)
        if 'SentenceTransformer' in source and 'CrossEncoder' in source and 'bge_reranker' in source:
            
            # Replace the generic device selection with a dual-GPU aware one
            if "device = 'cuda' if torch.cuda.is_available() else 'cpu'" in source:
                new_device_logic = """\
n_gpus = torch.cuda.device_count()
device_emb = 'cuda:0' if n_gpus > 0 else 'cpu'
device_rr = 'cuda:1' if n_gpus > 1 else device_emb

print(f"GPUs detected: {n_gpus}")
print(f"Assigning Embedder -> {device_emb}")
print(f"Assigning Reranker -> {device_rr}")
"""
                s = source.replace("device = 'cuda' if torch.cuda.is_available() else 'cpu'", new_device_logic)
                
                # Update the instantiations
                s = s.replace("device=device", "device=device_emb", 1)  # Embedder
                s = s.replace("device=device", "device=device_rr", 1)   # Reranker
                
                c['source'] = [s]
                print("Model loading cell updated for dual GPU.")

        # Also update HyDE loading cell if it exists separately and uses 'device'
        if 'class FTHyDE' in source or 'hyde =' in source:
            s = source.replace("device=device", "device=device_emb")
            if "device.type" in s:
                s = s.replace("device.type", "'cuda'")
            c['source'] = [s]

# Add a notebook cell near the top confirming dual GPU utilization logic
for i, c in enumerate(nb['cells']):
    s = "".join(c['source']) if isinstance(c['source'], list) else c['source']
    if 'import torch' in s and 'print(torch.cuda.get_device_name' in s:
        s = s.replace("print(torch.cuda.get_device_name(0))", 
                      "for i in range(torch.cuda.device_count()):\n    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')")
        c['source'] = [s]
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print(f"Successfully patched {nb_path} for dual A100 compatibility.")
