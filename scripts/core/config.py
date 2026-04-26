import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# directories & paths
# find project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

# hardware & model config
# use cuda if available
import torch
EMBED_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EMBED_MODEL_ID = 'BAAI/bge-m3'
RERANK_MODEL_ID = 'BAAI/bge-reranker-v2-m3'

# global initializations
# Models are loaded once when this module is imported.
print(f"Loading embedder: {EMBED_MODEL_ID} on {EMBED_DEVICE}...")
embedder = SentenceTransformer(EMBED_MODEL_ID, device=EMBED_DEVICE)

print(f"Loading reranker: {RERANK_MODEL_ID} on {EMBED_DEVICE}...")
bge_reranker = CrossEncoder(RERANK_MODEL_ID, device=EMBED_DEVICE)

from .generator import T5Generator
adapter_dir = str(DATA_DIR.parent / 'best_adapters')
t5_model = T5Generator(adapter_path=adapter_dir, device=EMBED_DEVICE)
ft_gen = t5_model.ft_gen
