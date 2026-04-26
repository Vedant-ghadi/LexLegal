import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

import json

nb_path = str(BASE_DIR / 'titanragv1_rarity.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    s = "".join(c.get('source', []))
    if 'class BM25Idx:' in s and 'from rank_bm25 import BM25Okapi' in s:
        print(f"Found in cell {i}")
        
        # We know exactly where the BM25 code starts and ends in the dump
        start_idx = s.find('# === BM25 + RRF ===')
        end_idx = s.find("print('Chunkers + HyDE + BM25 ready")
        
        new_bm25 = """# === FAST SPARSE BM25 + RRF ===
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

class BM25Idx:
    def __init__(self):
        self.chunks, self.tf_sparse, self.idf, self.doc_len, self.avgdl = [], None, None, None, None
        self.k1 = 1.5
        self.b = 0.75
        self.vectorizer = None

    def build(self, ch):
        self.chunks = ch
        if not ch: return
        self.vectorizer = CountVectorizer(token_pattern=r'(?u)\\\\b\\\\w+\\\\b', lowercase=True)
        self.tf_sparse = self.vectorizer.fit_transform([c['text'] for c in ch]).tocsc()
        
        N = len(ch)
        df = np.array((self.tf_sparse > 0).sum(axis=0)).ravel()
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        idf[idf < 0] = 0
        self.idf = idf
        
        self.doc_len = np.array(self.tf_sparse.sum(axis=1)).ravel()
        self.avgdl = self.doc_len.mean() if N > 0 else 1.0

    def search(self, q, k=150):
        if self.tf_sparse is None: return []
        q_vec = self.vectorizer.transform([q])
        q_terms = q_vec.indices
        if len(q_terms) == 0: return []
        
        tf_q = self.tf_sparse[:, q_terms].toarray()
        idf_q = self.idf[q_terms]
        
        den = tf_q + self.k1 * (1 - self.b + self.b * (self.doc_len[:, None] / self.avgdl))
        num = tf_q * (self.k1 + 1)
        
        scores = np.sum(idf_q * (num / den), axis=1)
        ix = scores.argsort()[-k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in ix if scores[idx] > 0]

def rrf(d, b, k=60):
    s = {}
    for r, (idx, _) in enumerate(d): s[idx] = s.get(idx, 0) + 1.0/(k+r+1)
    for r, (idx, _) in enumerate(b): s[idx] = s.get(idx, 0) + 1.0/(k+r+1)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)

"""
        s = s[:start_idx] + new_bm25 + s[end_idx:]
        s = s.replace("from rank_bm25 import BM25Okapi", "")
        
        c['source'] = [s]
        
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print("BM25 patched.")
