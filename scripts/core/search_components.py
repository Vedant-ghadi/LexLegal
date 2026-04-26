import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from .config import ft_gen, embedder

# multi-query hyde (fine-tuned)
class FTHyDE:
    PROMPTS = [
        'Write a legal policy passage that answers this question: {q}',
        'Generate a privacy policy paragraph addressing: {q}',
        'Draft a contractual provision that covers: {q}',
    ]
    def __init__(self, emb, enabled=True):
        self.emb, self.enabled, self._c = emb, enabled, {}
    def get_query_embedding(self, q):
        if not self.enabled: return self.emb.encode([q], normalize_embeddings=True)[0]
        if q in self._c: return self._c[q]
        texts = [q]
        for t in self.PROMPTS:
            texts.append(f'{q} {ft_gen(t.format(q=q), 128)}')
        vecs = self.emb.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        avg = np.mean(vecs, axis=0); avg = avg / np.linalg.norm(avg)
        self._c[q] = avg; return avg

hyde = FTHyDE(embedder, enabled=True)

# bm25 + rrf
class BM25Idx:
    def __init__(self): self.chunks, self.bm25 = [], None
    def build(self, ch):
        self.chunks = ch
        self.bm25 = BM25Okapi([_re.findall(r'\w+', c['text'].lower()) for c in ch])
    def search(self, q, k=150):
        if not self.bm25: return []
        sc = self.bm25.get_scores(_re.findall(r'\w+', q.lower()))
        ix = sc.argsort()[-k:][::-1]
        return [(int(i), float(sc[i])) for i in ix if sc[i] > 0]

def rrf(d, b, k=60):
    s = {}
    for r, (i, _) in enumerate(d): s[i] = s.get(i, 0) + 1.0/(k+r+1)
    for r, (i, _) in enumerate(b): s[i] = s.get(i, 0) + 1.0/(k+r+1)
    return sorted(s.items(), key=lambda x: x[1], reverse=True)