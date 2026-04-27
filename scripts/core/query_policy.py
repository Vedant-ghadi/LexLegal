import re as _re
import json as _json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

# pilot signals - multi-signal feature vector from pilot retrieval
@dataclass
class PilotSignals:
    max_dense: float = 0.0
    mean_dense: float = 0.0
    std_dense: float = 0.0
    top1_top5_margin: float = 0.0
    bm25_max: float = 0.0
    bm25_mean: float = 0.0
    dense_bm25_overlap: float = 0.0
    unique_doc_count: int = 0
    entropy: float = 0.0
    query_rarity: float = 0.0
    max_term_rarity: float = 0.0

# route decision - retrieval strategy chosen by the router
@dataclass
class RouteDecision:
    route: str = 'normal'       # narrow_precise | normal | broad_expand | reject_or_fallback
    pool: int = 150
    top_k: int = 3
    use_hyde: bool = True
    use_bm25: bool = True
    use_xref: bool = True
    use_span_refine: bool = True
    fallback_enabled: bool = True
    xref_hops: int = 1

# adaptive query policy - pilot -> signals -> route -> fallback
class QueryPolicy:

    def __init__(self, emb, bm25_idx, faiss_index, chunks,
                 t_high=0.75, t_low=0.35, m_high=0.15, s_high=0.20, d_high=5,
                 reranker_threshold=0.3, log_file=None):
        self.emb = emb
        self.bm25_idx = bm25_idx
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.t_high, self.t_low = t_high, t_low
        self.m_high, self.s_high, self.d_high = m_high, s_high, d_high
        self.reranker_threshold = reranker_threshold
        self._log_fh = open(log_file, 'a', encoding='utf-8') if log_file else None
        self._route_counts = defaultdict(int)
        self._qcount = 0

    def pilot_retrieval(self, query, pilot_k=10):
        """Fast pilot retrieval via FAISS to gather routing signals."""
        qv = self.emb.encode([query], normalize_embeddings=True)[0]
        qv_np = np.array([qv], dtype=np.float32)
        scores, indices = self.faiss_index.search(qv_np, min(pilot_k, self.faiss_index.ntotal))
        dense_scores, dense_docs, dense_idxs = [], [], []
        for j in range(len(indices[0])):
            idx = int(indices[0][j])
            if idx < 0 or idx >= len(self.chunks): continue
            dense_scores.append(float(scores[0][j]))
            dense_docs.append(self.chunks[idx]['file_path'])
            dense_idxs.append(idx)

        bm25_scores, bm25_docs = [], []
        if self.bm25_idx:
            bm25_hits = self.bm25_idx.search(query, pilot_k)
            for bi, bs in bm25_hits:
                if 0 <= bi < len(self.chunks):
                    bm25_scores.append(bs)
                    bm25_docs.append(self.chunks[bi]['file_path'])

        return dense_scores, dense_docs, dense_idxs, bm25_scores, bm25_docs

    def compute_signals(self, query, dense_scores, bm25_scores, dense_docs, bm25_docs):
        """Extract 11 features from pilot results to drive routing."""
        sig = PilotSignals()
        if dense_scores:
            sig.max_dense = max(dense_scores)
            sig.mean_dense = float(np.mean(dense_scores))
            sig.std_dense = float(np.std(dense_scores))
            sig.top1_top5_margin = (
                dense_scores[0] - dense_scores[min(4, len(dense_scores)-1)]
                if len(dense_scores) > 1 else 0.0)
        if bm25_scores:
            mx = max(bm25_scores) if bm25_scores else 1.0
            norm_bm25 = [s / (mx + 1e-9) for s in bm25_scores]
            sig.bm25_max = max(norm_bm25)
            sig.bm25_mean = float(np.mean(norm_bm25))
        # dense-bm25 agreement
        if dense_docs and bm25_docs:
            k = min(len(dense_docs), len(bm25_docs), 10)
            sig.dense_bm25_overlap = len(set(dense_docs[:k]) & set(bm25_docs[:k])) / max(k, 1)
        # entropy of score distribution
        if dense_scores and len(dense_scores) > 1:
            arr = np.array(dense_scores, dtype=np.float64)
            arr = arr - arr.max()
            exp_arr = np.exp(arr)
            p = exp_arr / (exp_arr.sum() + 1e-9)
            sig.entropy = float(-np.sum(p * np.log(p + 1e-9)))
        # document diversity
        sig.unique_doc_count = len(set(dense_docs + bm25_docs))
        # statistical query rarity using BM25 IDF
        tokens = _re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        if tokens and self.bm25_idx and hasattr(self.bm25_idx, 'bm25') and self.bm25_idx.bm25:
            idf_dict = self.bm25_idx.bm25.idf
            N = self.bm25_idx.bm25.corpus_size
            max_unseen_idf = math.log((N + 1) / 1) + 1

            rarity_scores = []
            for t in tokens:
                if t in idf_dict:
                    freq = N / math.exp(idf_dict[t])
                    r = math.log((N + 1) / (freq + 1)) + 1
                    rarity_scores.append(r)
                else:
                    rarity_scores.append(max_unseen_idf)

            if rarity_scores:
                sig.query_rarity = float(sum(rarity_scores))
                sig.max_term_rarity = float(max(rarity_scores))
        return sig

    def route_query(self, signals):
        """Map 11 pilot signals to one of 4 retrieval strategies."""
        # reject: both retrievers weak AND no rare legal anchors
        if signals.max_dense < self.t_low and signals.bm25_max < self.t_low and signals.max_term_rarity < 10.0:
            return RouteDecision(route='reject_or_fallback', pool=0, top_k=0,
                                use_hyde=False, use_bm25=False, use_xref=False,
                                use_span_refine=False, fallback_enabled=True)

        # narrow precise: highly rare anchor word OR very confident top-1
        if signals.max_term_rarity > 11.0 or (signals.max_dense > self.t_high and signals.top1_top5_margin > self.m_high and signals.unique_doc_count <= 2):
            return RouteDecision(route='narrow_precise', pool=50, top_k=3,
                                use_hyde=False, use_bm25=True, use_xref=False,
                                use_span_refine=False, fallback_enabled=False)

        # broad expand: generic words, high variance, or many unique docs
        if signals.query_rarity < 15.0 or signals.std_dense > self.s_high or signals.unique_doc_count >= self.d_high:
            return RouteDecision(route='broad_expand', pool=200, top_k=5,
                                use_hyde=True, use_bm25=True, use_xref=True,
                                use_span_refine=False, fallback_enabled=True)

        # normal: balanced default
        return RouteDecision(route='normal', pool=100, top_k=4,
                            use_hyde=True, use_bm25=True, use_xref=True,
                            use_span_refine=False, fallback_enabled=True)

    def should_fallback(self, top_reranker_score, signals):
        """Post-reranking safety net: triggers fallback if confidence too low."""
        if top_reranker_score < self.reranker_threshold:
            return True
        if signals.dense_bm25_overlap < 0.1 and signals.entropy > 2.0:
            return True
        return False

    def get_fallback_config(self, current_pool):
        """Fallback: double pool, force all features on, deeper xref traversal."""
        rd = RouteDecision(route='fallback', pool=min(max(current_pool*2, 200), 300),
                          top_k=5, use_hyde=True, use_bm25=True, use_xref=True,
                          use_span_refine=False, fallback_enabled=False)
        rd.xref_hops = 2
        return rd

    @staticmethod
    def simplify_query(query):
        """Remove filler words, keep legal entities for fallback retries."""
        filler = {'the','a','an','is','are','was','were','do','does','did',
                  'will','would','could','should','can','may','might','must',
                  'has','have','had','be','been','being','it','its','this',
                  'that','these','those','what','which','who','whom','how',
                  'when','where','why','if','or','and','but','so','for',
                  'to','of','in','on','at','by','with','from','about',
                  'into','through','during','before','after','above','below'}
        tokens = query.split()
        kept = [t for t in tokens if t.lower() not in filler or t[0].isupper()]
        return ' '.join(kept) if kept else query

    def log_query(self, query, signals, route, fallback_triggered,
                  n_results, elapsed_ms, retrieved_spans=None):
        """Log every routing decision to JSONL for future ML training."""
        if not self._log_fh: return
        self._qcount += 1
        entry = {'query_id': self._qcount, 'query': query,
                 'signals': asdict(signals), 'route': route.route,
                 'config': {'pool': route.pool, 'top_k': route.top_k,
                            'use_hyde': route.use_hyde, 'use_bm25': route.use_bm25,
                            'use_xref': route.use_xref, 'use_span_refine': route.use_span_refine},
                 'fallback_triggered': fallback_triggered,
                 'n_results': n_results, 'time_ms': round(elapsed_ms, 1)}
        if retrieved_spans:
            entry['retrieved_spans'] = [
                {'file_path': sp['file_path'],
                 'char_offset': sp['char_offset'],
                 'char_end': sp.get('char_end', sp['char_offset'] + len(sp.get('text','')))
                } for sp in retrieved_spans]
        self._log_fh.write(_json.dumps(entry, ensure_ascii=False) + '\n')
        self._log_fh.flush()

    def print_route_summary(self):
        total = sum(self._route_counts.values())
        if total == 0: print('  No queries routed yet.'); return
        print(f'\nRoute Distribution ({total} queries):')
        for r, c in sorted(self._route_counts.items()):
            print(f'  {r:<20} {c:>4}  ({100*c/max(total,1):.1f}%)')

    def close(self):
        if self._log_fh: self._log_fh.close(); self._log_fh = None
