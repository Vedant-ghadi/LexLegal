import time
import numpy as np
import faiss
import torch
import gc
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from legalbenchrag.benchmark_types import QueryResponse, RetrievedSnippet

from .chunking import sliding_chunk, semantic_chunk, proposition_chunk, ensemble_chunk
from .search_components import FTHyDE, BM25Idx, rrf
from .cross_reference import CrossRefGraph
# Note: QueryPolicy is bypassed in this executable to focus on raw vector performance

class RetrievalMethod:
    async def ingest_document(self, doc): pass
    async def sync_all_documents(self): pass
    async def query(self, query: str) -> QueryResponse: pass


class TitanRAGv3(RetrievalMethod):
    def __init__(self, emb, rr=None, hyde_exp=None,
                 top_k=3, pool=150, chunker='ensemble',
                 use_bm25=True, use_hyde=True,
                 use_span_refine=True, use_ctx=True,
                 use_xref_graph=True,
                 use_adaptive_policy=True,
                 log_file=None):
        self.emb, self.rr, self.hyde_exp = emb, rr, hyde_exp
        self.top_k, self.pool, self.chunker = top_k, pool, chunker
        self.use_bm25 = use_bm25
        self.use_hyde = use_hyde and hyde_exp is not None
        self.use_span_refine, self.use_ctx = use_span_refine, use_ctx
        self.use_xref = use_xref_graph
        self.use_adaptive = use_adaptive_policy
        self._log_file = log_file
        self._docs, self._chunks = {}, []
        self._bm25 = BM25Idx()
        self._xref = CrossRefGraph()
        self._policy = None
        # faiss gpu index
        self._dim = emb.get_sentence_embedding_dimension()
        self._faiss_index = None
        self._vectors = None
        # batch reranking config
        self._rr_batch_size = 128  # aggressive batching for H100

    async def ingest_document(self, doc):
        self._docs[doc.file_path] = doc.content
        if self.chunker == 'proposition': raw = proposition_chunk(doc.content, doc.file_path)
        elif self.chunker == 'semantic': raw = semantic_chunk(doc.content, doc.file_path)
        elif self.chunker == 'ensemble': raw = ensemble_chunk(doc.content, doc.file_path)
        else: raw = sliding_chunk(doc.content, doc.file_path)
        for c in raw:
            c['raw_text'] = c['text']
            if self.use_ctx:
                nm = c['file_path'].split('/')[-1].replace('.txt','')
                pct = int(100 * c['char_offset'] / max(len(doc.content), 1))
                c['text'] = f'[{nm}|{pct}%] {c["text"]}'
        self._chunks.extend(raw)

    async def sync_all_documents(self):
        if not self._chunks: return
        vs = self.emb.encode([c['text'] for c in self._chunks], batch_size=16,
                             normalize_embeddings=True, show_progress_bar=True)
        # faiss gpu index
        self._vectors = np.array(vs, dtype=np.float32)
        cpu_index = faiss.IndexFlatIP(self._dim)
        if faiss.get_num_gpus() > 0:
            gpu_res = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
        else:
            self._faiss_index = cpu_index
        self._faiss_index.add(self._vectors)

        if self.use_bm25: self._bm25.build(self._chunks)
        if self.use_xref:
            for i, c in enumerate(self._chunks): self._xref.add_chunk(i, c)
            self._xref.build_edges()
            print(f'  XRef graph: {self._xref.stats()}')
        if self.use_adaptive:
            self._policy = QueryPolicy(
                emb=self.emb, bm25_idx=self._bm25, faiss_index=self._faiss_index,
                chunks=self._chunks, log_file=self._log_file)
        gpu_str = "GPU" if faiss.get_num_gpus() > 0 else "CPU"
        print(f'  {len(self._chunks)} chunks ({self.chunker})'
              f'{" | Adaptive ON" if self.use_adaptive else ""}'
              f' | FAISS {gpu_str} | Rerank batch={self._rr_batch_size}')

    def _run_retrieval(self, query, pool, top_k, use_hyde, use_bm25,
                       use_xref, use_span_refine, xref_hops=1):
        # embedding (hyde or direct)
        qv = (self.hyde_exp.get_query_embedding(query) if use_hyde and self.hyde_exp
              else self.emb.encode([query], normalize_embeddings=True)[0])
        qv_np = np.array([qv], dtype=np.float32)

        # dense search via faiss gpu
        k_search = min(pool, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(qv_np, k_search)
        dense = [(int(indices[0][j]), float(scores[0][j]))
                 for j in range(len(indices[0])) if indices[0][j] >= 0]

        # bm25 + rrf
        bm25 = self._bm25.search(query, pool) if use_bm25 else []
        idxs = [i for i, _ in (rrf(dense, bm25) if bm25 else dense)][:pool]
        if not idxs: return []

        # cross-reference expansion
        if use_xref and idxs:
            xref_idxs = self._xref.get_related(idxs[:20], max_hops=xref_hops)
            for xi in xref_idxs:
                if xi not in idxs: idxs.append(xi)

        # build candidates
        cands = [{'text': self._chunks[i].get('raw_text', self._chunks[i]['text']),
                  'file_path': self._chunks[i]['file_path'],
                  'char_offset': self._chunks[i]['char_offset'], 'score': 0.0}
                 for i in idxs if 0 <= i < len(self._chunks)]

        # cross-encoder reranking (batch128 for H100) ──
        if self.rr and len(cands) > 1:
            pairs = [(query, c['text']) for c in cands]
            sc = self.rr.predict(pairs, batch_size=self._rr_batch_size)
            for i, c in enumerate(cands): c['score'] = float(sc[i])
            cands.sort(key=lambda x: x['score'], reverse=True)

        top = cands[:top_k * 3]

        # span refinement
        if use_span_refine and top:
            qv2 = self.emb.encode([query], normalize_embeddings=True)
            ref = []
            for c in top:
                ss = sent_tokenize(c['text'])
                if len(ss) <= 1: ref.append(c); continue
                sv = self.emb.encode(ss, normalize_embeddings=True, show_progress_bar=False)
                sims = cosine_similarity(qv2, sv)[0]
                bi = int(np.argmax(sims))
                si, ei = max(0, bi-1), min(len(ss), bi+2)
                rt = ' '.join(ss[si:ei])
                od = c['text'].find(ss[si][:30])
                ref.append({**c, 'text': rt, 'char_offset': c['char_offset']+max(0,od),
                            'score': c['score']+float(sims[bi])})
            top = ref

        # overlap dedup
        kept = []
        for c in top:
            cs, ce = c['char_offset'], c['char_offset']+len(c['text'])
            dup = any(c['file_path']==k['file_path'] and cs<k['char_offset']+len(k['text'])
                      and ce>k['char_offset'] for k in kept)
            if not dup: kept.append(c)
        return kept[:top_k]

    async def query(self, query):
        t0 = time.time()

        # adaptive path
        if self.use_adaptive and self._policy:
            dsc, ddoc, didx, bsc, bdoc = self._policy.pilot_retrieval(query, pilot_k=10)
            signals = self._policy.compute_signals(query, dsc, bsc, ddoc, bdoc)
            route = self._policy.route_query(signals)
            self._policy._route_counts[route.route] += 1

            if route.route == 'reject_or_fallback':
                simplified = QueryPolicy.simplify_query(query)
                fb = self._policy.get_fallback_config(150)
                results = self._run_retrieval(
                    simplified, fb.pool, fb.top_k,
                    fb.use_hyde, fb.use_bm25, fb.use_xref, fb.use_span_refine,
                    getattr(fb, 'xref_hops', 1))
                self._policy.log_query(query, signals, route, True,
                                       len(results), (time.time()-t0)*1000, results)
                return self._build_response(results)

            results = self._run_retrieval(
                query, route.pool, route.top_k,
                route.use_hyde, route.use_bm25, route.use_xref, route.use_span_refine,
                getattr(route, 'xref_hops', 1))

            fallback = False
            if results and route.fallback_enabled:
                top_score = results[0]['score']
                if self._policy.should_fallback(top_score, signals):
                    fallback = True
                    simplified = QueryPolicy.simplify_query(query)
                    fb = self._policy.get_fallback_config(route.pool)
                    fb_results = self._run_retrieval(
                        simplified, fb.pool, fb.top_k,
                        fb.use_hyde, fb.use_bm25, fb.use_xref, fb.use_span_refine,
                        getattr(fb, 'xref_hops', 2))
                    if fb_results and (not results or fb_results[0]['score'] > results[0]['score']):
                        results = fb_results

            self._policy.log_query(query, signals, route, fallback,
                                   len(results), (time.time()-t0)*1000, results)
            return self._build_response(results)

        # original path
        results = self._run_retrieval(
            query, self.pool, self.top_k,
            self.use_hyde, self.use_bm25, self.use_xref, self.use_span_refine)
        return self._build_response(results)

    def _build_response(self, results):
        if not results: return QueryResponse(retrieved_snippets=[])
        mx = max(c['score'] for c in results) or 1.0
        return QueryResponse(retrieved_snippets=[
            RetrievedSnippet(file_path=c['file_path'],
                span=(c['char_offset'],
                      min(c['char_offset']+len(c['text']),
                          len(self._docs.get(c['file_path'],'')))),
                score=c['score']/mx) for c in results])

    async def cleanup(self):
        self._faiss_index = None
        self._vectors = None
        if self._policy: self._policy.close()
        self._docs, self._chunks = {}, []
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()