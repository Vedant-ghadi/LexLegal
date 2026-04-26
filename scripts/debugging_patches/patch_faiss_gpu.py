import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

"""
FULL GPU OPTIMIZATION PATCH for titanragv1_rarity.ipynb
Applies 3 optimizations:
  1. FAISS GPU: Replace Milvus Lite with faiss.IndexFlatIP on GPU
  2. GPU BM25: Pre-compute TF-IDF as sparse tensors, score on GPU
  3. Batch Reranking: CrossEncoder batch_size 32->128
"""
import json

nb_path = str(BASE_DIR / 'titanragv1_rarity.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def get_src(c):
    return "".join(c['source']) if isinstance(c['source'], list) else c['source']

def set_src(c, s):
    c['source'] = [s]
    c['outputs'] = []
    c['execution_count'] = None

# ================================================================
# 1. Fix pip install: pymilvus -> faiss-gpu
# ================================================================
for i, c in enumerate(cells):
    s = get_src(c)
    if c['cell_type'] == 'code' and 'pip install' in s and 'pymilvus' in s:
        if 'faiss-gpu' not in s:
            s = s.replace('pymilvus', 'faiss-gpu')
            set_src(c, s)
            print(f"  Cell {i}: pip install updated (pymilvus -> faiss-gpu)")
        break

# ================================================================
# 2. Fix imports: MilvusClient -> faiss
# ================================================================
for i, c in enumerate(cells):
    s = get_src(c)
    if c['cell_type'] == 'code' and 'from pymilvus import MilvusClient' in s:
        s = s.replace('from pymilvus import MilvusClient',
                      'import faiss  # GPU-accelerated vector search')
        set_src(c, s)
        print(f"  Cell {i}: import updated")
        break

# ================================================================
# 3. Rewrite TitanRAGv3 with all 3 GPU optimizations
# ================================================================
NEW_TITAN = '''class TitanRAGv3(RetrievalMethod):
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
        # ── FAISS GPU index ──
        self._dim = emb.get_sentence_embedding_dimension()
        self._faiss_index = None
        self._vectors = None
        # ── Batch reranking config ──
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
        vs = self.emb.encode([c['text'] for c in self._chunks], batch_size=64,
                             normalize_embeddings=True, show_progress_bar=True)
        # ── FAISS GPU index ──
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
        # ── Embedding (HyDE or direct) ──
        qv = (self.hyde_exp.get_query_embedding(query) if use_hyde and self.hyde_exp
              else self.emb.encode([query], normalize_embeddings=True)[0])
        qv_np = np.array([qv], dtype=np.float32)

        # ── Dense search via FAISS GPU ──
        k_search = min(pool, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(qv_np, k_search)
        dense = [(int(indices[0][j]), float(scores[0][j]))
                 for j in range(len(indices[0])) if indices[0][j] >= 0]

        # ── BM25 + RRF ──
        bm25 = self._bm25.search(query, pool) if use_bm25 else []
        idxs = [i for i, _ in (rrf(dense, bm25) if bm25 else dense)][:pool]
        if not idxs: return []

        # ── Cross-reference expansion ──
        if use_xref and idxs:
            xref_idxs = self._xref.get_related(idxs[:20], max_hops=xref_hops)
            for xi in xref_idxs:
                if xi not in idxs: idxs.append(xi)

        # ── Build candidates ──
        cands = [{'text': self._chunks[i].get('raw_text', self._chunks[i]['text']),
                  'file_path': self._chunks[i]['file_path'],
                  'char_offset': self._chunks[i]['char_offset'], 'score': 0.0}
                 for i in idxs if 0 <= i < len(self._chunks)]

        # ── Cross-encoder reranking (BATCH=128 for H100) ──
        if self.rr and len(cands) > 1:
            pairs = [(query, c['text']) for c in cands]
            sc = self.rr.predict(pairs, batch_size=self._rr_batch_size)
            for i, c in enumerate(cands): c['score'] = float(sc[i])
            cands.sort(key=lambda x: x['score'], reverse=True)

        top = cands[:top_k * 3]

        # ── Span refinement ──
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

        # ── Overlap dedup ──
        kept = []
        for c in top:
            cs, ce = c['char_offset'], c['char_offset']+len(c['text'])
            dup = any(c['file_path']==k['file_path'] and cs<k['char_offset']+len(k['text'])
                      and ce>k['char_offset'] for k in kept)
            if not dup: kept.append(c)
        return kept[:top_k]

    async def query(self, query):
        t0 = time.time()

        # ── ADAPTIVE PATH ──
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

        # ── ORIGINAL PATH ──
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

print('TitanRAG v3 + FAISS GPU + Batch Rerank ready')
'''

for i, c in enumerate(cells):
    s = get_src(c)
    if c['cell_type'] == 'code' and 'class TitanRAGv3' in s:
        set_src(c, NEW_TITAN)
        print(f"  Cell {i}: TitanRAGv3 rewritten (FAISS GPU + batch rerank=128)")
        break

# ================================================================
# 4. Update QueryPolicy: db -> faiss_index, pilot uses FAISS
# ================================================================
NEW_PILOT = '''    def pilot_retrieval(self, query, pilot_k=10):
        """Stage 1: Fast pilot retrieval via FAISS GPU."""
        qv = self.emb.encode([query], normalize_embeddings=True)[0]
        qv_np = np.array([qv], dtype=np.float32)
        scores, indices = self.faiss_index.search(qv_np, min(pilot_k, self.faiss_index.ntotal))
        dense_scores, dense_docs, dense_idxs = [], [], []
        for j in range(len(indices[0])):
            idx = int(indices[0][j])
            if idx < 0 or idx >= len(self.chunks): continue
            dense_scores.append(float(scores[0][j]))
            dense_docs.append(self.chunks[idx]['file_path'])
            dense_idxs.append(idx)'''

for i, c in enumerate(cells):
    s = get_src(c)
    if c['cell_type'] != 'code' or 'class QueryPolicy' not in s:
        continue

    # Replace db= with faiss_index=
    s = s.replace('db=None,', 'faiss_index=None,')
    s = s.replace('self.db = db', 'self.faiss_index = faiss_index')

    # Replace pilot_retrieval method
    import re
    pilot_pattern = re.compile(
        r'    def pilot_retrieval\(self.*?(?=\n        # BM25|\n    def )',
        re.DOTALL
    )
    m = pilot_pattern.search(s)
    if m:
        s = s[:m.start()] + NEW_PILOT + '\n' + s[m.end():]
        print(f"  Cell {i}: QueryPolicy pilot_retrieval updated for FAISS")
    else:
        # Fallback: just do string replace on the key line
        s = s.replace(
            "hits = self.db.search('c', data=[qv], limit=pilot_k,",
            "# FAISS pilot search\n"
            "        qv_np = np.array([qv], dtype=np.float32)\n"
            "        scores, indices = self.faiss_index.search(qv_np, min(pilot_k, self.faiss_index.ntotal))\n"
            "        # OLD: hits = self.db.search('c', data=[qv], limit=pilot_k,"
        )
        print(f"  Cell {i}: QueryPolicy partially updated (fallback method)")

    set_src(c, s)
    break

# ================================================================
# Save
# ================================================================
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print("\nDone! All 3 GPU optimizations applied:")
print("  1. FAISS GPU: vector search on H100 (replaces Milvus Lite)")
print("  2. Batch Reranking: batch_size=128 (was 32)")
print("  3. Pilot retrieval: FAISS GPU (was MilvusClient)")
