import json
import time
import asyncio
from pathlib import Path
from collections import defaultdict
import gc
import os
import torch
from tqdm import tqdm as _tqdm
from legalbenchrag.benchmark_types import BenchmarkResult, QAResult, Document, Benchmark
from .core.titan_engine import TitanRAGv3
from .core.config import DATA_DIR, embedder, bge_reranker

# Mock ft_hyde since we are using FastBM25 & direct search primarily
from .core.search_components import FTHyDE
hyde = FTHyDE(embedder, enabled=True)


def load_bench(name):
    with open(DATA_DIR / 'benchmarks' / f'{name}.json', encoding='utf-8', errors='replace') as f:
        bench = Benchmark.model_validate_json(f.read())
    needed = {s.file_path for t in bench.tests for s in t.snippets}
    corpus = [Document(file_path=fp,
        content=(DATA_DIR / 'corpus' / fp).read_text(encoding='utf-8', errors='replace')
    ) for fp in sorted(needed) if (DATA_DIR / 'corpus' / fp).exists()]
    print(f'{name}: {len(corpus)} docs | {sum(len(d.content) for d in corpus):,} chars | {len(bench.tests)} q')
    return bench, corpus

print('imports ready')


# override run_benchmark for gpu efficiency
# The default run_benchmark uses asyncio.gather() which fires ALL queries
# simultaneously as coroutines. Since our query() uses synchronous GPU ops,
# this causes massive scheduling overhead. Sequential is 3-5x faster.
from tqdm import tqdm as _tqdm
from legalbenchrag.run_benchmark import QAResult, BenchmarkResult

async def run_benchmark(qa_gt_list, corpus, retrieval_method, *, weights=None):
    """GPU-optimized benchmark: sequential queries + tqdm progress bar."""
    # Ingest documents
    doc_map = {}
    if isinstance(corpus, dict):
        doc_map = corpus
    elif isinstance(corpus, list):
        for d in corpus:
            doc_map[d.file_path] = d.content if hasattr(d, 'content') else d
    
    # If retrieval method needs ingestion
    if hasattr(retrieval_method, '_chunks') and len(retrieval_method._chunks) == 0:
        for fp, content in doc_map.items():
            await retrieval_method.ingest_document(
                Document(file_path=fp, content=content))
        await retrieval_method.sync_all_documents()
    
    # Run queries SEQUENTIALLY with progress bar
    from legalbenchrag.run_benchmark import QAResult, BenchmarkResult
    results = []
    for qa_gt in _tqdm(qa_gt_list, desc="Queries", unit="q", ncols=80):
        query_response = await retrieval_method.query(qa_gt.query)
        results.append(QAResult(
            qa_gt=qa_gt, retrieved_snippets=query_response.retrieved_snippets
        ))
    
    return BenchmarkResult(
        qa_result_list=results,
        weights=weights if weights is not None else [1.0] * len(results),
    )

print("run_benchmark overridden: sequential + tqdm (GPU-optimized)")


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

print('TitanRAG v3 + FAISS GPU + Batch Rerank ready')


gc.collect()
for i in range(torch.cuda.device_count()): torch.cuda.empty_cache()
bench, corpus = load_bench('privacy_qa')

# configs from ablation study
runs = [
    ('ADAPTIVE POLICY (dynamic)',
     dict(top_k=4, pool=100, chunker='ensemble', use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True,
          use_adaptive_policy=True,
          log_file='/teamspace/studios/this_studio/working/policy_log_pqa.jsonl')),
]

print(f"{'Config':<35} {'P':>7} {'R':>7} {'F1':>7} {'t':>5}")
print('\u2500' * 65)

async def qe():
    best_f1 = 0
    for label, kw in runs:
        gc.collect()
        m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde, **kw)
        t0 = time.time()
        r = await run_benchmark(bench.tests, corpus, m)
        p, rc = r.avg_precision, r.avg_recall
        f1 = 2*p*rc/(p+rc) if p+rc > 0 else 0
        tag = '\U0001f3c6' if f1 > max(best_f1, 0.1851) else ('\u2705' if f1 > 0.1851 else '')
        if f1 > best_f1: best_f1 = f1
        print(f"{label:<35} {p:>7.2%} {rc:>7.2%} {f1:>7.2%} {time.time()-t0:>4.0f}s  {tag}")
        # Print route distribution for adaptive
        if hasattr(m, '_policy') and m._policy:
            m._policy.print_route_summary()
        await m.cleanup()
    print(f'\n  Previous bests:')
    print(f'    v2++ F1=18.51% (c250 k5 p100+G) | v2 F1=19.93%')
    print(f'    Best P=18.83% (c250 k3 p80)     | RCTS P=14.38%')

await qe()


gc.collect()
for i in range(torch.cuda.device_count()): torch.cuda.empty_cache()
bench, corpus = load_bench('privacy_qa')

# ablation grid
# Each row: (label, kwargs dict)
# Best prior configs + chunker variations + adaptive toggle
ablation_grid = [
    # chunker comparison (fixed config: k4, p=100) ──
    ('sliding   c250 k4 p100',
     dict(chunker='sliding',  top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('semantic  c250 k4 p100',
     dict(chunker='semantic', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('ensemble  c250 k4 p100',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),

    # pool / k sweep (ensemble chunker)
    ('ensemble  c250 k3 p80',
     dict(chunker='ensemble', top_k=3, pool=80,  use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('ensemble  c250 k5 p100',
     dict(chunker='ensemble', top_k=5, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('ensemble  c250 k5 p150',
     dict(chunker='ensemble', top_k=5, pool=150, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),

    # feature toggles (ensemble k4 p=100) ──
    ('- no HyDE',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=False,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('- no BM25',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=False, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('- no ctx headers',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=False, use_xref_graph=True, use_adaptive_policy=False)),
    ('- no xref graph',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=False, use_adaptive_policy=False)),
    ('+ span refine ON',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=True,  use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),

    # adaptive policy (best chunker configs)
    ('ADAPTIVE sliding',
     dict(chunker='sliding',  top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=True)),
    ('ADAPTIVE ensemble',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=True)),
]

print(f"{'#':<3} {'Config':<28} {'P':>7} {'R':>7} {'F1':>7} {'Chunks':>7} {'t':>5}")
print('\u2500' * 68)

async def grid_ablation():
    results = []
    best_f1, best_label = 0, ''
    for idx, (label, kw) in enumerate(ablation_grid, 1):
        gc.collect()
        for g in range(torch.cuda.device_count()): torch.cuda.empty_cache()
        m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde, **kw)
        t0 = time.time()
        r = await run_benchmark(bench.tests, corpus, m)
        p, rc = r.avg_precision, r.avg_recall
        f1 = 2*p*rc/(p+rc) if p+rc > 0 else 0
        nc = len(m._chunks)
        el = time.time() - t0
        tag = ''
        if f1 > best_f1: best_f1 = f1; best_label = label; tag = ' \u2b50'
        print(f"{idx:<3} {label:<28} {p:>7.2%} {rc:>7.2%} {f1:>7.2%} {nc:>7} {el:>4.0f}s{tag}")
        results.append({'label': label, 'p': round(p,4), 'r': round(rc,4),
                        'f1': round(f1,4), 'chunks': nc})
        # Print routes for adaptive runs
        if hasattr(m, '_policy') and m._policy:
            m._policy.print_route_summary()
        await m.cleanup()

    # summary
    print(f'\n{"="*68}')
    print(f'  BEST CONFIG: {best_label} (F1={best_f1:.2%})')
    print(f'  Prior: v2++=18.51% | v2=19.93% | RCTS P=14.38%')
    print(f'{"="*68}')

    # leaderboard
    results.sort(key=lambda x: x['f1'], reverse=True)
    print(f'\n{"Rank":<5} {"Config":<28} {"P":>7} {"R":>7} {"F1":>7}')
    print('\u2500' * 55)
    medals = ['\U0001f947','\U0001f948','\U0001f949']
    for i, r in enumerate(results):
        m = medals[i] if i < 3 else f'{i+1}.'
        print(f"{m:<5} {r['label']:<28} {r['p']:>7.2%} {r['r']:>7.2%} {r['f1']:>7.2%}")

await grid_ablation()


# final eval: cuad, maud, contractnli (memory-safe)
import gc, time, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DATASETS = ['cuad', 'maud', 'contractnli']

BEST = dict(
    top_k=4, pool=100, chunker='sliding',
    use_bm25=True, use_hyde=True,
    use_span_refine=False, use_ctx=True,
    use_xref_graph=True, use_adaptive_policy=False
)

print(f"{'Dataset':<15} {'P':>7} {'R':>7} {'F1':>7} {'Chunks':>7} {'Queries':>7} {'t':>5}")
print('─' * 65)

async

async def safe_eval():
    for ds in DATASETS:
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
            bench, corpus = load_bench(ds)
        except Exception as e:
            print(f"{ds:<15} SKIPPED: {e}")
            continue
        
        # Use smaller embedding batch for large datasets
        m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde, **BEST)
        t0 = time.time()
        r = await run_benchmark(bench.tests, corpus, m)
        p, rc = r.avg_precision, r.avg_recall
        f1 = 2*p*rc/(p+rc) if p+rc > 0 else 0
        nq = len(bench.tests)
        nc = len(m._chunks) if hasattr(m, '_chunks') else 0
        print(f"{ds:<15} {p:>7.2%} {rc:>7.2%} {f1:>7.2%} {nc:>7} {nq:>7} {time.time()-t0:>4.0f}s")
        await m.cleanup()
        gc.collect()
        torch.cuda.empty_cache()

await safe_eval()


from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
gc.collect()
bench, corpus = load_bench('privacy_qa')
cmap = {d.file_path: d.content for d in corpus}
m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde,
               top_k=3, pool=150, chunker='ensemble',
               use_bm25=True, use_hyde=True, use_span_refine=True, use_ctx=True)
for doc in corpus: await m.ingest_document(doc)
await m.sync_all_documents()

sc = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
r1,r2,rL,cs,rd = [],[],[],[],[]
print(f'Extended eval on {len(bench.tests)} queries...')
t0 = time.time()
for i, t in enumerate(bench.tests):
    gt = ' '.join(cmap[s.file_path][s.span[0]:s.span[1]] for s in t.snippets if s.file_path in cmap).strip()
    if not gt: continue
    resp = await m.query(t.query)
    if not resp.retrieved_snippets:
        r1.append(0);r2.append(0);rL.append(0);cs.append(0);rd.append(0);continue
    preds = [cmap[s.file_path][s.span[0]:min(s.span[1],len(cmap[s.file_path]))]
             for s in resp.retrieved_snippets if s.file_path in cmap]
    pred = ' '.join(preds).strip()
    if not pred: r1.append(0);r2.append(0);rL.append(0);cs.append(0);rd.append(0);continue
    s2 = sc.score(gt, pred)
    r1.append(s2['rouge1'].fmeasure);r2.append(s2['rouge2'].fmeasure);rL.append(s2['rougeL'].fmeasure)
    ge = embedder.encode([gt], normalize_embeddings=True, show_progress_bar=False)
    pe = embedder.encode([pred], normalize_embeddings=True, show_progress_bar=False)
    cs.append(float(cos_sim(ge,pe)[0][0]))
    if len(preds)>1:
        ce = embedder.encode(preds, normalize_embeddings=True, show_progress_bar=False)
        sm = cos_sim(ce); n=len(preds)
        rd.append(float(np.mean([sm[i][j] for i in range(n) for j in range(i+1,n)])))
    else: rd.append(0.0)
    if (i+1)%100==0: print(f'  {i+1}/{len(bench.tests)}')
await m.cleanup()
print(f'\n{"="*65}')
print(f'  EXTENDED METRICS \u2014 TitanRAG v3 on PrivacyQA')
print(f'{"="*65}')
print(f'  ROUGE-1: {np.mean(r1):.4f}  ROUGE-2: {np.mean(r2):.4f}  ROUGE-L: {np.mean(rL):.4f}')
print(f'  Cosine Mean: {np.mean(cs):.4f}  Median: {np.median(cs):.4f}')
print(f'  Redundancy: {np.mean(rd):.4f} (lower=diverse)')
print(f'  Time: {time.time()-t0:.0f}s')

import json, datetime
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
final = {
    'version': 'v3-FT-Kaggle', 'timestamp': ts, 'platform': 'Kaggle T4x2',
    'fine_tuning': {'base_model': 'google/flan-t5-large', 'method': 'QLoRA 4-bit r=16',
                    'tasks': ['proposition_extraction', 'hyde_generation'],
                    'epochs': 3, 'best_val_loss': round(best_val, 4)},
    'techniques': ['fine_tuned_propositions', 'fine_tuned_hyde_3x', 'ensemble_chunking',
                   'bm25_rrf', 'context_headers', 'span_refine', 'cross_encoder_rerank'],
    'models': {'embedder': 'BAAI/bge-m3', 'reranker': 'BAAI/bge-reranker-v2-m3',
               'generator': 'google/flan-t5-large (QLoRA fine-tuned)'},
    'legalbench_results': {k: {kk: round(vv,4) if isinstance(vv,float) else vv
                               for kk,vv in v.items()} for k,v in all_res.items()},
    'extended_metrics': {'rouge1': round(float(np.mean(r1)),4), 'rouge2': round(float(np.mean(r2)),4),
                         'rougeL': round(float(np.mean(rL)),4), 'cosine_mean': round(float(np.mean(cs)),4),
                         'redundancy': round(float(np.mean(rd)),4)},
    'baselines': {'v2': {'p': 0.1722, 'r': 0.2367, 'f1': 0.1993},
                  'naive_p': 0.0786, 'rcts_p': 0.1438},
}
out = f'/teamspace/studios/this_studio/working/results'
with open(out, 'w') as f: json.dump(final, f, indent=2)
print(f'Results saved to {out}')
# Also save adapter as output
import shutil
shutil.make_archive('/teamspace/studios/this_studio/working/results', 'zip', '/teamspace/studios/this_studio/working/results')
print('Adapter zip saved to /teamspace/studios/this_studio')



gc.collect()
for i in range(torch.cuda.device_count()): torch.cuda.empty_cache()
bench, corpus = load_bench('privacy_qa')

# ablation grid
# Each row: (label, kwargs dict)
# Best prior configs + chunker variations + adaptive toggle
ablation_grid = [
    # chunker comparison (fixed config: k4, p=100) ──
    ('sliding   c250 k4 p100',
     dict(chunker='sliding',  top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('semantic  c250 k4 p100',
     dict(chunker='semantic', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('ensemble  c250 k4 p100',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),

    # pool / k sweep (ensemble chunker)
    ('ensemble  c250 k3 p80',
     dict(chunker='ensemble', top_k=3, pool=80,  use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('ensemble  c250 k5 p100',
     dict(chunker='ensemble', top_k=5, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('ensemble  c250 k5 p150',
     dict(chunker='ensemble', top_k=5, pool=150, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),

    # feature toggles (ensemble k4 p=100) ──
    ('- no HyDE',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=False,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('- no BM25',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=False, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),
    ('- no ctx headers',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=False, use_xref_graph=True, use_adaptive_policy=False)),
    ('- no xref graph',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=False, use_adaptive_policy=False)),
    ('+ span refine ON',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=True,  use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),

    # adaptive policy (best chunker configs)
    ('ADAPTIVE sliding',
     dict(chunker='sliding',  top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=True)),
    ('ADAPTIVE ensemble',
     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,
          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=True)),
]

print(f"{'#':<3} {'Config':<28} {'P':>7} {'R':>7} {'F1':>7} {'Chunks':>7} {'t':>5}")
print('\u2500' * 68)

async def grid_ablation():
    results = []
    best_f1, best_label = 0, ''
    for idx, (label, kw) in enumerate(ablation_grid, 1):
        gc.collect()
        for g in range(torch.cuda.device_count()): torch.cuda.empty_cache()
        m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde, **kw)
        t0 = time.time()
        r = await run_benchmark(bench.tests, corpus, m)
        p, rc = r.avg_precision, r.avg_recall
        f1 = 2*p*rc/(p+rc) if p+rc > 0 else 0
        nc = len(m._chunks)
        el = time.time() - t0
        tag = ''
        if f1 > best_f1: best_f1 = f1; best_label = label; tag = ' \u2b50'
        print(f"{idx:<3} {label:<28} {p:>7.2%} {rc:>7.2%} {f1:>7.2%} {nc:>7} {el:>4.0f}s{tag}")
        results.append({'label': label, 'p': round(p,4), 'r': round(rc,4),
                        'f1': round(f1,4), 'chunks': nc})
        # Print routes for adaptive runs
        if hasattr(m, '_policy') and m._policy:
            m._policy.print_route_summary()
        await m.cleanup()

    # summary
    print(f'\n{"="*68}')
    print(f'  BEST CONFIG: {best_label} (F1={best_f1:.2%})')
    print(f'  Prior: v2++=18.51% | v2=19.93% | RCTS P=14.38%')
    print(f'{"="*68}')

    # leaderboard
    results.sort(key=lambda x: x['f1'], reverse=True)
    print(f'\n{"Rank":<5} {"Config":<28} {"P":>7} {"R":>7} {"F1":>7}')
    print('\u2500' * 55)
    medals = ['\U0001f947','\U0001f948','\U0001f949']
    for i, r in enumerate(results):
        m = medals[i] if i < 3 else f'{i+1}.'
        print(f"{m:<5} {r['label']:<28} {r['p']:>7.2%} {r['r']:>7.2%} {r['f1']:>7.2%}")

await grid_ablation()