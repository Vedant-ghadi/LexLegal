import time
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

def run_extended_eval(engine, bench, corpus, embedder):
    """Run ROUGE, cosine similarity, and redundancy metrics on a benchmark."""
    cmap = {d.file_path: d.content for d in corpus}

    sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1, r2, rL, cs, rd = [], [], [], [], []

    print(f'Extended eval on {len(bench.tests)} queries...')
    t0 = time.time()

    for test in bench.tests:
        # ground truth text
        gt_parts = []
        for snippet in test.snippets:
            fp = snippet.file_path
            if fp in cmap:
                gt_parts.append(cmap[fp][snippet.span[0]:snippet.span[1]])
        gt_text = ' '.join(gt_parts).strip()
        if not gt_text:
            continue

        # retrieved text
        import asyncio
        resp = asyncio.get_event_loop().run_until_complete(engine.query(test.query))
        ret_parts = []
        for s in resp.retrieved_snippets:
            if s.file_path in cmap:
                ret_parts.append(cmap[s.file_path][s.span[0]:s.span[1]])
        ret_text = ' '.join(ret_parts).strip()
        if not ret_text:
            continue

        # rouge scores
        scores = sc.score(gt_text, ret_text)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rL.append(scores['rougeL'].fmeasure)

        # cosine similarity between embeddings
        gt_vec = embedder.encode([gt_text], normalize_embeddings=True)
        ret_vec = embedder.encode([ret_text], normalize_embeddings=True)
        cs.append(float(cos_sim(gt_vec, ret_vec)[0][0]))

        # redundancy: pairwise similarity among retrieved snippets
        if len(ret_parts) > 1:
            vecs = embedder.encode(ret_parts, normalize_embeddings=True)
            sim_matrix = cos_sim(vecs, vecs)
            n = len(ret_parts)
            pairwise = [sim_matrix[i][j] for i in range(n) for j in range(i+1, n)]
            rd.append(float(np.mean(pairwise)))
        else:
            rd.append(0.0)

    elapsed = time.time() - t0

    results = {
        'rouge1': round(float(np.mean(r1)), 4) if r1 else 0.0,
        'rouge2': round(float(np.mean(r2)), 4) if r2 else 0.0,
        'rougeL': round(float(np.mean(rL)), 4) if rL else 0.0,
        'cosine_mean': round(float(np.mean(cs)), 4) if cs else 0.0,
        'redundancy': round(float(np.mean(rd)), 4) if rd else 0.0,
        'n_queries': len(r1),
        'elapsed_s': round(elapsed, 1),
    }

    print(f'ROUGE-1={results["rouge1"]:.4f}  ROUGE-2={results["rouge2"]:.4f}  '
          f'ROUGE-L={results["rougeL"]:.4f}  Cosine={results["cosine_mean"]:.4f}  '
          f'Redundancy={results["redundancy"]:.4f}  ({elapsed:.1f}s)')

    return results
