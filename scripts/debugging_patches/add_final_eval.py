import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

"""
Add final eval cell to titanragv1_rarity.ipynb.
Uses the BEST config from ablation: sliding c250 k4 p100 (F1=18.01%)
Runs across ALL LegalBench-RAG datasets.
"""
import json

nb_path = str(BASE_DIR / 'titanragv1_rarity.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the ablation cell and add the final eval right after it
final_eval_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## \U0001f3c6 Cell 14 — Final Evaluation on All Datasets\n",
               "\n",
               "**Best config from ablation:** `sliding c250 k4 p100` (F1=18.01% on PrivacyQA)\n",
               "\n",
               "Running this config + the runner-up across all 4 LegalBench-RAG datasets."]
}

final_eval_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": ["""# ── Final Evaluation: Best Config on ALL Datasets ──
import gc, time
from tqdm import tqdm

DATASETS = ['privacy_qa', 'cuad', 'maud', 'contractnli']

# Best config from ablation: sliding c250 k4 p100 (F1=18.01%)
BEST_CONFIG = dict(
    top_k=4, pool=100, chunker='sliding',
    use_bm25=True, use_hyde=True,
    use_span_refine=False, use_ctx=True,
    use_xref_graph=True,
    use_adaptive_policy=False
)

# Runner-up: ensemble c250 k3 p80 (F1=17.58%, highest recall)
RUNNER_UP = dict(
    top_k=3, pool=80, chunker='ensemble',
    use_bm25=True, use_hyde=True,
    use_span_refine=False, use_ctx=True,
    use_xref_graph=True,
    use_adaptive_policy=False
)

configs = [
    ('BEST: sliding k4 p100', BEST_CONFIG),
    ('RUNNER-UP: ensemble k3 p80', RUNNER_UP),
]

print(f"{'Dataset':<15} {'Config':<30} {'P':>7} {'R':>7} {'F1':>7} {'t':>5}")
print('\\u2500' * 75)

all_results = []

for ds_name in DATASETS:
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    try:
        bench, corpus = load_bench(ds_name)
    except Exception as e:
        print(f"{ds_name:<15} SKIPPED: {e}")
        continue
    
    for label, cfg in configs:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde, **cfg)
        t0 = time.time()
        r = await run_benchmark(bench.tests, corpus, m)
        elapsed = time.time() - t0
        
        p, rc = r.avg_precision, r.avg_recall
        f1 = 2*p*rc/(p+rc) if p+rc > 0 else 0
        
        print(f"{ds_name:<15} {label:<30} {p:>7.2%} {rc:>7.2%} {f1:>7.2%} {elapsed:>4.0f}s")
        all_results.append({'dataset': ds_name, 'config': label, 'P': p, 'R': rc, 'F1': f1, 'time': elapsed})
        
        await m.cleanup()

# ── Summary Table ──
print('\\n' + '=' * 75)
print('FINAL RESULTS SUMMARY')
print('=' * 75)
print(f"{'Dataset':<15} {'Config':<30} {'P':>7} {'R':>7} {'F1':>7}")
print('-' * 75)
for r in all_results:
    tag = ' \\U0001f3c6' if r['F1'] == max(x['F1'] for x in all_results if x['dataset'] == r['dataset']) else ''
    print(f"{r['dataset']:<15} {r['config']:<30} {r['P']:>7.2%} {r['R']:>7.2%} {r['F1']:>7.2%}{tag}")

# Grand average
best_rows = [r for r in all_results if 'BEST' in r['config']]
if best_rows:
    avg_p = sum(r['P'] for r in best_rows) / len(best_rows)
    avg_r = sum(r['R'] for r in best_rows) / len(best_rows)
    avg_f1 = sum(r['F1'] for r in best_rows) / len(best_rows)
    total_t = sum(r['time'] for r in best_rows)
    print(f"\\n{'GRAND AVG':<15} {'BEST config':<30} {avg_p:>7.2%} {avg_r:>7.2%} {avg_f1:>7.2%}  ({total_t:.0f}s total)")
"""]
}

# Insert at the end (before any existing final cells)
nb['cells'].append(final_eval_md)
nb['cells'].append(final_eval_code)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print("Done! Added final eval cell with:")
print("  - BEST: sliding c250 k4 p100 (F1=18.01%)")
print("  - RUNNER-UP: ensemble c250 k3 p80 (F1=17.58%)")
print("  - Runs across: privacy_qa, cuad, maud, contractnli")
