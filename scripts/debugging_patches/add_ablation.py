import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

"""
Add comprehensive chunker × config ablation cell to titanragv1.ipynb.
Tests: 4 chunkers × multiple pool/k/hyde/adaptive combos.
"""
import json

with open(str(BASE_DIR / 'titanragv1.ipynb'), 'r', encoding='utf-8') as f:
    nb = json.load(f)
cells = nb['cells']

def src(c):
    return ''.join(c['source']) if isinstance(c['source'], list) else c['source']

# Find Cell 13 (Ablation Study) — replace it with comprehensive grid
abl_md_idx = -1
abl_code_idx = -1
for i, c in enumerate(cells):
    s = src(c)
    if c['cell_type'] == 'markdown' and 'Ablation' in s:
        abl_md_idx = i
    if c['cell_type'] == 'code' and ("abl = [" in s or "('ALL ON'" in s):
        abl_code_idx = i

print(f"Ablation markdown: {abl_md_idx}, code: {abl_code_idx}")

# If not found, insert after eval cell
if abl_md_idx < 0:
    for i, c in enumerate(cells):
        s = src(c)
        if 'BEST STATIC' in s or 'ADAPTIVE POLICY' in s:
            abl_md_idx = i + 1
            break

new_md = {
    "cell_type": "markdown", "metadata": {},
    "source": [
        "## \U0001f9ea Cell 13 \u2014 Full Ablation Grid: Chunkers \u00d7 Configs\n",
        "\n",
        "**Systematic grid search** across:\n",
        "- **4 Chunkers**: sliding, semantic, proposition, ensemble\n",
        "- **Multiple configs**: pool, top_k, HyDE, BM25, adaptive policy\n",
        "- **Adaptive vs Static**: does dynamic routing beat fixed params?\n",
        "\n",
        "All runs use `span_refine=False` and `ctx_headers=True` (proven best from prior ablations)."
    ]
}

new_code = {
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
"gc.collect()\n",
"for i in range(torch.cuda.device_count()): torch.cuda.empty_cache()\n",
"bench, corpus = load_bench('privacy_qa')\n",
"\n",
"# ── Ablation Grid ──\n",
"# Each row: (label, kwargs dict)\n",
"# Best prior configs + chunker variations + adaptive toggle\n",
"ablation_grid = [\n",
"    # ── CHUNKER COMPARISON (fixed config: k=4, p=100) ──\n",
"    ('sliding   c250 k4 p100',\n",
"     dict(chunker='sliding',  top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('semantic  c250 k4 p100',\n",
"     dict(chunker='semantic', top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('ensemble  c250 k4 p100',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"\n",
"    # ── POOL / K SWEEP (ensemble chunker) ──\n",
"    ('ensemble  c250 k3 p80',\n",
"     dict(chunker='ensemble', top_k=3, pool=80,  use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('ensemble  c250 k5 p100',\n",
"     dict(chunker='ensemble', top_k=5, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('ensemble  c250 k5 p150',\n",
"     dict(chunker='ensemble', top_k=5, pool=150, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"\n",
"    # ── FEATURE TOGGLES (ensemble k=4 p=100) ──\n",
"    ('- no HyDE',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=False,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('- no BM25',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=False, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('- no ctx headers',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=False, use_xref_graph=True, use_adaptive_policy=False)),\n",
"    ('- no xref graph',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=False, use_adaptive_policy=False)),\n",
"    ('+ span refine ON',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=True,  use_ctx=True, use_xref_graph=True, use_adaptive_policy=False)),\n",
"\n",
"    # ── ADAPTIVE POLICY (best chunker configs) ──\n",
"    ('ADAPTIVE sliding',\n",
"     dict(chunker='sliding',  top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=True)),\n",
"    ('ADAPTIVE ensemble',\n",
"     dict(chunker='ensemble', top_k=4, pool=100, use_bm25=True, use_hyde=True,\n",
"          use_span_refine=False, use_ctx=True, use_xref_graph=True, use_adaptive_policy=True)),\n",
"]\n",
"\n",
"print(f\"{'#':<3} {'Config':<28} {'P':>7} {'R':>7} {'F1':>7} {'Chunks':>7} {'t':>5}\")\n",
"print('\\u2500' * 68)\n",
"\n",
"async def grid_ablation():\n",
"    results = []\n",
"    best_f1, best_label = 0, ''\n",
"    for idx, (label, kw) in enumerate(ablation_grid, 1):\n",
"        gc.collect()\n",
"        for g in range(torch.cuda.device_count()): torch.cuda.empty_cache()\n",
"        m = TitanRAGv3(embedder, rr=bge_reranker, hyde_exp=hyde, **kw)\n",
"        t0 = time.time()\n",
"        r = await run_benchmark(bench.tests, corpus, m)\n",
"        p, rc = r.avg_precision, r.avg_recall\n",
"        f1 = 2*p*rc/(p+rc) if p+rc > 0 else 0\n",
"        nc = len(m._chunks)\n",
"        el = time.time() - t0\n",
"        tag = ''\n",
"        if f1 > best_f1: best_f1 = f1; best_label = label; tag = ' \\u2b50'\n",
"        print(f\"{idx:<3} {label:<28} {p:>7.2%} {rc:>7.2%} {f1:>7.2%} {nc:>7} {el:>4.0f}s{tag}\")\n",
"        results.append({'label': label, 'p': round(p,4), 'r': round(rc,4),\n",
"                        'f1': round(f1,4), 'chunks': nc})\n",
"        # Print routes for adaptive runs\n",
"        if hasattr(m, '_policy') and m._policy:\n",
"            m._policy.print_route_summary()\n",
"        await m.cleanup()\n",
"\n",
"    # ── Summary ──\n",
"    print(f'\\n{\"=\"*68}')\n",
"    print(f'  BEST CONFIG: {best_label} (F1={best_f1:.2%})')\n",
"    print(f'  Prior: v2++=18.51% | v2=19.93% | RCTS P=14.38%')\n",
"    print(f'{\"=\"*68}')\n",
"\n",
"    # ── Leaderboard ──\n",
"    results.sort(key=lambda x: x['f1'], reverse=True)\n",
"    print(f'\\n{\"Rank\":<5} {\"Config\":<28} {\"P\":>7} {\"R\":>7} {\"F1\":>7}')\n",
"    print('\\u2500' * 55)\n",
"    medals = ['\\U0001f947','\\U0001f948','\\U0001f949']\n",
"    for i, r in enumerate(results):\n",
"        m = medals[i] if i < 3 else f'{i+1}.'\n",
"        print(f\"{m:<5} {r['label']:<28} {r['p']:>7.2%} {r['r']:>7.2%} {r['f1']:>7.2%}\")\n",
"\n",
"await grid_ablation()\n",
    ]
}

# Replace or insert
if abl_md_idx >= 0 and abl_code_idx >= 0:
    cells[abl_md_idx] = new_md
    cells[abl_code_idx] = new_code
    print(f"  Replaced ablation cells at {abl_md_idx}, {abl_code_idx}")
elif abl_md_idx >= 0:
    cells[abl_md_idx] = new_md
    cells.insert(abl_md_idx + 1, new_code)
    print(f"  Replaced md at {abl_md_idx}, inserted code at {abl_md_idx + 1}")
else:
    # Insert at end before save results
    save_idx = len(cells) - 2
    cells.insert(save_idx, new_md)
    cells.insert(save_idx + 1, new_code)
    print(f"  Inserted ablation cells at {save_idx}")

with open(str(BASE_DIR / 'titanragv1.ipynb'), 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print(f"\nDone! {len(cells)} cells. Grid: 13 configs total")
print("  3 chunker comparisons (sliding/semantic/ensemble)")
print("  3 pool/k sweeps (k3p80, k5p100, k5p150)")
print("  5 feature toggles (-HyDE, -BM25, -ctx, -xref, +span_refine)")
print("  2 adaptive runs (sliding, ensemble)")
