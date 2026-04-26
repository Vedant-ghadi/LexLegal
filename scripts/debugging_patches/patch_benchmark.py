import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

"""
Patch titanragv1_rarity.ipynb to override run_benchmark directly in the notebook.
Replaces asyncio.gather (fires 194 queries simultaneously) with sequential loop + tqdm.
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

# Find the cell that imports run_benchmark
import_cell_idx = -1
for i, c in enumerate(cells):
    s = get_src(c)
    if 'from legalbenchrag' in s and 'run_benchmark' in s:
        import_cell_idx = i
        break

print(f"Found import cell: {import_cell_idx}")

# Insert a NEW cell right AFTER the import cell that overrides run_benchmark
override_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
'''# ── Override run_benchmark for GPU efficiency ──
# The default run_benchmark uses asyncio.gather() which fires ALL queries
# simultaneously as coroutines. Since our query() uses synchronous GPU ops,
# this causes massive scheduling overhead. Sequential is 3-5x faster.
from tqdm import tqdm as _tqdm

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
    from legalbenchrag.benchmark_types import QAResult, BenchmarkResult
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
'''
    ]
}

# Check if we already inserted this override
already_exists = False
for c in cells:
    if 'GPU-optimized benchmark' in get_src(c):
        already_exists = True
        break

if not already_exists and import_cell_idx >= 0:
    cells.insert(import_cell_idx + 1, override_cell)
    print(f"Inserted override cell at position {import_cell_idx + 1}")
elif already_exists:
    # Update it
    for i, c in enumerate(cells):
        if 'GPU-optimized benchmark' in get_src(c):
            cells[i] = override_cell
            print(f"Updated existing override cell at position {i}")
            break
else:
    print("Could not find import cell, inserting at position 30")
    cells.insert(30, override_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print(f"\nDone! {len(cells)} cells total.")
print("Changes made:")
print("  1. Added run_benchmark override cell (sequential + tqdm)")
print("  2. Queries now process ONE AT A TIME (no asyncio.gather overhead)")
print("  3. Live progress bar: 'Queries: 45/194 [00:32, 2.18q/s]'")
