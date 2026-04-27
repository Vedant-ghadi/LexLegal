import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

def enrich_logs(log_path, bench_name='privacy_qa'):
    """Attach per-query P/R/F1 to JSONL routing logs using ground truth spans."""
    gt_path = DATA_DIR / 'benchmarks' / f'{bench_name}.json'
    if not gt_path.exists() or not os.path.exists(log_path):
        print('Log or ground truth not found.'); return

    bench_data = json.loads(gt_path.read_text(encoding='utf-8', errors='replace'))
    gt_map = {}
    for t in bench_data['tests']:
        gt_map[t['query']] = [(s['file_path'], s['span'][0], s['span'][1]) for s in t['snippets']]

    with open(log_path, 'r', encoding='utf-8') as f:
        logs = [json.loads(line) for line in f if line.strip()]

    enriched = 0
    for entry in logs:
        q = entry.get('query', '')
        if q not in gt_map: continue
        gt_spans = gt_map[q]

        retrieved = entry.get('retrieved_spans', [])
        if not retrieved: continue

        # character-level overlap P/R/F1
        gt_chars = set()
        for fp, s, e in gt_spans:
            for c in range(s, e):
                gt_chars.add((fp, c))

        ret_chars = set()
        for sp in retrieved:
            fp = sp['file_path']
            s = sp['char_offset']
            e = sp.get('char_end', s + 200)
            for c in range(s, e):
                ret_chars.add((fp, c))

        overlap = len(gt_chars & ret_chars)
        p = overlap / max(len(ret_chars), 1)
        r = overlap / max(len(gt_chars), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        entry['char_precision'] = round(p, 4)
        entry['char_recall'] = round(r, 4)
        entry['char_f1'] = round(f1, 4)
        enriched += 1

    out_path = log_path.replace('.jsonl', '_enriched.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in logs:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f'Enriched {enriched}/{len(logs)} entries. Saved to {out_path}')
    return out_path
