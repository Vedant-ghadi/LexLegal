import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from .config import embedder, ft_gen

# sliding window
def sliding_chunk(content, fp, cs=300, st=75):
    chunks, pos = [], 0
    while pos < len(content):
        end = min(pos+cs, len(content))
        if end < len(content):
            nb = content.find('.', end-20, end+50)
            if nb > 0: end = nb+1
        t = content[pos:end].strip()
        if len(t) >= 30: chunks.append({'text': t, 'file_path': fp, 'char_offset': pos})
        pos += st
    return chunks

# semantic chunker
def semantic_chunk(content, fp, thr=0.45, mx=5, mn=2):
    ss = sent_tokenize(content)
    if len(ss) <= 1:
        return [{'text': content.strip(), 'file_path': fp, 'char_offset': 0}] if content.strip() else []
    em = embedder.encode(ss, normalize_embeddings=True, show_progress_bar=False, batch_size=256)
    gs, cur = [], [0]
    for i in range(1, len(ss)):
        si = float(cosine_similarity([em[i-1]], [em[i]])[0][0])
        if (si < thr and len(cur) >= mn) or len(cur) >= mx: gs.append(cur); cur = [i]
        else: cur.append(i)
    if cur: gs.append(cur)
    ch = []
    for g in gs:
        t = ' '.join(ss[i] for i in g).strip()
        if len(t) >= 20:
            off = content.find(ss[g[0]][:40])
            ch.append({'text': t, 'file_path': fp, 'char_offset': max(0, off)})
    return ch

# proposition chunker (fine-tuned t5)
def proposition_chunk(content, fp, window=600, stride=300):
    chunks, pos = [], 0
    while pos < len(content):
        end = min(pos+window, len(content))
        seg = content[pos:end].strip()
        if len(seg) < 30: pos += stride; continue
        result = ft_gen(f'Break this legal text into independent, self-contained facts:\n\n{seg}')
        props = [p.strip().lstrip('0123456789.-) ') for p in result.replace('. ', '.\n').split('\n') if p.strip()]
        if not props: props = sent_tokenize(seg)
        for p in props:
            p = p.strip()
            if len(p) >= 15: chunks.append({'text': p, 'file_path': fp, 'char_offset': pos})
        pos += stride
    return chunks

# ensemble (semantic + sliding merged)
def ensemble_chunk(content, fp, cs=300, st=75):
    all_c = semantic_chunk(content, fp) + sliding_chunk(content, fp, cs, st)
    kept = []
    for c in all_c:
        c_s, c_e = c['char_offset'], c['char_offset'] + len(c['text'])
        dup = any(c['file_path'] == k['file_path'] and
                  max(0, min(c_e, k['char_offset']+len(k['text'])) - max(c_s, k['char_offset'])) > 0.7*len(c['text'])
                  for k in kept)
        if not dup: kept.append(c)
    return kept