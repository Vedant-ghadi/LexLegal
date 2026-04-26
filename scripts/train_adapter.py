import os
import json
import random
from pathlib import Path
import time
import gc
import math
import torch
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'
ADAPTER_DIR = BASE_DIR / 'best_adapters'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
)

print('Loading Flan-T5-Large (4-bit) on GPU 0...')
tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large', legacy=False)
base_model = T5ForConditionalGeneration.from_pretrained(
    'google/flan-t5-large', quantization_config=bnb_config,
    device_map={'': 0}
)
base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32,
    lora_dropout=0.05, target_modules=['q', 'v'], bias='none',
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

import json, random, pathlib
from nltk.tokenize import sent_tokenize


DATASETS = ['privacy_qa', 'cuad', 'maud', 'contract_nli']
train_data = []

for ds in DATASETS:
    path = DATA_DIR / 'benchmarks' / f'{ds}.json'
    if not path.exists(): continue
    bench = json.loads(path.read_text(encoding='utf-8', errors='replace'))
    for test in bench['tests']:
        query = test['query']
        for snippet in test['snippets']:
            fp = snippet['file_path']
            span = snippet['span']
            doc_path = DATA_DIR / 'corpus' / fp
            if not doc_path.exists(): continue
            content = doc_path.read_text(encoding='utf-8', errors='replace')
            gt = content[span[0]:span[1]].strip()
            if len(gt) < 20 or len(gt) > 800: continue  # skip extremes
            ctx = content[max(0,span[0]-100):min(len(content),span[1]+100)].strip()
            gt_sents = sent_tokenize(gt)
            if gt_sents:
                prop_out = ' '.join(f'{i+1}. {s}' for i, s in enumerate(gt_sents))
                train_data.append({'input': f'Extract facts: {ctx[:600]}',
                                   'output': prop_out[:400], 'task': 'proposition'})
            # Only 1 HyDE variant (not 3)
            train_data.append({'input': f'Write a legal passage answering: {query[:200]}',
                               'output': gt[:400], 'task': 'hyde'})

random.seed(42)
random.shuffle(train_data)

# ── CAP at 2000 examples total ──
MAX_EXAMPLES = 2000
train_data = train_data[:MAX_EXAMPLES]

val_size = min(200, len(train_data) // 10)
val_data, train_data = train_data[:val_size], train_data[val_size:]
pc = sum(1 for d in train_data if d['task']=='proposition')
hc = sum(1 for d in train_data if d['task']=='hyde')
print(f'Train: {len(train_data):,} (prop={pc:,}, hyde={hc:,})  |  Val: {len(val_data):,}')

class LegalDS(Dataset):
    def __init__(self, data, tok, mi=384, mo=128):  # shorter sequences
        self.data, self.tok, self.mi, self.mo = data, tok, mi, mo
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        d = self.data[i]
        inp = self.tok(d['input'], max_length=self.mi, truncation=True, padding='max_length', return_tensors='pt')
        out = self.tok(d['output'], max_length=self.mo, truncation=True, padding='max_length', return_tensors='pt')
        labels = out.input_ids.squeeze()
        labels[labels == self.tok.pad_token_id] = -100
        return {'input_ids': inp.input_ids.squeeze(), 'attention_mask': inp.attention_mask.squeeze(), 'labels': labels}

BS = 8         # bigger batch (was 4)
GA = 2         # less accumulation (was 4), effective=16
EPOCHS = 2     # fewer epochs (was 3)
LR = 2e-4      # lower LR (was 3e-4) — prevents NaN

train_loader = DataLoader(LegalDS(train_data, tokenizer), batch_size=BS, shuffle=True, drop_last=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(LegalDS(val_data, tokenizer), batch_size=BS, num_workers=2, pin_memory=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GA) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, total_steps // 5, total_steps)  # 20% warmup

print(f'Train: {len(train_data):,} examples | {EPOCHS} epochs | eff_batch={BS*GA}')
print(f'Steps/epoch: {len(train_loader)//GA} | Total: {total_steps}')
print(f'Estimated time: ~{total_steps * 4 / 60:.0f} min')
print('='*60)

model.train()
best_val = float('inf')

for epoch in range(EPOCHS):
    t0 = time.time()
    el = 0
    nan_count = 0
    opt.zero_grad()
    
    for step, batch in enumerate(train_loader):
        batch = {k: v.to('cuda:0') for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GA
        
        # ── NaN PROTECTION ──
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            opt.zero_grad()  # skip this batch
            if nan_count > 20:
                print(f'  WARNING: {nan_count} NaN losses — stopping early')
                break
            continue
        
        loss.backward()
        el += loss.item() * GA
        
        if (step + 1) % GA == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # tighter clipping (was 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        
        if (step + 1) % 50 == 0:
            avg = el / (step + 1 - nan_count) if (step + 1 - nan_count) > 0 else 0
            print(f'  E{epoch+1} step {step+1}/{len(train_loader)} loss={avg:.4f} (NaN skipped: {nan_count})')
    
    avg_t = el / max(len(train_loader) - nan_count, 1)
    
    # Validation
    model.eval()
    vl = 0
    v_count = 0
    with torch.no_grad():
        for b in val_loader:
            b = {k: v.to('cuda:0') for k, v in b.items()}
            v_loss = model(**b).loss
            if not torch.isnan(v_loss):
                vl += v_loss.item()
                v_count += 1
    avg_v = vl / max(v_count, 1)
    model.train()
    
    tag = ' ⭐ BEST' if avg_v < best_val else ''
    if avg_v < best_val:
        best_val = avg_v
        model.save_pretrained(str(ADAPTER_DIR))
    
    elapsed = time.time() - t0
    print(f'Epoch {epoch+1}/{EPOCHS} | Train={avg_t:.4f} | Val={avg_v:.4f} | {elapsed:.0f}s | NaN={nan_count}{tag}')
    gc.collect(); torch.cuda.empty_cache()

print(f'\nDone! Best val loss: {best_val:.4f}')
print(f'Total time: {sum(1 for _ in range(1))} epochs completed')


model.eval()
def ft_gen(prompt, ml=256):
    ids = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True).input_ids.to('cuda:0')
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=ml, num_beams=3, early_stopping=True)
    return tokenizer.decode(out[0], skip_special_tokens=True)

test_text = """The Company shall not disclose Personal Data to third parties except as 
required by applicable law or with the Data Subject's prior consent. Any disclosure 
shall be limited to the minimum data necessary."""

print('PROPOSITION EXTRACTION:')
print(ft_gen(f'Break this legal text into independent, self-contained facts:\n\n{test_text}'))

print('\nHyDE GENERATION:')
for q in ['Does Uber share data with law enforcement?', 'Can I opt out of data collection?']:
    print(f'  Q: {q}')
    print(f'  A: {ft_gen(f"Write a legal passage answering: {q}")[:200]}')
    print()


