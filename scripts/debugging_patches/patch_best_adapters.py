import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

"""
Patch script to update the Lightning AI path to correctly include the /best_adapters folder.
Updates titanragv1_rarity.ipynb
"""
import json
import re

nb_path = str(BASE_DIR / 'titanragv1_rarity.ipynb')

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The updated path from the user
OLD_PATH = str(BASE_DIR)
NEW_PATH = str(BASE_DIR / 'best_adapters')

updated_cells = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        s = "".join(c['source']) if isinstance(c['source'], list) else c['source']
        original_s = s
        
        # Replace the precise old path with the precise new path if it has quotations around it
        s = s.replace(f'"{OLD_PATH}"', f'"{NEW_PATH}"')
        s = s.replace(f"'{OLD_PATH}'", f"'{NEW_PATH}'")
        
        # Also catch occurrences where they had a trailing slash
        s = s.replace(f'"{OLD_PATH}/"', f'"{NEW_PATH}"')
        s = s.replace(f"'{OLD_PATH}/'", f"'{NEW_PATH}'")
        
        if s != original_s:
            c['source'] = [s]
            updated_cells += 1
            print(f"Patched Cell {i} to use /best_adapters")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=None, ensure_ascii=False)

print(f"\nSuccessfully updated {updated_cells} cells to point to {NEW_PATH}")
