import os
import re
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'legalbench-rag' / 'data'

DROPBOX_URL = (
    "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/"
    "AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&dl=1"
)

def safe_name(n):
    return re.sub(r'[|*?<>]', '_', n)

def download_legalbench():
    """Download and extract the LegalBench-RAG dataset if not already present."""
    if (DATA_DIR / 'benchmarks' / 'cuad.json').exists():
        print('LegalBench-RAG data already exists, skipping download.')
        return

    print('Downloading LegalBench-RAG (~90MB)...')

    zip_path = BASE_DIR / 'legalbench_rag.zip'

    # try wget first, fall back to urllib
    try:
        import subprocess
        subprocess.run(['wget', '-q', '-O', str(zip_path), DROPBOX_URL], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        import urllib.request
        urllib.request.urlretrieve(DROPBOX_URL, str(zip_path))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as z:
        for name in z.namelist():
            if name == '/': continue
            t = DATA_DIR / safe_name(name)
            if name.endswith('/'):
                t.mkdir(parents=True, exist_ok=True)
            else:
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(z.read(name))

    os.remove(str(zip_path))
    print('LegalBench-RAG extracted successfully.')

if __name__ == '__main__':
    download_legalbench()
