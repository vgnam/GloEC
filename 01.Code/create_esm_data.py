"""
Generate ESM-2 (650M) embeddings via NVIDIA NIM API.
API returns npz binary -> extract mean embeddings -> save as .pt

Usage:
    $env:NVIDIA_API_KEY = "nvapi-xxxxx"
    python create_esm_data.py
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import torch
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from zipfile import ZipFile

os.environ['NVIDIA_API_KEY'] = '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'
# ---------- Settings ----------
DATA_DIR = '../02.Datasets/uniport_2022_5/'
API_URL = 'https://health.api.nvidia.com/v1/biology/meta/esm2-650m'
MAX_SEQ_LEN = 1024       # NIM API limit
BATCH_SIZE = 32           # NIM API max 32 sequences per request
EMB_FORMAT = 'npz'
MAX_RETRIES = 5
RETRY_DELAY = 5

FILES = [
    ('f_train.csv',     'esm_f_train.pt'),
    ('f_eval.csv',      'esm_f_eval.pt'),
    ('f_time_test.csv', 'esm_f_time_test.pt'),
]


def get_api_key():
    key = os.environ.get('NVIDIA_API_KEY', '')
    if not key:
        print('ERROR: Set NVIDIA_API_KEY environment variable first.')
        print('  $env:NVIDIA_API_KEY = "nvapi-xxxxx"')
        sys.exit(1)
    return key


def parse_npz_response(content, content_type):
    """Parse npz/zip response from NIM API, extract mean embeddings."""
    embeddings = []

    if content_type == 'application/zip':
        # Multiple sequences -> zip of npz files
        with ZipFile(BytesIO(content)) as zf:
            for name in sorted(zf.namelist()):
                with zf.open(name) as f:
                    data = np.load(BytesIO(f.read()))
                    # 'mean' key contains the mean-pooled embedding (1280,)
                    if 'mean' in data:
                        embeddings.append(data['mean'])
                    elif 'representations' in data:
                        # Per-token representations -> mean pool manually
                        reps = data['representations']
                        embeddings.append(reps.mean(axis=0))
                    else:
                        # Use first available array
                        key = list(data.keys())[0]
                        arr = data[key]
                        if arr.ndim == 2:
                            embeddings.append(arr.mean(axis=0))
                        else:
                            embeddings.append(arr)
    else:
        # Single sequence -> single npz
        data = np.load(BytesIO(content))
        if 'mean' in data:
            embeddings.append(data['mean'])
        elif 'representations' in data:
            reps = data['representations']
            embeddings.append(reps.mean(axis=0))
        else:
            key = list(data.keys())[0]
            arr = data[key]
            if arr.ndim == 2:
                embeddings.append(arr.mean(axis=0))
            else:
                embeddings.append(arr)

    return embeddings


def call_nim_api(sequences, api_key):
    """Call NVIDIA NIM ESM2-650M API, returns list of numpy embeddings."""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    payload = {
        'sequences': sequences,
        'format': EMB_FORMAT,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if resp.status_code == 200:
                content_type = resp.headers.get('Content-Type', '')
                embeddings = parse_npz_response(resp.content, content_type)
                return embeddings

            elif resp.status_code == 429:
                wait = RETRY_DELAY * (attempt + 1)
                print(f'  Rate limited, waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'  API error {resp.status_code}: {resp.text[:300]}')
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f'  Request error: {e}')
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(f'Failed after {MAX_RETRIES} retries')


def generate_embeddings(csv_path, output_path, api_key):
    if os.path.exists(output_path):
        print(f'[SKIP] {output_path} already exists')
        return

    print(f'\n=== Processing {csv_path} -> {output_path} ===')
    df = pd.read_csv(csv_path)
    sequences = df.iloc[:, 1].tolist()
    total = len(sequences)
    print(f'Total sequences: {total}')

    # Truncate sequences to API limit
    seqs = []
    for s in sequences:
        s = str(s).strip()
        if len(s) > MAX_SEQ_LEN:
            s = s[:MAX_SEQ_LEN]
        seqs.append(s)

    all_embeddings = []

    for start in tqdm(range(0, total, BATCH_SIZE), desc='API calls'):
        end = min(start + BATCH_SIZE, total)
        batch_seqs = seqs[start:end]

        embeddings = call_nim_api(batch_seqs, api_key)

        for emb in embeddings:
            all_embeddings.append(torch.tensor(emb, dtype=torch.float32))

        # Small delay to be nice to the API
        time.sleep(0.05)

    esm_tensor = torch.stack(all_embeddings, dim=0)
    print(f'Output tensor shape: {esm_tensor.shape}  (expected: [{total}, 1280])')
    torch.save(esm_tensor, output_path)
    print(f'Saved to {output_path}')


def main():
    api_key = get_api_key()
    print(f'API Key: {api_key[:16]}...')

    # Quick test with 1 sequence
    print('Testing API with 1 sequence...')
    test_emb = call_nim_api(['MKTVRQERLKSIVRILERSKEPVSGAQL'], api_key)
    print(f'Test OK! Embedding dim: {len(test_emb[0])}')

    for csv_name, pt_name in FILES:
        csv_path = os.path.join(DATA_DIR, csv_name)
        output_path = os.path.join(DATA_DIR, pt_name)
        if not os.path.exists(csv_path):
            print(f'[WARN] {csv_path} not found, skipping')
            continue
        generate_embeddings(csv_path, output_path, api_key)

    print('\nDone!')


if __name__ == '__main__':
    main()
