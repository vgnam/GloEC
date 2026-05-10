"""
Generate ESM-2 (650M) embeddings locally via Hugging Face.
Replaces create_esm_data.py (NVIDIA API) with local inference.

Usage:
    cd 01.Code
    python create_esm_hf.py
"""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ---------- Settings ----------
DATA_DIR = '../02.Datasets/uniport_2022_5/'
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
BATCH_SIZE = 4          # Reduced for Quadro T2000 4GB VRAM
MAX_SEQ_LEN = 1024

FILES = [
    ('f_train.csv',     'esm_f_train.pt'),
    ('f_eval.csv',      'esm_f_eval.pt'),
    ('f_time_test.csv', 'esm_f_time_test.pt'),
]


def generate_embeddings(csv_path, output_path):
    if os.path.exists(output_path):
        print(f'[SKIP] {output_path} already exists')
        return

    print(f'\n=== Processing {csv_path} -> {output_path} ===')
    df = pd.read_csv(csv_path)
    sequences = df.iloc[:, 1].astype(str).str.strip().tolist()
    total = len(sequences)
    print(f'Total sequences: {total}')

    # Truncate to model max length
    seqs = [s[:MAX_SEQ_LEN] for s in sequences]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {MODEL_NAME} on {device} ...')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, total, BATCH_SIZE), desc='Batches'):
            end = min(start + BATCH_SIZE, total)
            batch_seqs = seqs[start:end]

            # Tokenize (ESM adds <cls> and <eos> automatically)
            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state  # [batch, seq_len, 1280]

            # Mean pooling with attention mask (exclude padding tokens)
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_embeddings.cpu())

    esm_tensor = torch.cat(all_embeddings, dim=0)
    print(f'Output tensor shape: {esm_tensor.shape}  (expected: [{total}, 1280])')
    torch.save(esm_tensor, output_path)
    print(f'Saved to {output_path}')


def main():
    for csv_name, pt_name in FILES:
        csv_path = os.path.join(DATA_DIR, csv_name)
        output_path = os.path.join(DATA_DIR, pt_name)
        if not os.path.exists(csv_path):
            print(f'[WARN] {csv_path} not found, skipping')
            continue
        generate_embeddings(csv_path, output_path)

    print('\nDone! All embeddings generated with Hugging Face.')


if __name__ == '__main__':
    main()
