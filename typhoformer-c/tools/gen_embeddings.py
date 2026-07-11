#!/usr/bin/env python3
"""
gen_embeddings.py — encode the GPT-4o descriptions into MiniLM
(all-MiniLM-L6-v2, 384-d, L2-normalized) sentence embeddings, written as
row-aligned `emb_chunk_000.npy, emb_chunk_001.npy, ...` (float32) that the C
loader reads directly. Offline preprocessing for TyphoFormer-C (Python +
sentence-transformers).

Usage:
    python tools/gen_embeddings.py <with_descriptions.csv> <out_dir> [chunk_size=500]
    # e.g.  ... ../HURDAT_2new_3000_with_descriptions.csv ../embedding_chunks 500
"""
import os
import sys

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main(csv_path, out_dir, chunk_size=500):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    if "description" not in df.columns:
        sys.exit("input CSV must contain a 'description' column")
    texts = df["description"].astype(str).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    for start in range(0, len(texts), chunk_size):
        cid = start // chunk_size
        emb = model.encode(
            texts[start:start + chunk_size],
            batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
        ).astype("<f4")
        path = os.path.join(out_dir, f"emb_chunk_{cid:03d}.npy")
        np.save(path, emb)
        print(f"chunk {cid}: {emb.shape} -> {path}")
    print(f"done: {len(texts)} embeddings in {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 500)
