#!/usr/bin/env python3
"""
npy_dict_to_bin.py — convert the repository's pre-split, pickled .npy sample
files (dicts of {"input": X[in_len, feat], "target": Y[pred_len, 2]}) into a
flat binary (.tfb) that the pure-C loader can read without a pickle parser.

Usage:
    python tools/npy_dict_to_bin.py <in_dir> <out.tfb>
    # e.g.  python tools/npy_dict_to_bin.py ../data/val ../data/val.tfb

.tfb layout (little-endian):
    magic  "TFB1"                         (4 bytes)
    int32  n_samples, in_len, feat_dim, pred_len, out_dim
    then, per sample:
        float32 input[in_len * feat_dim]
        float32 target[pred_len * out_dim]
"""
import glob
import os
import struct
import sys

import numpy as np


def main(in_dir, out_path):
    files = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    if not files:
        sys.exit(f"no .npy files in {in_dir}")

    samples = []
    in_len = feat = pred_len = out_dim = None
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        X = np.ascontiguousarray(d["input"],  dtype="<f4")   # [in_len, feat]
        Y = np.ascontiguousarray(d["target"], dtype="<f4")   # [pred_len, 2]
        if in_len is None:
            in_len, feat = X.shape
            pred_len, out_dim = Y.shape
        elif X.shape != (in_len, feat) or Y.shape != (pred_len, out_dim):
            sys.exit(f"shape mismatch in {f}: {X.shape}, {Y.shape}")
        samples.append((X, Y))

    with open(out_path, "wb") as w:
        w.write(b"TFB1")
        w.write(struct.pack("<5i", len(samples), in_len, feat, pred_len, out_dim))
        for X, Y in samples:
            w.write(X.tobytes())
            w.write(Y.tobytes())

    print(f"wrote {out_path}: {len(samples)} samples, "
          f"in_len={in_len}, feat={feat}, pred_len={pred_len}, out_dim={out_dim}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
