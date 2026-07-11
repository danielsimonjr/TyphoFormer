# Legacy — Original PyTorch Implementation

This folder holds the **original PyTorch implementation** of TyphoFormer. It is
kept for reference and reproducibility, and has been superseded by the
dependency-free pure-C port in [`../typhoformer-c/`](../typhoformer-c/).

## Contents

| Path | Purpose |
|:--|:--|
| `model/PGF_module.py` | Prompt-aware Gating Fusion module |
| `model/STTransformer.py` | Spatio-temporal Transformer backbone |
| `model/TyphoFormer.py` | Full model (PGF + encoder + autoregressive decoder) |
| `train_typhoformer.py` | Training entry point |
| `eval_typhoformer.py` | Evaluation script |
| `prepare_typhoformer_data.py` | Dataset preparation (sliding windows) |
| `generate_text_description_new.py` | GPT-4o natural-language descriptions |
| `generate_text_embeddings.py` | MiniLM (`all-MiniLM-L6-v2`) embeddings |
| `utils.py` | Shared helpers |

## Running

Run these from the **repository root** so the shared data (`data/`,
`embedding_chunks/`, `HURDAT_2new_3000.csv`) resolves and the `model` package
imports correctly:

```bash
python legacy/prepare_typhoformer_data.py     # build data/{train,val,test}
python legacy/train_typhoformer.py            # train (checkpoints -> ./checkpoints)
python legacy/eval_typhoformer.py             # evaluate
```

Requires the PyTorch stack: `torch>=2.1.0`, `transformers`,
`sentence-transformers`, `scikit-learn`, `torchinfo`, `openai`, `backoff`,
`tqdm`, `pandas`, `numpy`.
