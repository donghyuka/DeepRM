# Getting Started

Welcome to **DeepRM**! This short guide shows the fastest path from zero to results.

## 1. Installation

Follow the instructions in the project **README** or on the documentation's *Installation* page.

## 2. Run the quick‑start script

We provide a tiny example pipeline (requires GPU‑enabled PyTorch and some sample data):

```bash
RAW_DIR=/data/raw_reads  \
MODEL=/models/deeprm_weights.pt \
examples/quickstart.sh
```

The script will:

1. Preprocess raw reads (`deeprm inference prep`).
2. Run model inference on GPU (`deeprm inference run`).
3. Aggregate predictions into site‑level scores (`deeprm inference pileup`).

Outputs land in `./work/pileup/` by default.

## 3. Next Steps

- **CLI reference** → Use the sidebar or run `deeprm <group> --help`.
