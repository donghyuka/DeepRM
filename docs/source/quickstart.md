# Getting Started

Welcome to **DeepRM**! This short guide shows the fastest path from zero to results.

```{toctree}
:maxdepth: 1
:hidden:

cli/inference
cli/train
cli/qc
```

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

## 3. Keep a run log

Copy **RUN.md** to your experiment folder and fill in environment details, command line, and data provenance. This will make your results fully reproducible.

## Next Steps

- **CLI reference** → Use the sidebar or run `deeprm <group> --help`.
- **Examples** → See `examples/README.md` for more scripts.
