# `deeprm inference`

Run the inference workflow (prep → run → pileup). Heavy deps (pysam, pod5, torch) are **optional** and only needed for the relevant subcommands.

## Group help

```{argparse}
:module: deeprm.inference.cli
:func: parser
:prog: deeprm inference
```

## Examples

```bash
# 1) Preprocess inputs (external C++ binary or Python fallback)
deeprm inference prep -i raw/ -o prep/ --threads 8

# 2) Run model inference
deeprm inference run -m model/deeprm_model.pt -d prep/ -o pred/

# 3) Aggregate site-level metrics
deeprm inference pileup -i pred/ -o pileup/ -b mpileup.filtered.pkl
```

## Notes

- `prep` may call an external binary; specify it with `--preprocess-bin` or `DEEPRM_PREPROCESS_BIN`.
- GPU users: install the correct Torch build first (CUDA/ROCm), then `pip install "deeprm[inference]"`.
