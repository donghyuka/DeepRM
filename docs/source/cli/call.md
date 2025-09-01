# `deeprm call`

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
deeprm call prep -i raw/ -o prep/ --threads 8

# 2) Run model inference
deeprm call run -m model/deeprm_model.pt -d prep/ -o pred/

# 3) Aggregate site-level metrics
deeprm call pileup -i pred/ -o pileup/ -b mpileup.filtered.pkl
```