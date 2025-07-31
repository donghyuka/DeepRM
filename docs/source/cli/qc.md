# `deeprm qc`

Quality-control utilities for predictions and site-level outputs.

## Group help

```{argparse}
:module: deeprm.qc.cli
:func: parser
:prog: deeprm qc
```

## Examples

```bash
# Summarize per-sample metrics
deeprm qc summary --in pileup/ --out qc/summary.csv

# Visualize distribution of DOM/PM6A (requires matplotlib)
deeprm qc plot --in pileup/ --out qc/plots/
```
