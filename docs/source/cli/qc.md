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
# Inspect run
deeprm qc run -i <prediction_dir> -o <output_dir>

# Inspect training data
deeprm qc block -i <block_file> -o <output_dir> -t m6A

# Inspect alignment
deeprm qc alignment -i <bam_file> -o <output_dir>
```
