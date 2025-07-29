# `deeprm qc` – QC utilities

```bash
deeprm qc --help
```

```{eval-rst}
.. argparse::
   :module: deeprm.qc.cli
   :func: parser
   :prog: deeprm qc
```

Subcommands:
- `run` – basic QC
- `alignment` – inspect alignments and metrics
- `block` – inspect block-level signals

Example:
```bash
deeprm qc alignment -i <reads.bam> -o qc/alignment
```
