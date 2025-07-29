# `deeprm inference` – Inference commands

```bash
deeprm inference --help
```

```{eval-rst}
.. argparse::
   :module: deeprm.inference.cli
   :func: parser
   :prog: deeprm inference
```

Subcommands:
- `prep` – preprocess raw inputs for inference
- `run` – run model inference on preprocessed data
- `pileup` – aggregate predictions into site-level metrics

Examples:
```bash
deeprm inference prep -i <raw_dir> -o <prep_dir>
deeprm inference run -m <weights.pt> -d <prep_dir> -o <pred_dir>
deeprm inference pileup -i <pred_dir> -o <pileup_dir>
```
