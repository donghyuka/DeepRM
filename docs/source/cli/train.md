# `deeprm train` – Training commands

```bash
deeprm train --help
```

```{eval-rst}
.. argparse::
   :module: deeprm.train.cli
   :func: parser
   :prog: deeprm train
```

Subcommands:
- `prep` – prepare training data
- `run` – launch training
- `compile` – optional ahead-of-time compilation helpers

Example:
```bash
deeprm train prep -i <data_dir> -o <work_dir>
deeprm train run -c <config.yaml> -o <runs/exp1>
```
