# `deeprm train`

Tools for preparing data and training models. Torch is optional at install time; imported only when needed.

## Group help

```{argparse}
:module: deeprm.train.cli
:func: parser
:prog: deeprm train
```

## Examples

```bash
# Prepare training data
deeprm train prep --in train_raw/ --out train_prep/

# Compile dataset shards
deeprm train compile --in train_prep/ --out ds/

# Launch training
deeprm train run --config configs/train.yaml --out runs/exp1
```