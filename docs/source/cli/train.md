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

# Launch training (DDP optional)
deeprm train run --config configs/train.yaml --out runs/exp1
```

## Notes

- Install training extras (CPU): `pip install "deeprm[torch,train]"`.
- GPU/ROCm: install PyTorch from the official index URL first, then `pip install "deeprm[train]"`.
