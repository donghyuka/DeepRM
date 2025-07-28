# PR1: Packaging refactor

- Introduces `src/deeprm/` package layout.
- Updates `pyproject.toml` for src layout and console script `deeprm = "deeprm.cli:main"`.
- Moves subpackages: `inference`, `train`, `qc`, `model`, `utils`; adds `cli.py` under `deeprm/`.
- Adds missing `__init__.py` files.
- Updates intra-package imports to `deeprm.<subpkg>` where applicable.

**Manual step after merge**: remove old top-level directories (`inference/`, `train/`, `qc/`, `model/`, `utils/`) and `cli.py` if present in repo root.
