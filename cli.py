"""Command‑line interface for DeepRM.

This file installs a `deeprm` shell command (via the entry‑point in
`pyproject.toml`) that exposes four high‑level sub‑commands:

  • preprocess – data preparation utilities
  • train      – model training pipeline
  • inference  – run the trained model and post‑processing (pileup)
  • qc         – quality‑control helpers

Each sub‑command delegates to a corresponding module (`deeprm.preprocess.cli`,
`deeprm.train.cli`, etc.) if it exists.  If the target module defines a
``main(argv: list[str])`` function, we call it; otherwise we execute the
module as ``python -m`` so users can keep their existing scripts unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from types import ModuleType
from typing import List

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SUBMODULES: dict[str, str] = {
    "preprocess": "deeprm.preprocess.cli",
    "train": "deeprm.train.cli",
    "inference": "deeprm.inference.cli",
    "qc": "deeprm.qc.cli",
}


def _load_submodule(path: str) -> ModuleType:
    """Import *path* and return the module object."""
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError as exc:
        raise SystemExit(f"✖ Submodule '{path}' not found: {exc}") from exc


def _delegate(module: ModuleType, argv: List[str]) -> None:  # pragma: no cover
    """Run ``module.main`` if present; else emulate ``python -m module``."""
    if hasattr(module, "main"):
        module.main(argv)  # type: ignore[arg-type]
    else:
        # Fallback: run as script so existing __main__.py still works
        runpy.run_module(module.__name__, run_name="__main__")


# ---------------------------------------------------------------------------
# Top‑level argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deeprm",
        description="DeepRM unified command‑line interface",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    for cmd, help_text in (
            ("preprocess", "Prepare raw data for training / inference"),
            ("train", "Train a DeepRM model"),
            ("inference", "Run model prediction + pileup aggregation"),
            ("qc", "Run quality‑control routines"),
    ):
        # Each parser captures *all* remaining args to forward unchanged
        sp = subparsers.add_parser(cmd, help=help_text, add_help=False)
        sp.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help=f"Arguments passed through to 'deeprm {cmd}'",
        )

    return parser


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401 – imperative mood
    """Entry point for the ``deeprm`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)

    parser = _build_parser()
    ns = parser.parse_args(argv)

    if ns.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Dispatch to the appropriate sub‑module
    mod_path = _SUBMODULES[ns.command]
    module = _load_submodule(mod_path)
    _delegate(module, ns.args)


if __name__ == "__main__":  # pragma: no cover
    main()