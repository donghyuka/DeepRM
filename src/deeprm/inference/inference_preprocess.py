"""Wrapper that uses a C++ preprocessing binary when available."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List

from deeprm.inference import inference_preprocess_python as py_prep
from deeprm.utils.logging import get_logger

log = get_logger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments, mirroring the Python preprocessing module."""
    py_prep.add_arguments(parser)
    parser.add_argument(
        "--preprocess-bin",
        type=str,
        default=None,
        help="Path to accelerated preprocessing binary (optional)",
    )


DEFAULT_BIN = os.path.join(Path(__file__).parent.resolve(), "deeprm_preprocess")


def _args_to_cmd(args: argparse.Namespace) -> List[str]:
    cmd: List[str] = []
    for key, value in vars(args).items():
        if key == "preprocess_bin":
            continue
        opt = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(opt)
        else:
            cmd.extend([opt, str(value)])
    return cmd


def main(args: argparse.Namespace) -> int:
    bin_path = args.preprocess_bin or os.environ.get("DEEPRM_PREPROCESS_BIN") or DEFAULT_BIN

    ## Check executable permissions
    if not os.path.isfile(bin_path):
        log.warning("Preprocessing binary not found at %s", bin_path)
    else:
        if not os.access(bin_path, os.X_OK):
            log.warning("Preprocessing binary at %s is not executable", bin_path)
            try:
                os.chmod(bin_path, os.stat(bin_path).st_mode | 0o111)
                log.info("Setting executable permissions for %s", bin_path)
            except Exception as exc:
                log.warning("Failed to set executable permissions: %s", exc)

    cmd = [str(bin_path)] + _args_to_cmd(args)
    try:
        log.info("Attempting to use accelerated preprocessing binary: %s", bin_path)
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        log.warning("Falling back to Python preprocessing: %s", exc)
        py_prep.main(args)
    return 0
