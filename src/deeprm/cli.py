# src/deeprm/cli.py
from __future__ import annotations

import argparse
import sys
from argparse import RawTextHelpFormatter
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List

# -------------------------------------------------------------------
# Static registry used ONLY to render a rich top-level help page.
# This keeps top-level help detailed without importing heavy modules.
# -------------------------------------------------------------------
_HELP_REGISTRY: Dict[str, dict] = {
    "call": {
        "desc": "Call (inference) helpers",
        "module": "deeprm.inference.cli",
        "subcommands": [
            ("prep", "Preprocess raw inputs for inference"),
            ("run", "Run model inference on preprocessed data"),
            ("pileup", "Aggregate predictions into site-level metrics"),
        ],
    },
    "train": {
        "desc": "Training utilities",
        "module": "deeprm.train.cli",
        "subcommands": [
            ("prep", "Prepare training data"),
            ("compile", "Compile training dataset"),
            ("run", "Launch training (DDP supported)"),
        ],
    },
    "qc": {
        "desc": "Quality-control tools",
        "module": "deeprm.qc.cli",
        "subcommands": [
            ("run", "Basic QC summaries"),
            ("alignment", "Inspect alignments and metrics"),
            ("block", "Inspect block-level signals"),
        ],
    },
    "check": {
        "desc": "Verify installation",
        "module": "deeprm.utils.check_installation",
    },
}


def resolved_version() -> str:
    try:
        return version("deeprm")
    except PackageNotFoundError:
        # Editable/uninstalled checkouts
        try:
            from . import __version__  # type: ignore

            return __version__
        except Exception:
            return "0+unknown"


def _format_top_description() -> str:
    return (
        "[DeepRM unified command-line interface]\n\n"
        "Usage:\n"
        "  deeprm <group> [--help]\n"
        "  deeprm <group> <subcommand> [args]\n"
    )


def _format_groups_block() -> str:
    lines: List[str] = []
    lines.append("Groups & subcommands:\n")
    for g, info in _HELP_REGISTRY.items():
        lines.append(f"  {g:<10} {info['desc']}")
        if "subcommands" in info:
            for name, desc in info["subcommands"]:
                lines.append(f"    {name:<10} {desc}")
        lines.append("")  # blank line between groups
    return "\n".join(lines).rstrip() + "\n"


def _format_tips() -> str:
    return (
        "Tips:\n"
        "  • Use '--help' after any command to see its flags.\n"
        "  • See docs: https://deeprm.readthedocs.io/\n"
    )


def _build_top_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="deeprm",
        description=_format_top_description(),
        formatter_class=RawTextHelpFormatter,
        add_help=False,  # we add -h/--help manually to control placement
        epilog="\n".join(
            [
                _format_groups_block(),
                _format_tips(),
            ]
        ),
    )
    p.register("action", "parsers", argparse._SubParsersAction)  # for typing clarity

    # Top-level flags
    p.add_argument("--version", "-v", action="version", version=f"DeepRM {resolved_version()}")
    p.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit.",
    )

    # Skeleton subparsers: we *do not* import group modules here.
    sub = p.add_subparsers(dest="group", metavar="{inference,train,qc}")
    for g, info in _HELP_REGISTRY.items():
        # 'help' here enriches the default subcommand listing in argparse,
        # but we still print a custom epilog with more details.
        sub.add_parser(g, help=info["desc"], add_help=False)

    return p


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_top_parser()

    # No args or explicit help -> print the rich top-level page
    if not argv or argv[0] in ("-h", "--help", "help"):
        parser.print_help()
        return 0

    # Parse only enough to know the selected group (still no imports)
    ns, rest = parser.parse_known_args(argv)
    if ns.group is None:
        parser.print_help()
        return 2

    # Lazy-import just the chosen group CLI; it will handle its own argparse
    group_info = _HELP_REGISTRY.get(ns.group)
    if not group_info:
        parser.error(f"Unknown group '{ns.group}'")

    mod = import_module(group_info["module"])
    # Delegate the remaining argv to the group's entry() function
    return int(mod.entry(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
