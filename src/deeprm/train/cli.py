# src/deeprm/qc/cli.py
import argparse, sys
from importlib import import_module

_COMMANDS = {
    "run":        "deeprm.train.train",
    "prep":      "deeprm.train.train_preprocess",
    "compile":  "deeprm.train.train_compile",
}

def _delegate(modname: str, argv):
    mod = import_module(modname)
    if hasattr(mod, "main"):
        return mod.main(argv)
    raise SystemExit(f"✖ {modname} has no main()")

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="deeprm Train", description="Train helpers")
    sp = p.add_subparsers(dest="cmd", required=True)
    for name, mod in _COMMANDS.items():
        sp.add_parser(name, help=f"{name} Train task")
    ns, rest = p.parse_known_args(argv)
    _delegate(_COMMANDS[ns.cmd], rest)

if __name__ == "__main__":
    main()
