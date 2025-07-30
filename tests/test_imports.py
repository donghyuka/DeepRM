import importlib


def test_import_package():
    ## move to src/deeprm/__init__.py
    m = importlib.import_module("deeprm")
    assert hasattr(m, "__version__")


def test_import_cli_module():
    m = importlib.import_module("deeprm.cli")
    assert hasattr(m, "main")
