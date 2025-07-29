"""
deeprm package init.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # resolve the installed package version at runtime
    __version__ = version("deeprm")
except PackageNotFoundError:  # editable/uninstalled checkouts
    __version__ = "0+unknown"

__all__ = ["__version__"]
