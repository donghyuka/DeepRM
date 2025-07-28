"""Sphinx configuration for the DeepRM project."""

# ─── Path so autodoc finds project packages ───────────────────────────────────
import os, sys
sys.path.insert(0, os.path.abspath('../../src'))

# ─── Basic project info ───────────────────────────────────────────────────────
project   = "DeepRM"
author    = "Hyeonseo Hwang"
copyright = "2025, Your Lab"
release   = "0.1"

# ─── Extensions ───────────────────────────────────────────────────────────────

extensions = [
        "sphinx.ext.autodoc",
        "sphinx.ext.autosummary",
        "sphinx.ext.napoleon",
        "sphinx.ext.intersphinx",
        "myst_parser",
        "sphinxcontrib.mermaid",
        "sphinx_copybutton",
    ]
autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False
autodoc_typehints = "description"
intersphinx_mapping = {
        "python": ("https://docs.python.org/3", {}),
        "numpy":  ("https://numpy.org/doc/stable/", {}),
        "pandas": ("https://pandas.pydata.org/docs/", {}),
        "torch":  ("https://pytorch.org/docs/stable/", {}),
    }
source_suffix = {".md": "markdown"}
html_theme = "furo"
nitpicky = True  # fail on broken references during CI
# mock heavy deps to keep RTD fast/stable
autodoc_mock_imports = ["torch", "torchmetrics", "pysam", "pod5"]


# ─── HTML tweaks ──────────────────────────────────────────────────────────────
html_logo   = "../images/deeprm.png"  # optional – adjust or remove
mermaid_version = "10.9.1"