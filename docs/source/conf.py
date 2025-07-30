"""Sphinx configuration for the DeepRM project."""

# ─── Path so autodoc finds project packages ───────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# ─── Basic project info ───────────────────────────────────────────────────────
project = "DeepRM"
author = "Laboratory of Computational Biology, School of Biological Sciences, Seoul National University"
copyright = "2025, Laboratory of Computational Biology, School of Biological Sciences, Seoul National University"
release = "0.1.0"

# ─── Extensions ───────────────────────────────────────────────────────────────

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]
myst_enable_extensions = ["colon_fence"]
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
autodoc_typehints = "description"

templates_path = ["_templates"]
exclude_patterns = []

autodoc_mock_imports = ["torch", "torchmetrics", "pysam", "pod5", "pandas", "networkx", "polyleven"]

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../images/deeprm.png"

mermaid_version = "10.9.1"
