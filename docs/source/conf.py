"""Sphinx configuration for the DeepRM project."""
import pathlib
import sys

# ─── Path so autodoc finds project packages ───────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

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
    "myst_parser",
    "sphinxcontrib.mermaid",
]

autosummary_generate = True  # create stub pages automatically
napoleon_google_docstring = True
html_theme = "sphinx_rtd_theme"

# ─── HTML tweaks ──────────────────────────────────────────────────────────────
html_logo   = "../images/deeprm_architecture.png"  # optional – adjust or remove
mermaid_version = "10.9.1"