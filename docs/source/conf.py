"""Sphinx configuration for the DeepRM project."""

# ─── Path so autodoc finds project packages ───────────────────────────────────
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# ─── Basic project info ───────────────────────────────────────────────────────
project = "DeepRM"
author = "Laboratory of Computational Biology, School of Biological Sciences, Seoul National University"
copyright = "2025, Laboratory of Computational Biology, School of Biological Sciences, Seoul National University"
release = "1.0.4"

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
    "sphinxarg.ext",
    "sphinx_design",
]


html_theme = "furo"
html_static_path = ["_static"]
html_logo = "../images/deeprm.png"

mermaid_version = "10.9.1"

templates_path = ["_templates"]
exclude_patterns = []

# --- Autodoc look & feel ---
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"  # types go to description (more readable)
autodoc_inherit_docstrings = True


# --- Napoleon for NumPy docstrings ---
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
napoleon_custom_sections = ["notes"]
# Make Napoleon rewrite common short names to canonical targets
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "np.ndarray": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
}

# --- MyST ---
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "substitution"]


# --- Make imports reliable for RTD and local builds ---
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))  # if using src/ layout
autodoc_mock_imports = ["torch", "pysam", "pod5", "pandas", "sklearn", "scipy", "tqdm", "matplotlib"]

# -- Intersphinx cross-links (optional) --------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", {}),
    "numpy": ("https://numpy.org/doc/stable/", {}),
    "pandas": ("https://pandas.pydata.org/docs/", {}),
    "torch": ("https://pytorch.org/docs/stable/", {}),  # <-- add this
}
