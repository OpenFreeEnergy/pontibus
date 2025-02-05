#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Incase the project was not installed
import os
from pathlib import Path
import shutil
import sys
sys.path.insert(0, os.path.abspath('..'))

import pontibus


# -- Project information -----------------------------------------------------

project = 'pontibus'
copyright = ("2024, Open Free Energy.")
author = 'The OpenFE Development Team'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
release = ''


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click.ext",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_toolbox.collapse",
    "sphinx.ext.autosectionlabel",
    "sphinx_design",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "docs._ext.sass",
    "myst_parser",
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.mathjax",
]


intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "scikit.learn": ("https://scikit-learn.org/stable", None),
    "openmm": ("http://docs.openmm.org/latest/api-python/", None),
    "rdkit": ("https://www.rdkit.org/docs", None),
    "openeye": ("https://docs.eyesopen.com/toolkits/python/", None),
    "mdtraj": ("https://www.mdtraj.org/1.9.5/", None),
    "openff.units": ("https://docs.openforcefield.org/projects/units/en/stable", None),
    "gufe": ("https://gufe.readthedocs.io/en/latest/", None),
    "openfe": ("https://openfe.readthedocs.io/en/latest/", None),
}

autoclass_content = "both"
# Make sure labels are unique
# https://www.sphinx-doc.org/en/master/usage/extensions/autosectionlabel.html#confval-autosectionlabel_prefix_document
autosectionlabel_prefix_document = True

autodoc_pydantic_model_show_json = False

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "inherited-members": "GufeTokenizable,BaseModel",
    "undoc-members": True,
    "special-members": "__call__",
}
toc_object_entries_show_parents = "hide"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "**/Thumbs.db",
    "**/.DS_Store",
    "_ext",
    "_sass",
    "README.md",
    "**/README.md",
    "ExampleNotebooks",
]

autodoc_mock_imports = [
    "matplotlib",
    "mdtraj",
    "openmmforcefields",
    "openmmtools",
    "pymbar",
]

# Extensions for the myst parser
myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
    "smartquotes",
    "replacements",
    "deflist",
    "attrs_inline",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "ofe_sphinx_theme"
html_theme_options = {
    "logo": {"text": "OpenFE GROMACS Protocols"},
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/OpenFreeEnergy/pontibus",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "accent_color": "DarkGoldenYellow",
    "navigation_with_keys": False,
}
html_logo = "_static/Squaredcircle.svg"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# replace macros
rst_prolog = """
.. |rdkit.mol| replace:: :class:`rdkit.Chem.rdchem.Mol`
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

## Add below when we have notebooks
#example_notebooks_path = Path("ExampleNotebooks")
#
#try:
#    if example_notebooks_path.exists():
#        pass
#    else:
#        source = Path("../examples")
#        shutil.copytree(source, example_notebooks_path)
#except Exception as e:
#    raise OSError("Could not copy over example notebooks")
