# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import lightly_train

author = "Lightly Team"
copyright = "2024-%Y, Lightly"
project = "LightlyTrain"
website_url = "https://www.lightly.ai/"
version = lightly_train.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Markdown support
    "sphinx.ext.autodoc",  # Automatic Python documentation
    "sphinx.ext.intersphinx",  # Link to class from type hints
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinx_design",  # Design
    "sphinx_inline_tabs",  # Tabs
    "sphinxcontrib.googleanalytics",  # Google Analytics
]

googleanalytics_id = "G-9ZFQ8ZQS6H"
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_css_files = [
    "custom.css",  # File in _static/custom.css
]
html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#a72264",
    },
    # TODO(Guarin, 09/24): Find better color for dark mode. The current one is a bit
    # too dark.
    "dark_css_variables": {
        "color-brand-primary": "#a72264",
    },
    "light_logo": "lightly_train_light.svg",
    "dark_logo": "lightly_train_dark.svg",
}
html_sidebars = {
    # Have to set this to include the version in the sidebar (`sidebar/version.html`).
    # All other files are defaults from the theme.
    "**": [
        "sidebar/brand.html",
        "sidebar/version.html",  # File in _templates/sidebar/version.html
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}
html_static_path = ["_static"]


# -- Myst Configuration -------------------------------------------------

myst_enable_extensions = [
    "colon_fence",  # Allows to use markdown within directives, see https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#code-fences-using-colons
]
myst_heading_anchors = 3  # Creates anchors for headings up to level 3
