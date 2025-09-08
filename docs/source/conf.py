import os
import sys

# Put project root on sys.path so autodoc can import the package if needed
sys.path.insert(0, os.path.abspath("../.."))
# Ensure local docs extensions are importable
sys.path.insert(0, os.path.abspath("."))

project = "EarlySign"
author = "Takeshi Teshima"

extensions = [
    "autoapi.extension",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_copybutton",
]
templates_path = []
exclude_patterns = []

nb_execution_mode = "off"

# Use a book-style theme for a book-like layout when available; fall back to
# a bundled theme (alabaster) so CI/local checks don't fail if the theme
# isn't installed yet (this lets `make check` run successfully while the
# dependency is being added via Poetry).
try:
    import importlib
    import importlib.util

    if importlib.util.find_spec("sphinx_book_theme") is not None:
        html_theme = "sphinx_book_theme"
        html_theme_options = {
            "repository_url": "https://github.com/early-sign/EarlySign",
            "use_repository_button": True,
            "use_issues_button": True,
            "path_to_docs": "docs/source",
            "launch_buttons": {"colab_url": "https://colab.research.google.com"},
        }
    else:
        html_theme = "alabaster"
        html_theme_options = {}
except Exception:
    html_theme = "alabaster"
    html_theme_options = {}


# ロゴ画像とCSSの設定（グローバル）
# myst config: enable useful parsing extensions for notebook-style content
myst_enable_extensions = [
    "deflist",
    "html_admonition",
    "html_image",
    "colon_fence",
]

master_doc = "index"

# Static assets and logo: place a logo image at docs/logo.png (project root)
# and reference it here so it appears in the top-left of the generated site.
# Use an absolute path to the repository-level docs/logo.png so Sphinx can
# include it even though it's outside docs/source/_static.
# Static assets: enable _static so we can inject a small CSS file to
# tweak the theme (hide prev/next footer when desired).
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logo.png"))

# ---------------------------------------------------------------------------
# sphinx-autoapi: generate API reference for the `earlysign` package
# Generated files will be placed under docs/source/reference/generated/
# ---------------------------------------------------------------------------
# Tell autoapi where the package source lives (project root / earlysign)
autoapi_type = "python"
# Restrict autoapi to the package source directory so it doesn't scan the
# virtualenv or unrelated repository folders. This keeps module names
# correctly rooted at `earlysign.*` while avoiding .venv recursion.
autoapi_dirs = ["../../earlysign"]

# Keep a conservative ignore list as a safety net
autoapi_ignore = [
    "**/docs/**",
    "**/scripts/**",
    "**/tests/**",
    "**/.venv/**",
    "**/__pycache__/**",
]

autodoc_typehints = "description"

autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
]

autoapi_python_class_content = "both"  # "class", "init"
