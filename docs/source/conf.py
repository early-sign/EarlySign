import os
import sys

# Put project root on sys.path so autodoc can import the package if needed
sys.path.insert(0, os.path.abspath("../.."))
# Ensure local docs extensions are importable
sys.path.insert(0, os.path.abspath("."))

project = "EarlySign"
author = "EarlySign Developers"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "myst_nb",
    "sphinx_copybutton",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "sphinx.ext.viewcode",
]

# autodoc_pydantic settings
autodoc_pydantic_model_show_json = True
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = True
autodoc_pydantic_model_show_validator_members = True
autodoc_pydantic_model_show_field_summary = True
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_field_list_validators = True
autodoc_pydantic_field_doc_policy = "both"
templates_path = ["_templates"]
exclude_patterns = ["ADR_template.rst", "_templates"]

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
            "show_navbar_depth": 1,
            "navigation_depth": 10,
            "max_navbar_depth": 10,
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
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.poetry/**",
    "**/spec/**",
    "**/.Trash/**",
    "**/.git/**",
    "**/.github/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "verify_*.py",
]

autodoc_typehints = "description"

autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
]

add_module_names = False

napoleon_use_ivar = True

autoapi_template_dir = os.path.join(os.path.dirname(__file__), "_templates", "autoapi")

autoapi_keep_files = True

autoapi_python_class_content = "both"  # "class", "init"
autoapi_python_use_implicit_namespaces = True

suppress_warnings = [
    "autoapi.python_import_resolution",
    "autodoc.import_object",
]


def generate_schema_rst(app):
    """
    Dynamically generates the ES3 Schema Reference page (schema.rst).
    Scans the ES3/schema directory for all .tsp files and creates literalinclude sections.
    """
    docs_source_dir = app.srcdir
    repo_root = os.path.abspath(os.path.join(docs_source_dir, "..", ".."))
    schema_dir = os.path.join(repo_root, "ES3", "schema")
    output_file = os.path.join(docs_source_dir, "reference", "schema.rst")

    if not os.path.exists(schema_dir):
        return

    # Collect all .tsp files in root
    tsp_files = []
    for f in sorted(os.listdir(schema_dir)):
        if f.endswith(".tsp"):
            tsp_files.append(os.path.join(schema_dir, f))

    # Builtin subdirs
    builtin_dir = os.path.join(repo_root, "earlysign", "builtin")
    if os.path.exists(builtin_dir):
        # We sort directories to ensure deterministic order (e.g., AVI before YEAST)
        for root, dirs, files in os.walk(builtin_dir):
            dirs.sort()
            for f in sorted(files):
                if f.endswith(".tsp"):
                    tsp_files.append(os.path.join(root, f))

    content = [
        "ES3 Schema Reference",
        "====================",
        "",
        "**ES3 (EarlySign Static Schema)** is the formal specification for all data structures",
        "used throughout the EarlySign framework. It ensures consistency across different",
        "components and provides a language-neutral definition of our data models.",
        "",
        "TypeSpec Origin",
        "---------------",
        "",
        "The authoritative source for these schemas is defined in **TypeSpec** (formerly ADL).",
        "These definitions serve as the primary source of truth, from which Pydantic models",
        "are automatically generated for the Python implementation.",
        "",
        "The source files are located in the ``ES3/schema`` directory of the repository.",
        "",
    ]

    for tsp_path in tsp_files:
        rel_path = os.path.relpath(tsp_path, os.path.join(docs_source_dir, "reference"))
        filename = os.path.basename(tsp_path)

        # Better title based on filename
        title = filename.replace(".tsp", "").replace("es3_v1", "Manifest")
        if title == "Manifest":
            title = "Core Manifest"
        elif title.upper() in ["GST", "AVI", "YEAST"]:
            title = title.upper()
        # If it's already PascalCase, just replace underscores if any
        else:
            title = title.replace("_", " ")

        content.extend(
            [
                f"{title}",
                f"{'-' * len(title)}",
                "",
                f".. literalinclude:: {rel_path}",
                "   :language: typescript",
                "   :linenos:",
                f"   :caption: {filename}",
                "",
            ]
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_content = "\n".join(content)

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            if f.read() == new_content:
                return

    with open(output_file, "w") as f:
        f.write(new_content)


def setup(app):
    app.connect("builder-inited", generate_schema_rst)
