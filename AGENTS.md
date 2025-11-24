---
applyTo: "**"
---

# TL;DR
- Language
  - In the code and comments, stick to English even if the conversation in the chat is in some other language. Proactively translate into English if you find other languages in the code (with the exception being in the internationalized versions of the docs).
- Toolkit
  - `make format` to auto-format code, and then `make check-lite` to run all sanity checks (lints, tests, and type checks). Therefore, you want to run `make format && make check-lite` for every set of edits you make.
  - We use `poetry` to manage dependencies. So you need to use `poetry run python` when you run Python commands.
  - tqdm progress bars follow the module logger level: INFO (or lower) enables progress, WARNING (or higher) suppresses it. Do not add per-call toggles like `enable_progress`; tune logging instead or use the `EARLYSIGN_ENABLE_TQDM` / `EARLYSIGN_DISABLE_TQDM` environment overrides when necessary.
- Main architectural decisions:
  - `__init__.py` purpose: Designed for documentation generation and namespace structure
    - ❌ **DO NOT** use for access restriction or selective imports (no `__all__` should be used except for the version exports in the top-level `__init__.py`)
    - ✅ **DO** provide namespace-level docstrings with doctests
  - `event-sourcing` to accommodate existing complex sequential procedures and ones that we are yet to see. We call the event-source a `ledger`.
    - For example, we separate the statistic calculation and the criteria to turn the statsitical value into signals or decisions. We can pass the values and other info via writing to and reading from the ledger.
    - Multiple ledger operators (e.g., observation, statistic, criteria, signal, other kinds of messages) are coordinated to form a statsitical method.
  - We rely on `ibis-framework` to enable efficient and consistent data-handling. The modules that operate on the ledger can leverage the features of `ibis`.

# Development
- Whenever you run a terminal command and the output seems empty, you should absolutely always autonomously check out "terminal_last_command" to fetch the results. Sometimes you need to wait for a few seconds and check again. Keep in mind that VSCode GitHub copilot extension (the platform you are working in) seems to have issues with the integrated terminal connection. Therefore, you need the above workaround.
- This repository uses `make` to organize workspace tasks. For the details, see the Makefile.
- We use `poetry`. So the commands usually need to be run as `poetry run python ...` etc.
- At the end of the edits, make sure to run `make format` to ensure the code is compliant to the formatting standards of this repository.
- Also run `make check-lite` occasionally to make sure the code passes the tests and lint checks.
- Notebook regressions (`make check-lite` runs nbregression) sometimes need permission to start Jupyter kernels/bind ports; request the necessary execution/port-binding approval when prompted so the tests can run.
- Whenever you are adding a dependency, use `poetry add`. Do not try to specify versions unless that is absolutely necessary, so that poetry can do the version resolution for you. Do not write them directly edit `pyproject.toml` for this purpose.
    - You can edit `pyproject.toml` after adding the packages for formatting purposes.

# Preferences
- We prefer keeping all the test and formatting configurations in `pyproject.toml`.

# Architecture Overview

EarlySign is built around an **event-sourcing** architecture with a typed, append-only **ledger** as the central abstraction. This design enables strict reproducibility and supports multiple statistical procedures operating over shared data.

The conceptual decisions are stored and updated in docs/source/reference/ADR.

## Core Components

- **Ledger** (`earlysign.core`): Central event store using ibis-framework for backend-agnostic data operations. Supports DuckDB, Polars, and other ibis backends
- **Framework** (`earlysign.framework`): Building blocks that standardize the read-write operations to the ledger.
- **Stats Engine** (`earlysign.stats`): Statistical methods organized as `essentials/` (the essential classes and computations) and `applications/` (closer to scenario-based interfaces, uses `earlysign.framework`)
- **API Layer** (`earlysign.templates`): Business-oriented facade using domain terminology (e.g., `ab_tests`, `guardrail_monitoring`)

## 🔧 Coding Standards

### Import Conventions
- **Absolute imports only**: Always use `from earlysign.package.module import Component`
- **Poetry-based imports**: Leverage poetry's package resolution for consistent imports
- **No relative imports**: Avoid `from .module import Component` patterns

### Module Organization
- **`__init__.py` purpose**: Designed for documentation generation and namespace structure
  - ❌ **DO NOT** use for access restriction or selective imports
  - ✅ **DO** provide namespace-level docstrings with doctests
- **Direct access**: Components should be accessible via absolute paths regardless of `__init__.py`

### Code Quality
- **No development artifacts**: Remove working comments before final delivery
  - ❌ Examples: "this method moved from...", "TODO: refactor later"
  - ✅ Keep only comments that add value to future maintainers
- **Clean commits**: Ensure production code doesn't contain debug prints or temporary code
- **Consistent formatting**: Follow project formatting standards (enforced via `make format`)
- **Exports & language features**: Avoid `__all__` exports (except the top-level version string) and do not rely on `from __future__ import annotations`; code should run without future-import shims.
- **Exception style**: Prefer control flow or helper queries instead of `try`/`except` blocks when checking for record existence (e.g., query for rows and branch on emptiness rather than catching `LookupError`).

### Type Safety
- **Type hints**: Use comprehensive type annotations for all public APIs
- **TypedDict**: Use for structured payloads and configuration objects
- **Protocol compliance**: Ensure components implement required protocols correctly
- **Optional arguments**: Do not annotate parameters as optional if passing `None` is not supported; require non-`None` inputs or provide a dedicated helper/factory for the optional flow.
