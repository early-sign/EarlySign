# Contributing to the Project

## Local setup

This project uses [mise](https://mise.jdx.dev/) to manage multiple Python versions and [Poetry](https://python-poetry.org/) for dependency management.

### Initial setup
```sh
# Install mise if you haven't already
$ curl https://mise.run | sh

# Install Python versions and project dependencies
$ mise run install-all
```

The `install-all` command will:
1. Install the support-target Python versions via mise
2. Install project dependencies using Poetry

### Multi-version environment details

We use `virtualenvs.in-project = true` in Poetry configuration, which means:
- All dependencies are installed in a single `.venv` directory
- All Python versions share the same dependency versions for consistency
- mise ensures each version uses the appropriate Python interpreter when executing commands

This approach provides efficient storage usage while maintaining compatibility across all supported Python versions.

## Commands
Always make sure to run the following to check if the tests and lint checks pass across all Python versions:
```sh
$ make check-lites
```
This uses one of the supported Python versions.

For a full check using all supported Python versions,
```sh
$ make check
```

If the lint fails, run the following to force the format:
```sh
$ make format
```

If you want to compile and check the documentation locally, run
```sh
$ make docs-serve
```

### Formatters / Linters: black, isort, ruff
We use the following tools for formatting and lint-based automatic fixes:
- **black**: The standard auto-formatter for the project. Run with `make format`.
- **isort**: Automatically sorts import statements (configured to follow Black's style).
- **ruff**: A fast linter with automatic-fix capabilities (`ruff --fix`). Use ruff to remove unused imports and apply lightweight lint-based fixes before running Black.

Typical ordering when formatting the repository (what `make format` performs):

1. `isort` — sort and group `import` statements (applies to `earlysign/` only).
2. `ruff --fix` — remove unused imports and apply quick lint fixes.
3. `black` — final deterministic formatting pass.

This order avoids conflicts between tools and ensures a consistent, reproducible code style.

### Type Checkers: mypy
Type checking is performed using **mypy**.
Run `make check` to execute type checks.

### Tests: pytest, doctest
Tests are run using **pytest** and **doctest**.
All tests are executed with `make check` or `make check-lite`.
Pytest is parametrized to run across multiple Python versions (if using `make check`).
