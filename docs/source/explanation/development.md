````markdown
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
$ make check
```

If the lint fails, run the following to force the format:
```sh
$ make format
```

If you want to compile the documentation locally, run
```sh
$ make docs-serve
```

## Development Toolkit

All the following checks can be run by the following command:
```sh
$ make check-lite
```

This uses one of the supported Python versions.

For a full check using all supported Python versions,
```sh
$ make check
```

We prefer to set the configs for the following in `pyproject.toml`.

### Formatters: black, isort, autopep8
We use the following tools for code formatting:
- **black**: The standard auto-formatter for the project. Run with `make format`.
- **isort**: Automatically sorts import statements.
- **autopep8**: Automatically fixes code to comply with PEP8.
All of these are executed together via `make format`.

### Type Checkers: mypy
Type checking is performed using **mypy**.
Run `make check` to execute type checks.

### Tests: pytest, doctest
Tests are run using **pytest** and **doctest**.
All tests are executed with `make check`.
Pytest is parametrized to run across multiple Python versions.
