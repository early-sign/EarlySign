# Development
- This repository uses `make` to organize workspace tasks. For the details, see the Makefile.
- We use `poetry`. So the commands usually need to be run as `poetry run python ...` etc.
- At the end of the edits, make sure to run `make format` to ensure the code is compliant to the formatting standards of this repository.
- Also run `make check` occasionally to make sure the code passes the tests and lint checks.
- Whenever you are adding a dependency, use `poetry add`. Do not try to specify versions unless that is absolutely necessary, so that poetry can do the version resolution for you. Do not write them directly edit `pyproject.toml` for this purpose.
    - You can edit `pyproject.toml` after adding the packages for formatting purposes.

# Preferences
- We prefer keeping all the test and formatting configurations in `pyproject.toml`.
