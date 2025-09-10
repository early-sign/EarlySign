.PHONY: install lint type test check format docs-build docs-serve

install:
	poetry install --with dev,ci

lint:
	poetry run black --check .

type:
	poetry run mypy -p earlysign

test:
	poetry run pytest

format:
	# Format code using one Python version
	poetry run black .

docs-build:
	# Build Sphinx docs into docs/_build/html
	poetry run sphinx-build -b html docs/source docs/_build/html

docs-serve:
	# Serve docs locally with live-reload (requires sphinx-autobuild)
	poetry run sphinx-autobuild docs/source docs/_build/html --open-browser \
		--ignore docs/source/autoapi

check:
	# Run comprehensive checks across all Python versions
	@for version in 3.11 3.12 3.13; do \
		echo "Checking with Python $$version..."; \
		mise exec python@$$version -- make lint; \
		mise exec python@$$version -- make type; \
		mise exec python@$$version -- make test; \
	done
	make docs-build
