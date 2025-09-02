.PHONY: install lint type test check format

install:
	poetry install --with dev,ci

lint:
	poetry run black --check .

type:
	poetry run mypy -p earlysign

test:
	poetry run pytest

format:
	poetry run black .

docs-build:
	# Build Sphinx docs into docs/_build/html
	poetry run sphinx-build -b html docs/source docs/_build/html

docs-serve:
	# Serve docs locally with live-reload (requires sphinx-autobuild)
	poetry run sphinx-autobuild docs/source docs/_build/html --open-browser \
		--ignore docs/source/autoapi

check: lint type test docs-build
