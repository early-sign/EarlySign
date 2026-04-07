.PHONY: install lint type test check format docs-build docs-serve lint-type-test compile-ES3

install:
	poetry install --with dev,ci
	npm install

lint:
	poetry run black --check .

type:
	poetry run mypy -p earlysign

test:
	poetry run pytest

spec-test:
	poetry run pytest earlysign/tests/spec_tests/

compile-ES3:
	$(MAKE) -C ES3 install
	$(MAKE) -C ES3 compile

format:
	# Format code using one Python version
	# Sort imports, fix lint, and then run Black for final opinionated formatting
	poetry run isort earlysign
	# ruff: apply safe fixes and also enable unsafe fixes to remove leftover
	# unused assignments/variables when desired. This can change code; run in
	# a branch and run tests after.
	poetry run ruff check --fix --unsafe-fixes earlysign
	poetry run black .

docs-build:
	# Build Sphinx docs into docs/_build/html
	poetry run sphinx-build -b html docs/source docs/_build/html

docs-serve:
	# Serve docs locally with live-reload (requires sphinx-autobuild)
	poetry run sphinx-autobuild docs/source docs/_build/html --open-browser --ignore docs/source/autoapi --watch earlysign

# Combined lint, type check, and test command
lint-type-test:
	poetry run ruff check earlysign
	poetry run isort --check-only earlysign
	poetry run black --check .
	poetry run mypy -p earlysign
	poetry run pytest

check:
	# Run comprehensive checks across all Python versions
	@for version in 3.11 3.12 3.13; do \
		echo "Checking with Python $$version..."; \
		mise exec python@$$version -- make lint-type-test; \
	done
	make docs-build

check-lite:
	$(MAKE) -C ES3 check-sync
	mise exec python -- make lint-type-test
	make docs-build
