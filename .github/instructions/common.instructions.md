---
applyTo: "**"
---

# TL;DR
- `make format` to auto-format code, and then `make check` to run all sanity checks (lints, tests, and type checks).
- Main architectural decisions:
  - `event-sourcing` to accommodate existing complex sequential procedures and ones that we are yet to see. We call the event-source a `ledger`.
    - For example, we separate the statistic calculation and the criteria to turn the statsitical value into signals or decisions. We can pass the values and other info via writing to and reading from the ledger.
    - Multiple ledger operators (e.g., observation, statistic, criteria, signal, other kinds of messages) are coordinated to form a statsitical method.
  - We rely on `ibis-framework` to enable efficient and consistent data-handling. The modules that operate on the ledger can leverage the features of `ibis`.

# Development
- Whenever you run a terminal command and the output seems empty, you should absolutely always autonomously check out "terminalLastCommand" to fetch the results. Sometimes you need to wait for a few seconds and check again. Keep in mind that VSCode GitHub copilot extension (the platform you are working in) seems to have issues with the integrated terminal connection. Therefore, you need the above workaround.
- This repository uses `make` to organize workspace tasks. For the details, see the Makefile.
- We use `poetry`. So the commands usually need to be run as `poetry run python ...` etc.
- At the end of the edits, make sure to run `make format` to ensure the code is compliant to the formatting standards of this repository.
- Also run `make check` occasionally to make sure the code passes the tests and lint checks.
- Whenever you are adding a dependency, use `poetry add`. Do not try to specify versions unless that is absolutely necessary, so that poetry can do the version resolution for you. Do not write them directly edit `pyproject.toml` for this purpose.
    - You can edit `pyproject.toml` after adding the packages for formatting purposes.

# Preferences
- We prefer keeping all the test and formatting configurations in `pyproject.toml`.

# Architecture Overview

EarlySign is built around an **event-sourcing** architecture with a typed, append-only **ledger** as the central abstraction. This design enables strict reproducibility and supports multiple statistical procedures operating over shared data.

## Core Components

- **Ledger** (`earlysign.core.ledger`): Central event store using ibis-framework for backend-agnostic data operations. Supports DuckDB, Polars, and other ibis backends
- **API Layer** (`earlysign.api`): Business-oriented facade using domain terminology (e.g., `interim_analysis()`, `guardrail_monitoring()`)
- **Stats Engine** (`earlysign.stats`): Statistical methods organized as `schemes/` (experiment types) and `methods/` (algorithms)
- **Components** (`earlysign.core.components`): Base classes for statistics, criteria, signalers, and observers
- **Runtime** (`earlysign.runtime`): Execution environments and experiment templates

## Data Flow

```
Observations → Ledger → Statistics → Criteria → Signalers → Decisions → Ledger
```

All operations are **pull-based**: components read from the ledger through typed query interfaces, never direct data passing.

## Key Conventions

- **Multi-backend testing**: Use parametrized fixtures for DuckDB/Polars: `@pytest.fixture(params=["duckdb", "polars"])`
- **Namespace organization**: Components use enum-based namespaces (`earlysign.core.names.Namespace`)
- **Template pattern**: Experiments inherit from `ExperimentTemplate` with `setup()`, `step()`, `analyze()` methods
- **API first**: User-facing functionality in `earlysign.api.*` using business terminology
- **Type safety**: Strict mypy with API layer excluded (`exclude = ["earlysign/api/.*"]`)

## Common Patterns

### Ledger Operations
```python
# Always use structured payloads with types
ledger.write_event(
    time_index="t1", namespace=Namespace.OBS, kind="observation",
    payload_type="TwoProportion", payload={"n_treatment": 100, ...}
)

# Query with structured data
rows = list(ledger.query_structured("TwoProportion", n_treatment=100))
```
