"""
EarlySign Schema (ES3)
-----------------------

This package contains the core data schemas for the EarlySign framework, collectively
known as **ES3 (EarlySign Static Schema)**.

The ES3 schema serves as the single source of truth for:
1. **Protocols**: Formal definitions of trial designs, including task specifications,
   method configurations, and stopping policies.
2. **Logs/Events**: Standardized formats for recorded evidence, analysis results,
   and decision markers.

TypeSpec and Code Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The authoritative source for ES3 is defined using **TypeSpec** (formerly ADL),
located in the `ES3/schema` directory at the project root.

- **Source (.tsp)**: The `.tsp` files provide a concise, language-neutral way to
  define complex nested structures and relationships.
- **Generated Python**: The Pydantic models in this `earlysign/schema` package are
  automatically generated from these TypeSpec definitions.

Note: Developers should modify the `.tsp` files and use the appropriate emitters
to regenerate these Python modules, rather than editing the Python code directly.
"""
