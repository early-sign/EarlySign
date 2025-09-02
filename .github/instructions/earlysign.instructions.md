---
applyTo: "earlysign/**"
---
## Docs and Tests
- Always update the documentation in the docstring as well as in docs/ where they apply. These documents are deployed and should always be up-to-date.
- Use doctests to write the primary examples of the implementations.
- Always prefer to use doctests over standalone tests, except under special circumstances.

## Coding standard
- Never use relative imports; always use absolute package imports based on the poetry's functionalities.
- The `__init__.py` files are meant for documentation generation.
  - DO NOT USE `__init__.py` to restrict the access to the namespaced functionalities. We use absolute imports with poetry, so we should always be able to access the components directly.
  - USE `__init__.py` to provide a namespace-level docstring: this can include doctests; see examples from other directories.
