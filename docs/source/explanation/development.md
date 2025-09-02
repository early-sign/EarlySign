# Contributing to the Project

## Local setup

```sh
$ pip install poetry
$ poetry install --all-extras --with dev
```

## Build check
Always make sure to run the following to check if the tests and lint checks pass.
```sh
$ make check
```

If the lint fails, run the following to force the format.
```sh
$ make format
```
