"""
earlysign.backends.polars.io
===========================

Pluggable persistence for the Polars-backed ledger via **sinks/sources**.

- Parquet (file/dir), CSV file
- Database (ADBC) sink and query source

This module deliberately contains no ledger semantics—just I/O.

Doctest (smoke):
>>> import polars as pl
>>> from earlysign.backends.polars.io import ParquetFileSink, ParquetFileSource
>>> df = pl.DataFrame({"x":[1,2,3]})
>>> ParquetFileSink("_tmp.parquet").write(df)  # doctest: +SKIP
>>> _ = ParquetFileSource("_tmp.parquet").read()  # doctest: +SKIP
"""

from __future__ import annotations
import os
from typing import Protocol

import polars as pl


class LedgerSink(Protocol):
    """A write-only sink: DataFrame -> storage."""
    def write(self, df: pl.DataFrame) -> None: ...


class LedgerSource(Protocol):
    """A read-only source: storage -> DataFrame."""
    def read(self) -> pl.DataFrame: ...


class ParquetFileSink:
    def __init__(self, path: str) -> None:
        self.path = path
    def write(self, df: pl.DataFrame) -> None:
        df.write_parquet(self.path)


class ParquetDirSink:
    def __init__(self, dirpath: str, filename: str = "records.parquet") -> None:
        self.dirpath = dirpath
        self.filename = filename
    def write(self, df: pl.DataFrame) -> None:
        os.makedirs(self.dirpath, exist_ok=True)
        df.write_parquet(os.path.join(self.dirpath, self.filename))


class CsvFileSink:
    def __init__(self, path: str) -> None:
        self.path = path
    def write(self, df: pl.DataFrame) -> None:
        df.write_csv(self.path)


class DatabaseTableSink:
    """ADBC-backed sink (BigQuery/Postgres/SQLite…). Requires appropriate driver."""
    def __init__(self, conn: str, table: str, if_exists: str = "append", batch_size: int = 5000) -> None:
        self.conn = conn
        self.table = table
        self.if_exists = if_exists
        self.batch_size = batch_size
    def write(self, df: pl.DataFrame) -> None:
        if not hasattr(pl.DataFrame, "write_database"):
            raise RuntimeError("Polars write_database not available; upgrade Polars & install an ADBC driver.")
        df.write_database(
            table_name=self.table,
            connection=self.conn,
            if_table_exists=self.if_exists,
            engine="adbc",
            batch_size=self.batch_size,
        )


class ParquetFileSource:
    def __init__(self, path: str) -> None:
        self.path = path
    def read(self) -> pl.DataFrame:
        return pl.read_parquet(self.path)


class DatabaseQuerySource:
    """ADBC-backed query source."""
    def __init__(self, conn: str, query: str) -> None:
        self.conn = conn
        self.query = query
    def read(self) -> pl.DataFrame:
        if not hasattr(pl, "read_database"):
            raise RuntimeError("Polars read_database not available; upgrade Polars & install an ADBC driver.")
        return pl.read_database(self.query, connection=self.conn, engine="adbc")
