"""
earlysign.reporting.generic
===========================

A scheme-agnostic reporter that shows raw ledger events and
namespace x kind counts. Works with ibis-based ledgers.

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.core.names import Namespace
>>> from earlysign.reporting.generic import LedgerReporter
>>> conn = ibis.duckdb.connect(":memory:")
>>> L = Ledger(conn, "test")
>>> rep = LedgerReporter(L.raw_table.to_pandas())
>>> isinstance(rep.ledger_table(), type(L.raw_table.to_pandas()))
True
"""

from __future__ import annotations

from typing import Optional

import warnings

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("pandas is required for reporting functionality") from e


class LedgerReporter:
    """
    A generic, scheme-agnostic reporter for any experiment ledger.
    Works with pandas DataFrames for compatibility across different backends.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("DataFrame must be pandas.DataFrame")
        self.df = df

    def ledger_table(self) -> pd.DataFrame:
        """Return the underlying ledger DataFrame."""
        return self.df

    def unique_entities(self) -> list[str]:
        """List all unique experiment entities."""
        if "entity" not in self.df.columns:
            return []
        return sorted(self.df["entity"].dropna().unique().tolist())

    def unique_namespaces(self) -> list[str]:
        """List all unique event namespaces."""
        if "namespace" not in self.df.columns:
            return []
        return sorted(self.df["namespace"].dropna().unique().tolist())

    def unique_kinds(self) -> list[str]:
        """List all unique event kinds."""
        if "kind" not in self.df.columns:
            return []
        return sorted(self.df["kind"].dropna().unique().tolist())

    def namespace_kind_counts(self) -> pd.DataFrame:
        """
        Return a pivot table of namespace x kind with event counts.

        Returns
        -------
        pd.DataFrame
            Rows = namespaces, Columns = kinds, Values = event counts
        """
        if (
            self.df.empty
            or "namespace" not in self.df.columns
            or "kind" not in self.df.columns
        ):
            return pd.DataFrame()

        counts = self.df.groupby(["namespace", "kind"]).size().reset_index(name="count")

        if counts.empty:
            return pd.DataFrame()

        pivot = counts.pivot(index="namespace", columns="kind", values="count").fillna(
            0
        )
        return pivot.astype(int)
