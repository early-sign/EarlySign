"""
earlysign.reporting.generic
===========================

A *scheme-agnostic* reporter that simply shows raw ledger events and
namespace×kind counts. Scheme-specific visualization lives elsewhere.

Doctest (smoke):
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> from earlysign.reporting.generic import LedgerReporter
>>> L = PolarsLedger()
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="E", step_key="s1",
...               payload_type="Obs", payload={"x":1})
>>> rep = LedgerReporter(L.frame())
>>> tbl = rep.ledger_table(); "namespace" in tbl.columns
True
"""

from __future__ import annotations
from dataclasses import dataclass

import polars as pl


@dataclass
class LedgerReporter:
    """Generic reporter for raw ledger frames."""
    df: pl.DataFrame

    def ledger_table(self, max_rows: int = 60) -> pl.DataFrame:
        df = self.df.sort("ts")
        if df.height <= max_rows:
            sample = df
        else:
            head_n = max_rows // 2
            sample = pl.concat([df.head(head_n), df.tail(max_rows - head_n)], how="vertical")
        return sample

    def counts(self) -> pl.DataFrame:
        return (
            self.df.group_by(["namespace", "kind"])
            .len()
            .rename({"len": "count"})
            .sort(["namespace", "kind"])
        )
