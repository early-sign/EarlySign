"""
earlysign.schemes.two_proportions.reduce
========================================

Scheme-specific reducers for two-proportions.

- `reduce_counts(ledger, experiment_id)`: accumulate nA, nB, mA, mB
  from the `obs` namespace.

Doctest (smoke):
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> L = PolarsLedger()
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":5,"nB":6,"mA":1,"mB":0})
>>> reduce_counts(L, experiment_id="exp#1")
(5, 6, 1, 0)
"""

from __future__ import annotations
from typing import Tuple

from earlysign.core.ledger import Ledger
from earlysign.core.names import Namespace


def reduce_counts(ledger: Ledger, *, experiment_id: str) -> Tuple[int, int, int, int]:
    """Aggregate counts from all `obs` records for the experiment."""
    nA = nB = mA = mB = 0
    for row in ledger.reader().iter_rows(namespace=Namespace.OBS, entity=experiment_id):
        p = row.payload
        if {"nA", "nB", "mA", "mB"} <= set(p.keys()):
            nA += int(p["nA"])
            nB += int(p["nB"])
            mA += int(p["mA"])
            mB += int(p["mB"])
    return nA, nB, mA, mB
