"""
Gates for two-proportions scheme.

Currently provides:
- GateMinTotalSamples : open the gate only if (nA + nB) >= min_total
"""

from dataclasses import dataclass
from typing import Dict

from earlysign.framework.gate import GateDecisionRecord
from earlysign.framework.operator import LedgerOperator, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.schemes.two_proportions.records import BinomialCountsRecord


class GateMinTotalSamples(LedgerOperator):
    """
    Open the gate only when total sample size is large enough.

    __init__ parameters
    -------------------
    counts    : BinomialCountsRecord   # attached input
    out_id    : str
    min_total : int

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.stats.schemes.two_proportions.records import BinomialCountsRecord
    >>> scoped = Ledger(ibis.duckdb.connect(":memory:"), "table").bind(experiment_id="exp1")
    >>> scoped.ensure()
    >>> counts_rec = BinomialCountsRecord(id="counts1").attach(scoped)
    >>> _ = counts_rec.insert({"nA": 80, "mA": 30, "nB": 70, "mB": 25})
    >>> gate = GateMinTotalSamples(scoped, counts=counts_rec, out_id="gate1", min_total=200)
    >>> gate.run()
    """

    out_id: str
    counts: BinomialCountsRecord
    min_total: int

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        gate: GateDecisionRecord

    outputs: Outputs

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"gate": GateDecisionRecord(id=self.out_id)}

    def run(self) -> None:
        counts = self.counts
        out = self.outputs.gate
        min_total = int(getattr(self, "min_total"))

        cdf = (
            counts.latest()
            .select(
                nA=counts.t.payload["nA"].cast("int64"),
                nB=counts.t.payload["nB"].cast("int64"),
            )
            .execute()
        )
        if len(cdf) == 0:
            return
        n_total = int(cdf.iloc[0]["nA"]) + int(cdf.iloc[0]["nB"])
        if n_total >= min_total:
            out.insert(
                {
                    "gate": "open",
                    "reason": "ok",
                    "n_total": n_total,
                    "min_total": min_total,
                }
            )
        else:
            out.insert(
                {
                    "gate": "closed",
                    "reason": "insufficient_samples",
                    "n_total": n_total,
                    "min_total": min_total,
                }
            )
