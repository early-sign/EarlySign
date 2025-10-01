"""
Two-proportions scheme operator(s).

- WaldZStatistic  : Z for (pB - pA) with optional pooled variance
- ScoreZStatistic : Z for (pB - pA) using pooled variance under H0 (score test)
"""

from math import sqrt
from typing import Dict
from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.schemes.two_proportions.records import (
    BinomialCountsRecord,
    WaldZStatisticRecord,
    ScoreZStatisticRecord,
)


def _phat(n: int, m: int) -> float:
    """
    Compute m/n with input validation.

    Examples
    --------
    >>> round(_phat(4, 1), 3)
    0.25
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0 <= m <= n):
        raise ValueError("m must be in [0, n].")
    return float(m) / float(n)


def _wald_z(nA: int, mA: int, nB: int, mB: int, pooled: bool) -> float:
    """
    Wald Z for (pB - pA).

    Examples
    --------
    >>> round(_wald_z(100, 40, 100, 55, True), 3)
    2.124
    """
    pA = _phat(nA, mA)
    pB = _phat(nB, mB)
    diff = pB - pA
    if pooled:
        p_pool = (mA + mB) / float(nA + nB)
        var = p_pool * (1.0 - p_pool) * (1.0 / nA + 1.0 / nB)
    else:
        var = pA * (1.0 - pA) / nA + pB * (1.0 - pB) / nB
    if var <= 0.0:
        return float("inf") if diff > 0 else float("-inf") if diff < 0 else 0.0
    return diff / sqrt(var)


def _score_z(nA: int, mA: int, nB: int, mB: int) -> float:
    """
    Score test Z for (pB - pA), using pooled variance under H0.

    Examples
    --------
    >>> round(_score_z(100, 40, 100, 55), 3)
    2.124
    """
    return _wald_z(nA, mA, nB, mB, pooled=True)


class WaldZStatistic(LedgerOperator):
    """Compute Wald Z and insert one row."""

    counts: BinomialCountsRecord
    pooled: bool

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"wald": WaldZStatisticRecord(id=self.out_id)}  # type: ignore[attr-defined]

    def run(self) -> None:
        counts = self.counts
        out = self.outputs["wald"]
        pooled = bool(getattr(self, "pooled", True))

        cdf = (
            counts.latest()
            .execute().iloc[0]
        )
        nA, mA, nB, mB = map(int, cdf[["nA", "mA", "nB", "mB"]])
        z = _wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=pooled)

        payload = {"wald_z": float(z)}
        try:
            payload["look"] = int(cdf["look"])
        except Exception:
            pass
        out.insert(payload)


class ScoreZStatistic(LedgerOperator):
    """Compute score Z (pooled variance) and insert one row."""

    counts: BinomialCountsRecord
    out_id: str

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"score": ScoreZStatisticRecord(id=self.out_id)}

    def run(self) -> None:
        counts = self.counts
        out = self.outputs["score"]

        cdf = (
            counts.latest()
            .execute().iloc[0]
        )

        nA, mA, nB, mB = map(int, cdf[["nA", "mA", "nB", "mB"]])
        z = _score_z(nA=nA, mA=mA, nB=nB, mB=mB)

        payload = {"score_z": float(z)}
        try:
            payload["look"] = int(cdf["look"])
        except Exception:
            pass
        out.insert(payload)
