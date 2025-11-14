"""
Operators for the "one-mean (Gaussian)" scheme.
"""

import math
from dataclasses import dataclass
from typing import Dict

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.schemes.one_mean.records import (
    OneMeanSummaryRecord,
    ZMeanKnownVarRecord,
)


def compute_z_mean_known_var(n: int, mean: float, sigma2: float) -> float:
    """
    Z for testing mu=0 with known variance sigma^2 (two-sided by default upstream).

    Z = mean * sqrt(n) / sqrt(sigma^2)

    Examples
    --------
    >>> round(compute_z_mean_known_var(100, 0.2, 1.0), 3)
    2.0
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma2 <= 0.0:
        raise ValueError("sigma^2 must be positive.")
    return float(mean) * math.sqrt(float(n) / float(sigma2))


class ZMeanKnownVar(LedgerOp):
    """
    Read OneMeanSummaryRecord (n, mean) and insert a ZKnownVar row.

    __init__ parameters
    -------------------
    summary : OneMeanSummaryRecord   # attached input
    out_id  : str
    sigma2  : float                   # known variance
    """

    out_id: str
    summary: OneMeanSummaryRecord

    def __init__(
        self,
        ledger: Ledger,
        *,
        summary: OneMeanSummaryRecord,
        out_id: str,
        sigma2: float,
    ):
        super().__init__(ledger, summary=summary, out_id=out_id, sigma2=sigma2)

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        z: ZMeanKnownVarRecord

    outputs: Outputs

    def build_outputs(self) -> Dict[str, LedgerRecord]:
        return {"z": ZMeanKnownVarRecord(name=self.out_id, ledger=self.ledger)}

    def run(self) -> None:
        out = self.outputs.z
        summary: OneMeanSummaryRecord = self.summary
        sigma2 = float(getattr(self, "sigma2"))

        sdf = (
            summary.latest()
            .select(
                n=summary.t.payload["n"].cast("int64"),
                mean=summary.t.payload["mean"].cast("float64"),
                look=summary.t.payload["look"].cast("int64"),
            )
            .execute()
        )
        if len(sdf) == 0:
            return

        n = int(sdf.iloc[0]["n"])
        mean = float(sdf.iloc[0]["mean"])
        z = compute_z_mean_known_var(n=n, mean=mean, sigma2=sigma2)

        payload = {"z": float(z)}
        try:
            payload["look"] = int(sdf.iloc[0]["look"])
        except Exception:
            pass
        out.insert(payload)
