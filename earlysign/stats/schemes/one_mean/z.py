"""Z-statistic operator for one-mean with known variance."""

from dataclasses import dataclass
from typing import Dict

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOp, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.schemes.one_mean.records import (
    OneMeanSummaryRecord,
    ZMeanKnownVarRecord,
)


def compute_z_mean_known_var(*, n: int, mean: float, sigma2: float) -> float:
    """
    Compute Z for a one-mean test with known variance.

    Parameters
    ----------
    n : int
        Sample size.
    mean : float
        Sample mean.
    sigma2 : float
        Known variance.
    """

    if n <= 0:
        raise ValueError("n must be positive.")
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be positive.")
    return float((mean * (n**0.5)) / (sigma2**0.5))


class ZMeanKnownVar(LedgerOp):
    """
    Read OneMeanSummaryRecord (n, mean) and insert a ZKnownVar row.

    __init__ parameters
    -------------------
    summary : OneMeanSummaryRecord   # attached input
    out_id  : str
    sigma2  : float                   # known variance

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events"); _ = ledger.ensure(); ledger = ledger.bind(experiment_id="demo")
    >>> summary = OneMeanSummaryRecord("s").attach(ledger)
    >>> _ = summary.insert({"n": 20, "mean": 0.1, "look": 1})
    >>> _ = ZMeanKnownVar(ledger, summary=summary, out_id="z1", sigma2=1.0).run()
    >>> zrec = ZMeanKnownVarRecord("z1").attach(ledger)
    >>> len(zrec.latest().execute()) == 1
    True
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
