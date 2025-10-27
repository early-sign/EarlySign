"""
Two-proportions scheme operator(s).

- WaldZStatistic  : Z for (pB - pA) with optional pooled variance
- ScoreZStatistic : Z for (pB - pA) using pooled variance under H0 (score test)
"""

from dataclasses import dataclass
from typing import Union

from earlysign.core.ledger import Ledger
from earlysign.framework.operator import LedgerOperator, LedgerOpOutputs
from earlysign.framework.records import LedgerRecord
from earlysign.stats.applications.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.stats.applications.execution.methods.group_sequential.records.statistics import (
    WaldZStatisticRecord,
)
from earlysign.stats.applications.execution.schemes.two_proportions.records import (
    BinomialCountsRecord,
    BinomialCountsSnapshotRecord,
    ScoreZStatisticRecord,
)
from earlysign.stats.essentials.methods.group_sequential.info_time import (
    info_time_from_sample_size,
)
from earlysign.stats.essentials.schemes.two_proportions.wald_z import compute_wald_z

__all__ = [
    "InformationTime",
    "BinomialCountsSnapshot",
    "WaldZStatistic",
    "ScoreZStatistic",
]


class InformationTime(LedgerOperator):
    """
    Insert information-time record from sample counts.

    Parameters
    ----------
    scoped : Ledger
        Scoped ledger instance.
    out_id : str
        ID of InformationTimeRecord to create.
    cum_counts : BinomialCountsRecord | BinomialCountsSnapshotRecord (attached)
        Record containing cumulative binomial counts.
    planned_max_n : int
        Maximum total sample size.
    """

    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        info: InformationTimeRecord

    outputs: Outputs

    def __init__(
        self,
        scoped: Ledger,
        *,
        out_id: str,
        cum_counts: Union[BinomialCountsRecord, BinomialCountsSnapshotRecord],
        planned_max_n: int,
    ):
        super().__init__(
            scoped,
            out_id=out_id,
            cum_counts=cum_counts,
            planned_max_n=planned_max_n,
        )

    def derived_records(self) -> dict[str, LedgerRecord]:
        return {"info": InformationTimeRecord(id=self.out_id)}

    def run(self) -> None:
        out = self.outputs.info
        cum_counts = getattr(self, "cum_counts")
        planned_max_n = getattr(self, "planned_max_n")

        cdf = cum_counts.latest().execute()
        if len(cdf) == 0:
            raise ValueError("No counts data available")

        latest = cdf.iloc[0]
        n_total = int(latest["nA"]) + int(latest["nB"])

        t = info_time_from_sample_size(n_current=n_total, n_max=planned_max_n)
        out.insert({"info_time": float(t)})


class BinomialCountsSnapshot(LedgerOperator):
    """Compute cumulative snapshot from incremental observations.

    Reads the latest incremental observation from BinomialCountsRecord,
    adds it to the previous snapshot (or initializes if first observation),
    and writes the new cumulative snapshot to BinomialCountsSnapshotRecord.
    """

    obs: BinomialCountsRecord
    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        snapshot: BinomialCountsSnapshotRecord

    # Type annotation for outputs - enables type inference!
    outputs: Outputs

    def derived_records(self) -> dict[str, LedgerRecord]:
        return {"snapshot": BinomialCountsSnapshotRecord(id=self.out_id)}

    def run(self) -> None:
        obs_rec = self.obs
        snapshot_rec = self.outputs.snapshot
        latest_obs = obs_rec.latest().execute().iloc[0]
        delta_nA = int(latest_obs["nA"])
        delta_mA = int(latest_obs["mA"])
        delta_nB = int(latest_obs["nB"])
        delta_mB = int(latest_obs["mB"])

        # Read previous snapshot (or initialize to zero)
        try:
            prev_snapshot = snapshot_rec.latest().execute().iloc[0]
            prev_nA = int(prev_snapshot["nA"])
            prev_mA = int(prev_snapshot["mA"])
            prev_nB = int(prev_snapshot["nB"])
            prev_mB = int(prev_snapshot["mB"])
        except (IndexError, KeyError):
            # First observation - no previous snapshot
            prev_nA = prev_mA = prev_nB = prev_mB = 0

        # Compute cumulative snapshot
        new_nA = prev_nA + delta_nA
        new_mA = prev_mA + delta_mA
        new_nB = prev_nB + delta_nB
        new_mB = prev_mB + delta_mB

        # Insert new snapshot
        snapshot_rec.insert(nA=new_nA, mA=new_mA, nB=new_nB, mB=new_mB)


class WaldZStatistic(LedgerOperator):
    """Compute Wald Z and insert one row.

    Accepts either BinomialCountsRecord or BinomialCountsSnapshotRecord.
    Both have the same schema structure (nA, mA, nB, mB).
    """

    cum_counts: Union[BinomialCountsRecord, BinomialCountsSnapshotRecord]
    pooled: bool
    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        wald: WaldZStatisticRecord

    # Type annotation for outputs - enables type inference!
    outputs: Outputs

    def derived_records(self) -> dict[str, LedgerRecord]:
        return {"wald": WaldZStatisticRecord(id=self.out_id)}

    def run(self) -> None:
        cum_counts = self.cum_counts
        out = self.outputs.wald
        pooled = bool(getattr(self, "pooled", True))

        cdf = cum_counts.latest().execute().iloc[0]
        nA, mA, nB, mB = map(int, cdf[["nA", "mA", "nB", "mB"]])
        z = compute_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=pooled)

        payload = {"wald_z": float(z)}
        out.insert(payload)


class ScoreZStatistic(LedgerOperator):
    """Compute score Z (pooled variance) and insert one row.

    Accepts either BinomialCountsRecord or BinomialCountsSnapshotRecord.
    Both have the same schema structure (nA, mA, nB, mB).
    """

    cum_counts: Union[BinomialCountsRecord, BinomialCountsSnapshotRecord]
    out_id: str

    @dataclass(frozen=True)
    class Outputs(LedgerOpOutputs):
        """Outputs for this operator."""

        score: ScoreZStatisticRecord

    # Type annotation for outputs - enables type inference!
    outputs: Outputs

    def derived_records(self) -> dict[str, LedgerRecord]:
        return {"score": ScoreZStatisticRecord(id=self.out_id)}

    def run(self) -> None:
        cum_counts = self.cum_counts
        out = self.outputs.score

        cdf = cum_counts.latest().execute().iloc[0]

        nA, mA, nB, mB = map(int, cdf[["nA", "mA", "nB", "mB"]])
        # Score Z is equivalent to Wald Z with pooled variance
        z = compute_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=True)

        payload = {"score_z": float(z)}
        out.insert(payload)
