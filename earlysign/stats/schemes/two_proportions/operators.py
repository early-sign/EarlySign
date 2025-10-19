"""
Two-proportions scheme operator(s).

- WaldZStatistic  : Z for (pB - pA) with optional pooled variance
- ScoreZStatistic : Z for (pB - pA) using pooled variance under H0 (score test)
"""

from typing import Dict, Union

from earlysign.framework.operator import LedgerOperator
from earlysign.framework.records import LedgerRecord
from earlysign.stats.common.two_proportions import compute_wald_z
from earlysign.stats.schemes.two_proportions.records import (
    BinomialCountsRecord,
    BinomialCountsSnapshotRecord,
    ScoreZStatisticRecord,
    WaldZStatisticRecord,
)


class BinomialCountsSnapshot(LedgerOperator):
    """Compute cumulative snapshot from incremental observations.

    Reads the latest incremental observation from BinomialCountsRecord,
    adds it to the previous snapshot (or initializes if first observation),
    and writes the new cumulative snapshot to BinomialCountsSnapshotRecord.
    """

    obs: BinomialCountsRecord
    out_id: str

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"snapshot": BinomialCountsSnapshotRecord(id=self.out_id)}

    def run(self) -> None:
        obs_rec = self.obs
        snapshot_rec: BinomialCountsSnapshotRecord = self.outputs["snapshot"]  # type: ignore

        # Read the latest incremental observation
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

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"wald": WaldZStatisticRecord(id=self.out_id)}

    def run(self) -> None:
        cum_counts = self.cum_counts
        out = self.outputs["wald"]
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

    def derived_records(self) -> Dict[str, LedgerRecord]:
        return {"score": ScoreZStatisticRecord(id=self.out_id)}

    def run(self) -> None:
        cum_counts = self.cum_counts
        out = self.outputs["score"]

        cdf = cum_counts.latest().execute().iloc[0]

        nA, mA, nB, mB = map(int, cdf[["nA", "mA", "nB", "mB"]])
        # Score Z is equivalent to Wald Z with pooled variance
        z = compute_wald_z(nA=nA, mA=mA, nB=nB, mB=mB, pooled=True)

        payload = {"score_z": float(z)}
        out.insert(payload)
