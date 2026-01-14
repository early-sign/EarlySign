from typing import Optional

import ibis
from pydantic import BaseModel

from earlysign.v1.framework.intermediate_fact import IntermediateFact, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId


class BinomialSummary(BaseModel):
    """Standard Bernoulli/Binomial summary statistics."""

    n: int
    successes: int
    p_hat: float


class BatchObservation(BaseModel):
    """Raw evidence: a batch of Bernoulli trials."""

    n: int
    success: int
    arm: str


class BinomialSummaryFact(IntermediateFact[BinomialSummary]):
    """
    Incremental Projector for Binomial data.
    Implements IntermediateFact to optimize state reconstruction.
    """

    data_type = BinomialSummary

    def __init__(self, identity: str, filter_arm: Optional[str] = None):
        super().__init__(identity)
        self.filter_arm = filter_arm

    def compute(
        self,
        snapshot: Optional[Snapshot[BinomialSummary]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[BinomialSummary]:
        """
        Projects raw delta onto a Snapshot.
        """
        # 1. Prepare delta aggregation
        obs_table = delta_expr.filter(delta_expr.payload_type == "Observation")
        batch_table = delta_expr.filter(delta_expr.payload_type == "BatchObservation")

        if self.filter_arm:
            arm_val = self.filter_arm
            obs_table = obs_table.filter(
                obs_table.payload["arm"].cast("string").re_replace('^"|"$', "")
                == arm_val
            )
            batch_table = batch_table.filter(
                batch_table.payload["arm"].cast("string").re_replace('^"|"$', "")
                == arm_val
            )

        # 2. Execute once to get both data and uuids
        obs_df = obs_table.select(
            "uuid",
            success=obs_table.payload["success"].cast("int"),
        ).execute()

        batch_df = batch_table.select(
            "uuid",
            n=batch_table.payload["n"].cast("int"),
            success=batch_table.payload["success"].cast("int"),
        ).execute()

        # 3. Aggregate from dataframes
        n_obs = len(obs_df)
        s_obs = int(obs_df["success"].sum()) if n_obs > 0 else 0

        n_batch = int(batch_df["n"].sum()) if len(batch_df) > 0 else 0
        s_batch = int(batch_df["success"].sum()) if len(batch_df) > 0 else 0

        # 4. Incremental Folding (Snapshot + Delta)
        n_snap = snapshot.data.n if snapshot else 0
        s_snap = snapshot.data.successes if snapshot else 0

        n_total = n_snap + n_obs + n_batch
        s_total = s_snap + s_obs + s_batch
        p_hat = s_total / n_total if n_total > 0 else 0.0

        summary = BinomialSummary(n=n_total, successes=s_total, p_hat=p_hat)

        # 5. Lineage Management - uuids already fetched above
        trace: list[TraceId] = []
        if snapshot:
            snapshot_uuid = getattr(snapshot, "uuid", None)
            if snapshot_uuid:
                trace.append(TraceId(str(snapshot_uuid)))

        # Collect delta uuids from already-executed dataframes
        trace.extend([TraceId(str(uid)) for uid in obs_df["uuid"].tolist()])
        trace.extend([TraceId(str(uid)) for uid in batch_df["uuid"].tolist()])

        return ProjectionResult(data=summary, trace=trace)
