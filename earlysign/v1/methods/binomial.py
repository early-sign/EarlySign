from typing import Optional

import ibis
import pandas as pd
from pydantic import BaseModel

from earlysign.v1.framework.intermediate_fact import IntermediateFact, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceHash


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

        # 2. Extract incremental stats via Ibis
        agg_obs = obs_table.aggregate(
            n=obs_table.count(),
            successes=obs_table.payload["success"].cast("int").sum(),
        )

        agg_batch = batch_table.aggregate(
            n=batch_table.payload["n"].cast("int").sum(),
            successes=batch_table.payload["success"].cast("int").sum(),
        )

        res_obs = agg_obs.execute()
        n_obs_raw = res_obs["n"].iloc[0]
        n_obs = int(n_obs_raw) if not pd.isna(n_obs_raw) else 0
        s_obs_raw = res_obs["successes"].iloc[0]
        s_obs = int(s_obs_raw) if not pd.isna(s_obs_raw) else 0

        res_batch = agg_batch.execute()
        n_batch_raw = res_batch["n"].iloc[0]
        n_batch = int(n_batch_raw) if not pd.isna(n_batch_raw) else 0
        s_batch_raw = res_batch["successes"].iloc[0]
        s_batch = int(s_batch_raw) if not pd.isna(s_batch_raw) else 0

        # 3. Incremental Folding (Snapshot + Delta)
        # The Snapshot record holds the data as T (BinomialSummary)
        n_snap = snapshot.data.n if snapshot else 0
        s_snap = snapshot.data.successes if snapshot else 0

        n_total = n_snap + n_obs + n_batch
        s_total = s_snap + s_obs + s_batch
        p_hat = s_total / n_total if n_total > 0 else 0.0

        summary = BinomialSummary(n=n_total, successes=s_total, p_hat=p_hat)

        # 4. Lineage Management
        try:
            h_obs = obs_table.labels["trace_hash"].execute().tolist()
            h_batch = batch_table.labels["trace_hash"].execute().tolist()
            trace = [TraceHash(h) for h in h_obs + h_batch]
            # If we had the snapshot's trace, we'd prepend it here.
        except Exception:
            trace = []

        return ProjectionResult(data=summary, trace=trace)
