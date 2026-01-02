from typing import Optional

import ibis
import numpy as np
import pandas as pd
from pydantic import BaseModel

from earlysign.v1.framework.intermediate_fact import IntermediateFact, Snapshot
from earlysign.v1.framework.projector import ProjectionResult, Projector
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
    variant: str


class BinomialSummaryFact(IntermediateFact[BinomialSummary]):
    """
    Incremental Projector for Binomial data.
    Implements IntermediateFact to optimize state reconstruction.
    """

    data_type = BinomialSummary

    def __init__(self, identity: str, filter_variant: Optional[str] = None):
        super().__init__(identity)
        self.filter_variant = filter_variant

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

        if self.filter_variant:
            variant_val = self.filter_variant
            obs_table = obs_table.filter(
                obs_table.payload["variant"].cast("string").re_replace('^"|"$', "")
                == variant_val
            )
            batch_table = batch_table.filter(
                batch_table.payload["variant"].cast("string").re_replace('^"|"$', "")
                == variant_val
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


class BinomialZResult(BaseModel):
    """Result of a two-proportion Z-test (Tier 2 realise)."""

    z_stat: float
    is_rejected: bool
    boundary: float


class BinomialZProjector(Projector[BinomialZResult]):
    """
    Tier 2 Projector: Pure statistical inference.
    Consumes results from Tier 1 Intermediate Facts.
    """

    def __init__(self, boundary: float):
        self.boundary = boundary

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialZResult]:
        # Tier 1: State Reconstruction
        ctrl_traced = BinomialSummaryFact(
            identity="summary_c", filter_variant="C"
        ).project(table)
        tret_traced = BinomialSummaryFact(
            identity="summary_t", filter_variant="T"
        ).project(table)

        sc, st = ctrl_traced.data, tret_traced.data
        n_c, n_t = sc.n, st.n

        if n_c < 2 or n_t < 2:  # Minimum requirements for Z-test
            res = BinomialZResult(z_stat=0.0, is_rejected=False, boundary=self.boundary)
        else:
            p_pool = (sc.successes + st.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z = (st.p_hat - sc.p_hat) / se if se > 0 else 0.0
            res = BinomialZResult(
                z_stat=float(z),
                is_rejected=abs(z) > self.boundary,
                boundary=self.boundary,
            )

        return ProjectionResult(data=res, trace=ctrl_traced.trace + tret_traced.trace)
