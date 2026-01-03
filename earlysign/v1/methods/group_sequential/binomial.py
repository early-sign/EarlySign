from typing import Optional

import ibis
import numpy as np
from pydantic import BaseModel

from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import BinomialSummaryFact


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


