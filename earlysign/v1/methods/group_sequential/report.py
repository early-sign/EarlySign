from typing import Optional

import ibis
import numpy as np
from pydantic import BaseModel

from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import BinomialSummaryFact


class BinomialProgressReport(BaseModel):
    """Interim progress report for Binomial A/B tests."""

    look: Optional[int]
    n_c: int
    n_t: int
    z_stat: Optional[float]
    boundary: Optional[float]
    info_frac: float
    status: str


class BinomialFinalReport(BaseModel):
    """Comprehensive final summary report for Binomial A/B tests."""

    n_c: int
    n_t: int
    successes_c: int
    successes_t: int
    p_hat_c: float
    p_hat_t: float
    delta_hat: float
    z_stat: float
    is_rejected: bool
    final_status: str


class BinomialProgressProjector(Projector[BinomialProgressReport]):
    """
    Tier 2 Projector for interim monitoring.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialProgressReport]:
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.group_sequential.protocol import GSTProtocol

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(GSTProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Summary
        sc_traced = BinomialSummaryFact(
            identity="summary_c", filter_variant="C"
        ).project(table)
        st_traced = BinomialSummaryFact(
            identity="summary_t", filter_variant="T"
        ).project(table)
        sc, st = sc_traced.data, st_traced.data

        # 3. Calculate Operating Stats
        n_c, n_t = sc.n, st.n
        n_total = n_c + n_t
        info_frac = n_total / p.n_max if p.n_max > 0 else 0.0

        # Determine current look and boundary
        look_num = None
        boundary = None
        for i, m in enumerate(p.milestones):
            if info_frac >= m:
                look_num = i + 1
                boundary = p.boundaries[i]

        z_stat = None
        if n_c >= 2 and n_t >= 2:
            p_pool = (sc.successes + st.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z_stat = float((st.p_hat - sc.p_hat) / se) if se > 0 else 0.0

        status = "MONITORING"
        if look_num:
            if boundary and z_stat is not None and abs(z_stat) > boundary:
                status = "STOP_EFFICACY"
            elif look_num == len(p.milestones):
                status = "STOP_FINAL"

        report = BinomialProgressReport(
            look=look_num,
            n_c=n_c,
            n_t=n_t,
            z_stat=z_stat,
            boundary=boundary,
            info_frac=info_frac,
            status=status,
        )
        return ProjectionResult(
            data=report, trace=sc_traced.trace + st_traced.trace + protocol_traced.trace
        )


class BinomialFinalProjector(Projector[BinomialFinalReport]):
    """
    Tier 2 Projector for final study summary.
    """

    def __init__(self, is_rejected: bool, final_status: str):
        self.is_rejected = is_rejected
        self.final_status = final_status

    def project(self, table: ibis.Expr) -> ProjectionResult[BinomialFinalReport]:
        sc = BinomialSummaryFact(identity="summary_c", filter_variant="C").project(
            table
        )
        st = BinomialSummaryFact(identity="summary_t", filter_variant="T").project(
            table
        )

        n_c, n_t = sc.data.n, st.data.n
        z_stat = 0.0
        if n_c >= 2 and n_t >= 2:
            p_pool = (sc.data.successes + st.data.successes) / (n_c + n_t)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            z_stat = float((st.data.p_hat - sc.data.p_hat) / se) if se > 0 else 0.0

        report = BinomialFinalReport(
            n_c=n_c,
            n_t=n_t,
            successes_c=sc.data.successes,
            successes_t=st.data.successes,
            p_hat_c=sc.data.p_hat,
            p_hat_t=st.data.p_hat,
            delta_hat=st.data.p_hat - sc.data.p_hat,
            z_stat=z_stat,
            is_rejected=self.is_rejected,
            final_status=self.final_status,
        )
        return ProjectionResult(data=report, trace=sc.trace + st.trace)
