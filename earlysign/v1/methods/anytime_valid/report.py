import ibis
from pydantic import BaseModel

from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import BinomialSummaryFact


class MonitoringProgressReport(BaseModel):
    """Interim progress report for continuous monitoring (E-process)."""

    n: int
    successes: int
    e_value: float
    threshold: float
    is_rejected: bool
    status: str


class MonitoringFinalReport(BaseModel):
    """Final summary report for continuous monitoring."""

    n: int
    successes: int
    p_hat: float
    e_value: float
    is_rejected: bool
    final_status: str


class MonitoringProgressProjector(Projector[MonitoringProgressReport]):
    """
    Tier 2 Projector for continuous monitoring.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[MonitoringProgressReport]:
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.anytime_valid.e_process import compute_binomial_e_value
        from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Summary
        traced_summary = BinomialSummaryFact(identity="monitoring_summary").project(table)
        s = traced_summary.data

        # 3. Compute e-value
        if p.alt_p is None:
            raise ValueError("EProcessProtocol must specify alt_p")

        res = compute_binomial_e_value(
            n=s.n,
            successes=s.successes,
            null_p=p.null_p,
            alt_p=p.alt_p,
            alpha=p.alpha,
        )

        status = "STOP_EVAL" if res.is_rejected else "MONITORING"

        report = MonitoringProgressReport(
            n=s.n,
            successes=s.successes,
            e_value=res.e_value,
            threshold=1.0 / p.alpha,
            is_rejected=res.is_rejected,
            status=status,
        )
        return ProjectionResult(
            data=report, trace=traced_summary.trace + protocol_traced.trace
        )


class MonitoringFinalProjector(Projector[MonitoringFinalReport]):
    """
    Tier 2 Projector for final monitoring status.
    """

    def __init__(self, e_value: float, is_rejected: bool, final_status: str):
        self.e_value = e_value
        self.is_rejected = is_rejected
        self.final_status = final_status

    def project(self, table: ibis.Expr) -> ProjectionResult[MonitoringFinalReport]:
        proj = BinomialSummaryFact(identity="monitoring_summary")
        traced_summary = proj.project(table)
        s = traced_summary.data

        report = MonitoringFinalReport(
            n=s.n,
            successes=s.successes,
            p_hat=s.p_hat,
            e_value=self.e_value,
            is_rejected=self.is_rejected,
            final_status=self.final_status,
        )
        return ProjectionResult(data=report, trace=traced_summary.trace)
