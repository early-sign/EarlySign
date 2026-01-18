import ibis
from pydantic import BaseModel

from earlysign.schema.ES3.Binomial import ArmMetrics
from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import Scoreboard


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
        from earlysign.v1.methods.anytime_valid.e_process import (
            compute_binomial_e_value,
        )
        from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Metrics
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        n_total = sum(a.metrics.n for a in metrics.arms.values())
        s_total = sum(a.metrics.successes for a in metrics.arms.values())
        p_total = s_total / n_total if n_total > 0 else 0.0
        s = ArmMetrics(n=n_total, successes=s_total, p_hat=p_total)

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
            data=report, trace=metrics_traced.trace + protocol_traced.trace
        )


class MonitoringFinalProjector(Projector[MonitoringFinalReport]):
    """
    Tier 2 Projector for final monitoring status.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[MonitoringFinalReport]:
        from earlysign.v1.framework.projector import ProtocolProjector
        from earlysign.v1.methods.anytime_valid.e_process import (
            compute_binomial_e_value,
        )
        from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Metrics
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        n_total = sum(a.metrics.n for a in metrics.arms.values())
        s_total = sum(a.metrics.successes for a in metrics.arms.values())
        p_total = s_total / n_total if n_total > 0 else 0.0
        s = ArmMetrics(n=n_total, successes=s_total, p_hat=p_total)

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

        # 4. Determine Final Status
        # For E-Process, statistical rejection is the primary Stopping Condition.
        final_status = "COMPLETED"
        if res.is_rejected:
            final_status = "STOP_EVAL"

        report = MonitoringFinalReport(
            n=s.n,
            successes=s.successes,
            p_hat=s.p_hat,
            e_value=res.e_value,
            is_rejected=res.is_rejected,
            final_status=final_status,
        )
        return ProjectionResult(
            data=report, trace=metrics_traced.trace + protocol_traced.trace
        )
