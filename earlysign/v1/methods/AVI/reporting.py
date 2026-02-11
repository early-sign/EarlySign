from typing import Any, Dict, Optional, Union

import ibis
import pandas as pd
from pydantic import BaseModel

from earlysign.schema.ES3.AVI import Protocol
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboardSchema
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboardSchema
from earlysign.v1.framework.projector import (
    ProjectionResult,
    Projector,
    ProtocolProjector,
)
from earlysign.v1.framework.trace import TraceId
from earlysign.v1.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.v1.methods.continuous import Scoreboard as ContinuousScoreboard


class ProgressReport(BaseModel):
    """AVI interim progress report."""

    sample_n: int
    trajectory: float
    boundary: Optional[float]
    status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class FinalReport(BaseModel):
    """AVI final summary report."""

    sample_n: int
    trajectory: float
    is_rejected: bool
    final_status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class ProgressProjector(Projector[ProgressReport]):
    """
    Projector for AVI interim monitoring.
    Reads the latest LookResult and Scoreboard from the ledger.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[ProgressReport]:
        # 1. Read the latest LookResult
        results_df = table.filter(table.type == "LookResult").execute()

        if results_df.empty:
            # Fallback or empty report
            return ProjectionResult(
                data=ProgressReport(
                    sample_n=0,
                    trajectory=0.0,
                    boundary=None,
                    status=DecisionStatus.CONTINUE_,
                ),
                trace=[],
            )

        # Get latest by index
        latest_row = results_df.iloc[-1]
        payload = latest_row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        latest_look = LookResult.model_validate(payload)

        # 2. Determine response type from Protocol
        protocol_traced = ProtocolProjector(Protocol).project(table)
        response_type = getattr(protocol_traced.data.task, "response_type", "binary")

        # 3. Read Scoreboard for arm metrics
        metrics_traced: Union[
            ProjectionResult[BinomialScoreboardSchema],
            ProjectionResult[ContinuousScoreboardSchema],
        ]
        if response_type == "binary":
            metrics_traced = BinomialScoreboard(identity="metrics").project(table)
        else:
            metrics_traced = ContinuousScoreboard(identity="metrics").project(table)

        metrics: ContinuousScoreboardSchema | BinomialScoreboardSchema = (
            metrics_traced.data
        )

        report = ProgressReport(
            sample_n=latest_look.sample_n,
            trajectory=latest_look.trajectory,
            boundary=latest_look.boundary,
            status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class FinalProjector(Projector[FinalReport]):
    """
    Projector for AVI final study summary.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[FinalReport]:
        # 1. Read the latest LookResult
        results_df = table.filter(table.type == "LookResult").execute()

        if results_df.empty:
            raise RuntimeError("No LookResult found in ledger for final report.")

        latest_row = results_df.iloc[-1]
        payload = latest_row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        latest_look = LookResult.model_validate(payload)

        # 2. Determine response type from Protocol
        protocol_traced = ProtocolProjector(Protocol).project(table)
        response_type = getattr(protocol_traced.data.task, "response_type", "binary")

        # 3. Read Scoreboard
        metrics_traced: Union[
            ProjectionResult[BinomialScoreboardSchema],
            ProjectionResult[ContinuousScoreboardSchema],
        ]
        if response_type == "binary":
            metrics_traced = BinomialScoreboard(identity="metrics").project(table)
        else:
            metrics_traced = ContinuousScoreboard(identity="metrics").project(table)

        metrics: Union[
            BinomialScoreboardSchema,
            ContinuousScoreboardSchema,
        ] = metrics_traced.data

        report = FinalReport(
            sample_n=latest_look.sample_n,
            trajectory=latest_look.trajectory,
            is_rejected=latest_look.is_crossed,
            final_status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class BinomialEValueProgressProjector(Projector[ProgressReport]):
    """
    Projector for 1-sample binomial e-value monitoring.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[ProgressReport]:
        from earlysign.schema.ES3.Binomial import ArmMetrics
        from earlysign.v1.methods.AVI.engines.binomial_e_value import (
            EProcessProtocol,
            compute_binomial_e_value,
        )

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Metrics
        metrics_traced = BinomialScoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        n_total = sum(a.metrics.n for a in metrics.arms.values())
        s_total = sum(a.metrics.successes for a in metrics.arms.values())
        p_total = s_total / n_total if n_total > 0 else 0.0
        s = ArmMetrics(n=n_total, successes=s_total, p_hat=p_total)

        # 3. Compute e-value
        res = compute_binomial_e_value(
            n=s.n,
            successes=s.successes,
            null_p=p.null_p,
            alt_p=p.alt_p,
            alpha=p.alpha,
        )

        status = DecisionStatus.CONTINUE_
        if res.is_rejected:
            status = DecisionStatus.STOP_EFFICACY

        report = ProgressReport(
            sample_n=s.n,
            trajectory=res.e_value,
            boundary=1.0 / p.alpha,
            status=status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )
        return ProjectionResult(
            data=report, trace=metrics_traced.trace + protocol_traced.trace
        )


class BinomialEValueFinalProjector(Projector[FinalReport]):
    """
    Projector for 1-sample binomial e-value final report.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[FinalReport]:
        from earlysign.v1.methods.AVI.engines.binomial_e_value import (
            EProcessProtocol,
            compute_binomial_e_value,
        )

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Metrics
        metrics_traced = BinomialScoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        n_total = sum(a.metrics.n for a in metrics.arms.values())
        s_total = sum(a.metrics.successes for a in metrics.arms.values())

        # 3. Compute e-value
        res = compute_binomial_e_value(
            n=n_total,
            successes=s_total,
            null_p=p.null_p,
            alt_p=p.alt_p,
            alpha=p.alpha,
        )

        # 4. Determine Final Status
        final_status = DecisionStatus.CONTINUE_
        if res.is_rejected:
            final_status = DecisionStatus.STOP_EFFICACY

        report = FinalReport(
            sample_n=n_total,
            trajectory=res.e_value,
            is_rejected=res.is_rejected,
            final_status=final_status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )
        return ProjectionResult(
            data=report, trace=metrics_traced.trace + protocol_traced.trace
        )


class TrajectoryProjector(Projector[pd.DataFrame]):
    """
    Projector that reconstructs the history of LookResults from the ledger.
    Useful for plotting trajectories of e-values or confidence sequences.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[pd.DataFrame]:
        # 1. Filter for LookResult facts
        results = table.filter(table.type == "LookResult")

        # 2. Extract and sort by timestamp
        df = results.order_by(ibis.asc("timestamp")).execute()

        if df.empty:
            return ProjectionResult(data=pd.DataFrame(), trace=[])

        # 3. Parse payloads
        import json

        def _parse(p: Any) -> Any:
            if isinstance(p, str):
                return json.loads(p)
            return p

        payloads = df["payload"].apply(_parse)
        plot_df = pd.DataFrame(payloads.tolist())

        # Add timestamp for continuity
        plot_df["timestamp"] = df["timestamp"].values

        # Collect trace
        trace = [TraceId(str(u)) for u in df["uuid"]]

        return ProjectionResult(data=plot_df, trace=trace)


def plot_avi_trajectory(
    ledger: Any,
    title: str = "AVI Trajectory",
    xlabel: str = "Cumulative Sample Size (N)",
    ylabel: str = "Effect Estimate",
    ax: Optional[Any] = None,
    figsize: tuple[int, int] = (10, 6),
) -> Any:
    """
    Plots the history of AVI trajectory and boundaries.

    This function uses TrajectoryProjector to reconstruct history from the ledger
    and provides a standard visualization for AVI methods.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_avi_trajectory")

    from earlysign.v1.framework.session import Session

    with Session(ledger) as sess:
        history = sess.read(TrajectoryProjector()).data

    if history.empty:
        print("No history found in ledger to plot.")
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        pass

    # Plot sequence
    n = history["sample_n"]
    est = history["trajectory"]
    bound = history["boundary"]

    # Trajectory
    ax.plot(n, est, marker="o", label="Estimate", color="#1f77b4", linewidth=2)

    # Boundaries (if present)
    if bound.notnull().any():
        # Confidence Sequence / Boundaries
        ax.fill_between(
            n,
            est - bound,
            est + bound,
            color="#1f77b4",
            alpha=0.2,
            label="Confidence Sequence",
        )
        ax.plot(n, est - bound, color="#1f77b4", linestyle="--", alpha=0.5)
        ax.plot(n, est + bound, color="#1f77b4", linestyle="--", alpha=0.5)

    # Styling
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return ax
