from typing import Any, Dict, List, Optional, Tuple, Union

import ibis
import pandas as pd
from pydantic import BaseModel

from earlysign.framework.projector import (
    ProjectionResult,
    Projector,
    ProtocolProjector,
)
from earlysign.framework.trace import TraceId
from earlysign.methods.binomial import Scoreboard as BinomialScoreboard
from earlysign.methods.continuous import Scoreboard as ContinuousScoreboard
from earlysign.schema.ES3.AVI import Protocol
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboardSchema
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboardSchema


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


class BacktestReport(BaseModel):
    """Report for historical backtesting (AVI)."""

    final_report: FinalReport
    efficiency: float
    samples_used: int
    total_samples: int
    stopping_reason: str


class ProgressProjector(Projector[ProgressReport]):
    """
    Projector for AVI interim monitoring.
    Reads the latest LookResult and Scoreboard from the ledger.
    """

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return ["LookResult", "Protocol", "Scoreboard"]

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
                    status=DecisionStatus.CONTINUE,
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

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return ["LookResult", "Protocol", "Scoreboard"]

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


class BacktestProjector(Projector[BacktestReport]):
    """
    Projector for AVI backtesting results.
    """

    def __init__(self, total_samples: int):
        self.total_samples = total_samples

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return ["LookResult", "Protocol", "Scoreboard"]

    def project(self, table: ibis.Expr) -> ProjectionResult[BacktestReport]:
        # 1. Get Final Report
        final_traced = FinalProjector().project(table)
        final_report = final_traced.data

        # 2. Calculate Efficiency
        efficiency = (
            final_report.sample_n / self.total_samples
            if self.total_samples > 0
            else 1.0
        )

        report = BacktestReport(
            final_report=final_report,
            efficiency=efficiency,
            samples_used=final_report.sample_n,
            total_samples=self.total_samples,
            stopping_reason=str(final_report.final_status),
        )

        return ProjectionResult(data=report, trace=final_traced.trace)


class BinomialEValueProgressProjector(Projector[ProgressReport]):
    """
    Projector for 1-sample binomial e-value monitoring.
    Stateless: Reconstructs the report from Protocol and Summary facts.
    """

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return ["Protocol", "Scoreboard"]

    def project(self, table: ibis.Expr) -> ProjectionResult[ProgressReport]:
        from earlysign.methods.AVI.adapters import BinomialAdapter
        from earlysign.methods.AVI.core import (
            BinomialEValueModel,
            EProcessProtocol,
        )

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Metrics
        metrics_traced = BinomialScoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        n_total, s_total = BinomialAdapter.extract_total_stats(metrics)

        # 3. Compute e-value
        res = BinomialEValueModel.compute(
            n=n_total,
            successes=s_total,
            null_p=p.null_p,
            alt_p=p.alt_p,
            alpha=p.alpha,
        )

        status = DecisionStatus.CONTINUE
        if res.is_rejected:
            status = DecisionStatus.STOP_DETECTED

        report = ProgressReport(
            sample_n=n_total,
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

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return ["Protocol", "Scoreboard"]

    def project(self, table: ibis.Expr) -> ProjectionResult[FinalReport]:
        from earlysign.methods.AVI.adapters import BinomialAdapter
        from earlysign.methods.AVI.core import (
            BinomialEValueModel,
            EProcessProtocol,
        )

        # 1. Read Protocol
        protocol_traced = ProtocolProjector(EProcessProtocol).project(table)
        p = protocol_traced.data

        # 2. Read Metrics
        metrics_traced = BinomialScoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        n_total, s_total = BinomialAdapter.extract_total_stats(metrics)

        # 3. Compute e-value
        res = BinomialEValueModel.compute(
            n=n_total,
            successes=s_total,
            null_p=p.null_p,
            alt_p=p.alt_p,
            alpha=p.alpha,
        )

        # 4. Determine Final Status
        final_status = DecisionStatus.CONTINUE
        if res.is_rejected:
            final_status = DecisionStatus.STOP_DETECTED

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

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return ["LookResult"]

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

    from earlysign.framework.session import Session

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
