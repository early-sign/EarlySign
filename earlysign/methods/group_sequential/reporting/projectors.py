"""
Generic Projectors for Group Sequential Testing Reporting.

This module provides projectors that transform ledger facts (LookResult, Scoreboard)
into human-readable or UI-ready report objects.
"""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import ibis
from pydantic import BaseModel, Field

import earlysign.schema.ES3.GST as GST
from earlysign.framework.projector import (
    ProjectionResult,
    Projector,
)
from earlysign.methods.binomial import (
    Scoreboard as BinomialScoreboard,
)
from earlysign.methods.continuous import (
    Scoreboard as ContinuousScoreboard,
)
from earlysign.methods.group_sequential.execution.calculators import (
    ZStatisticCalculatorFactory,
)
from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult


class ProgressReport(BaseModel):
    """Generic interim progress report."""

    look: Optional[int]
    sample_n: int
    z_stat: Optional[float]
    z_stats: Optional[Dict[str, float]] = None
    efficacy_boundary: Optional[float]
    futility_boundary: Optional[float]
    info_frac: float
    status: Union[DecisionStatus, str]
    is_milestone: bool = False
    next_milestone_n: Optional[int] = None
    arms: Dict[str, Any] = {}


class FinalReport(BaseModel):
    """Generic final summary report."""

    look: Optional[int] = None
    sample_n: int
    z_stat: float
    z_stats: Optional[Dict[str, float]] = None
    is_rejected: bool
    final_status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class BacktestReport(BaseModel):
    """Report for historical backtesting."""

    final_report: FinalReport
    efficiency: float = Field(
        ..., description="Percentage of total data used (lower is better)."
    )
    samples_used: int
    total_samples: int
    stopping_reason: str


class ProgressProjector(Projector[ProgressReport]):
    """
    Generic Projector for interim monitoring.
    Reads the latest LookResult and Scoreboard from the ledger.
    """

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return [
            "LookResult",
            ("Scoreboard", "metrics"),
            "BinomialArmData",
            "ContinuousArmData",
            "GSTProtocol",
            "ClassicProtocol",
        ]

    def project(self, table: ibis.Expr) -> ProjectionResult[ProgressReport]:
        # 1. Resolve Protocol context
        protocol = None
        n_max_total = 0.0
        schedule_points = []
        response_type = GST.ResponseType.BINARY

        try:
            protocol_df = (
                table.filter(table.type.like("%Protocol"))
                .order_by(ibis.desc("timestamp"))
                .limit(1)
                .execute()
            )
            if not protocol_df.empty:
                import json

                payload = protocol_df.iloc[0]["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)

                # We need a proper GST.Protocol object to use calculators
                protocol = GST.Protocol.model_validate(payload)

                response_type = protocol.task.response_type

                # Determine max sample size
                timer = protocol.method.stopping_policy.timer
                if isinstance(timer, GST.SampleSizeTimer):
                    if isinstance(timer.max_sample_size, dict):
                        n_max_total = float(sum(timer.max_sample_size.values()))
                    else:
                        n_max_total = float(timer.max_sample_size)

                # Determine schedule
                sched = protocol.method.stopping_policy.schedule
                if isinstance(sched, GST.FixedSchedule):
                    schedule_points = sched.analyses
                elif isinstance(sched, GST.EquidistantSchedule):
                    import numpy as np

                    schedule_points = list(
                        np.linspace(1 / sched.n_looks, 1.0, sched.n_looks)
                    )
        except Exception:
            pass

        # 2. Read the latest Scoreboard
        metrics_traced: ProjectionResult[Any]
        if response_type == GST.ResponseType.CONTINUOUS:
            metrics_traced = ContinuousScoreboard(identity="metrics").project(table)
        else:
            metrics_traced = BinomialScoreboard(identity="metrics").project(table)

        metrics = metrics_traced.data
        total_n = sum(arm.metrics.total for arm in metrics.arms.values())

        # 3. Read the latest LookResult
        latest_look_traced = (
            table.filter(table.type == "LookResult")
            .order_by(ibis.desc("timestamp"))
            .limit(1)
            .execute()
        )

        latest_look = None
        if not latest_look_traced.empty:
            payload = latest_look_traced.iloc[0]["payload"]
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)
            latest_look = LookResult.model_validate(payload)

        # 4. Calculate current Z-statistic (even if not a milestone)
        curr_z_stats = None
        representative_z = None
        if protocol and len(metrics.arms) >= 1:
            try:
                calculator = ZStatisticCalculatorFactory.build(protocol)
                curr_z_stats = calculator.calculate(metrics, protocol)
                if curr_z_stats:
                    representative_z = float(max(curr_z_stats.values()))
            except Exception:
                pass

        # 5. Determine milestone status
        is_milestone = False
        next_milestone_n = None
        curr_info_frac = total_n / n_max_total if n_max_total > 0 else 0.0

        for t in schedule_points:
            if t > curr_info_frac + 0.001:
                next_milestone_n = int(t * n_max_total)
                break

        if latest_look:
            if total_n == latest_look.sample_n:
                is_milestone = True

            # If at milestone, use recorded boundaries but show CURRENT Z (proactive)
            # or should we show milestone Z? Usually, ProgressReport is proactive.
            report = ProgressReport(
                look=latest_look.look,
                sample_n=total_n,
                z_stat=representative_z or latest_look.z_stat,
                z_stats=curr_z_stats or latest_look.z_stats,
                efficacy_boundary=latest_look.efficacy_boundary,
                futility_boundary=latest_look.futility_boundary,
                info_frac=curr_info_frac,
                status=latest_look.status,
                is_milestone=is_milestone,
                next_milestone_n=next_milestone_n,
                arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
            )
        else:
            report = ProgressReport(
                look=None,
                sample_n=total_n,
                z_stat=representative_z,
                z_stats=curr_z_stats,
                efficacy_boundary=None,
                futility_boundary=None,
                info_frac=curr_info_frac,
                status=DecisionStatus.CONTINUE_,
                is_milestone=False,
                next_milestone_n=next_milestone_n,
                arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
            )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class FinalProjector(Projector[FinalReport]):
    """
    Generic Projector for final study summary.
    """

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return [
            "LookResult",
            ("Scoreboard", "metrics"),
            "BinomialArmData",
            "ContinuousArmData",
            "GSTProtocol",
        ]

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

        # 2. Read Scoreboard (Determine response_type from protocol)
        response_type = GST.ResponseType.BINARY
        try:
            protocol_df = (
                table.filter(table.type.like("%Protocol"))
                .order_by(ibis.desc("timestamp"))
                .limit(1)
                .execute()
            )
            if not protocol_df.empty:
                import json

                p_load = (
                    json.loads(protocol_df.iloc[0]["payload"])
                    if isinstance(protocol_df.iloc[0]["payload"], str)
                    else protocol_df.iloc[0]["payload"]
                )
                response_type = p_load.get("task", {}).get(
                    "response_type", GST.ResponseType.BINARY
                )
        except Exception:
            pass

        metrics_traced: ProjectionResult[Any]
        if response_type == GST.ResponseType.CONTINUOUS:
            metrics_traced = ContinuousScoreboard(identity="metrics").project(table)
        else:
            metrics_traced = BinomialScoreboard(identity="metrics").project(table)

        metrics = metrics_traced.data

        report = FinalReport(
            look=latest_look.look,
            sample_n=latest_look.sample_n,
            z_stat=latest_look.z_stat,
            z_stats=latest_look.z_stats,
            is_rejected=latest_look.is_efficacy_crossed,
            final_status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class BacktestProjector(Projector[BacktestReport]):
    """
    Projector for backtesting results.
    """

    def __init__(self, total_samples: int):
        self.total_samples = total_samples

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        return [
            "LookResult",
            ("Scoreboard", "metrics"),
            "BinomialArmData",
            "ContinuousArmData",
        ]

    def project(self, table: ibis.Expr) -> ProjectionResult[BacktestReport]:
        final_traced = FinalProjector().project(table)
        final_report = final_traced.data

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
