"""
Generic Projectors for Group Sequential Testing Reporting.

This module provides projectors that transform ledger facts (LookResult, Scoreboard)
into human-readable or UI-ready report objects.
"""

from typing import Any, Dict, Optional, Union

import ibis
from pydantic import BaseModel

from earlysign.schema.ES3.GST.Log import DecisionStatus, LookResult
from earlysign.v1.framework.projector import (
    ProjectionResult,
    Projector,
)
from earlysign.v1.methods.binomial import Scoreboard


class ProgressReport(BaseModel):
    """Generic interim progress report."""

    look: Optional[int]
    sample_n: int
    z_stat: Optional[float]
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
    is_rejected: bool
    final_status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class ProgressProjector(Projector[ProgressReport]):
    """
    Generic Projector for interim monitoring.
    Reads the latest LookResult and Scoreboard from the ledger.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[ProgressReport]:
        # 1. Read the latest Scoreboard (always up-to-date real-time metrics)
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data
        total_n = sum(arm.metrics.n for arm in metrics.arms.values())

        # 2. Read the latest LookResult (analytical facts / milestones)
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

        # 3. Read Protocol (to get n_max and schedule)
        # Instead of generic peeking, we look for the latest Protocol
        n_max = None
        schedule = []
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
                n_max = (
                    payload.get("method", {})
                    .get("stopping_policy", {})
                    .get("timer", {})
                    .get("max_sample_size")
                )
                schedule = (
                    payload.get("method", {})
                    .get("stopping_policy", {})
                    .get("schedule", {})
                    .get("analyses", [])
                )
        except Exception:
            pass

        # 4. Consolidate results
        curr_z: Optional[float] = None
        # Only compute Z if we have enough data and protocol for arm names
        if n_max and len(metrics.arms) >= 2:
            # Identify arms from protocol metadata (stored in the payload above)
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
                    arms_cfg = p_load.get("task", {}).get("arms", {})
                    if arms_cfg.get("kind") == "two_arm":
                        control_name = arms_cfg.get("control_arm_name")
                        treatment_name = arms_cfg.get("treatment_arm_name")

                        if (
                            control_name
                            and treatment_name
                            and control_name in metrics.arms
                            and treatment_name in metrics.arms
                        ):
                            from earlysign.v1.methods.binomial import (
                                calculate_binomial_z_statistic,
                            )

                            curr_z = calculate_binomial_z_statistic(
                                metrics.arms[control_name].metrics,
                                metrics.arms[treatment_name].metrics,
                            )
            except Exception:
                # Fallback: if we can't reliably identify arms, Z remains None
                pass

        is_milestone = False
        next_milestone_n = None
        curr_info_frac = 0.0

        if n_max:
            curr_info_frac = total_n / n_max if n_max > 0 else 0.0
            for t in schedule:
                if t > curr_info_frac + 0.001:
                    next_milestone_n = int(t * n_max)
                    break

        if latest_look:
            if total_n == latest_look.sample_n:
                is_milestone = True

            # Only report the recorded Z-stat if we are at a milestone (look triggered)
            curr_z = latest_look.z_stat if is_milestone else None

            report = ProgressReport(
                look=latest_look.look,
                sample_n=total_n,
                z_stat=curr_z,
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
                z_stat=None,
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

        # 2. Read Scoreboard
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data

        report = FinalReport(
            look=latest_look.look,
            sample_n=latest_look.sample_n,
            z_stat=latest_look.z_stat,
            is_rejected=latest_look.is_efficacy_crossed,
            final_status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)
