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
    z_stat: float
    efficacy_boundary: Optional[float]
    futility_boundary: Optional[float]
    info_frac: float
    status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class FinalReport(BaseModel):
    """Generic final summary report."""

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
        # 1. Read the latest LookResult
        # We look for the most recent 'LookResult' payload
        results_df = table.filter(table.payload_type == "LookResult").execute()

        if results_df.empty:
            # Fallback or empty report
            return ProjectionResult(
                data=ProgressReport(
                    look=None,
                    sample_n=0,
                    z_stat=0.0,
                    efficacy_boundary=None,
                    futility_boundary=None,
                    info_frac=0.0,
                    status=DecisionStatus.CONTINUE_,
                ),
                trace=[],
            )

        # Get latest by index (assuming chronological order in ledger)
        # In a real system, we'd use 'created_at' or similar
        latest_row = results_df.iloc[-1]
        payload = latest_row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        latest_look = LookResult.model_validate(payload)

        # 2. Read Scoreboard for arm metrics (optional enrichment)
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data

        report = ProgressReport(
            look=latest_look.look,
            sample_n=latest_look.sample_n,
            z_stat=latest_look.z_stat,
            efficacy_boundary=latest_look.efficacy_boundary,
            futility_boundary=latest_look.futility_boundary,
            info_frac=latest_look.info_frac,
            status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class FinalProjector(Projector[FinalReport]):
    """
    Generic Projector for final study summary.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[FinalReport]:
        # 1. Read the latest LookResult
        results_df = table.filter(table.payload_type == "LookResult").execute()

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
            sample_n=latest_look.sample_n,
            z_stat=latest_look.z_stat,
            is_rejected=latest_look.is_efficacy_crossed,
            final_status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)
