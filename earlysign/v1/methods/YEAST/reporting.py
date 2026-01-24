from typing import Any, Dict, Optional, Union

import ibis
from pydantic import BaseModel

from earlysign.schema.ES3.YEAST.Log import DecisionStatus, LookResult
from earlysign.v1.framework.projector import (
    ProjectionResult,
    Projector,
)
from earlysign.v1.methods.binomial import Scoreboard


class ProgressReport(BaseModel):
    """YEAST interim progress report."""

    sample_n: int
    trajectory: float
    efficacy_boundary: Optional[float]
    status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class FinalReport(BaseModel):
    """YEAST final summary report."""

    sample_n: int
    trajectory: float
    is_rejected: bool
    final_status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class ProgressProjector(Projector[ProgressReport]):
    """
    Projector for YEAST interim monitoring.
    Reads the latest LookResult and Scoreboard from the ledger.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[ProgressReport]:
        # 1. Read the latest LookResult
        results_df = table.filter(table.payload_type == "LookResult").execute()

        if results_df.empty:
            # Fallback or empty report
            return ProjectionResult(
                data=ProgressReport(
                    sample_n=0,
                    trajectory=0.0,
                    efficacy_boundary=None,
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

        # 2. Read Scoreboard for arm metrics
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data

        report = ProgressReport(
            sample_n=latest_look.sample_n,
            trajectory=latest_look.trajectory,
            efficacy_boundary=latest_look.efficacy_boundary,
            status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class FinalProjector(Projector[FinalReport]):
    """
    Projector for YEAST final study summary.
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
            trajectory=latest_look.trajectory,
            is_rejected=latest_look.is_efficacy_crossed,
            final_status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)
