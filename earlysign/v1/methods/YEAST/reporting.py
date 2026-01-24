from typing import Any, Dict, Optional, Union

import ibis
from pydantic import BaseModel

from earlysign.schema.ES3.YEAST.Log import DecisionStatus, LookResult
from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.methods.binomial import Scoreboard


class YeastProgressReport(BaseModel):
    """Interim progress report for YEAST."""

    sample_n: int
    trajectory: float
    efficacy_boundary: Optional[float]
    info_frac: float
    status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class YeastFinalReport(BaseModel):
    """Final summary report for YEAST."""

    sample_n: int
    trajectory: float
    is_rejected: bool
    final_status: Union[DecisionStatus, str]
    arms: Dict[str, Any] = {}


class YeastProgressProjector(Projector[YeastProgressReport]):
    """
    Projector for interim monitoring of YEAST trials.
    Reads the latest YEAST LookResult and Scoreboard.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[YeastProgressReport]:
        # 1. Read the latest LookResult
        # We look for the most recent 'LookResult' payload (YEAST namespace)
        # Note: If multiple methods log 'LookResult', we might need stricter filtering
        # but usually payload_type is just "LookResult". If GST and YEAST mix in same ledger,
        # structure differs. Pydantic validation handles this discrimination.
        results_df = table.filter(table.payload_type == "LookResult").execute()

        if results_df.empty:
            # Fallback or empty report
            return ProjectionResult(
                data=YeastProgressReport(
                    sample_n=0,
                    trajectory=0.0,
                    efficacy_boundary=None,
                    info_frac=0.0,
                    status=DecisionStatus.CONTINUE_,
                ),
                trace=[],
            )

        latest_row = results_df.iloc[-1]
        payload = latest_row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        # Validate against YEAST.Log.LookResult
        latest_look = LookResult.model_validate(payload)

        # 2. Read Scoreboard for arm metrics
        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data

        report = YeastProgressReport(
            sample_n=latest_look.sample_n,
            trajectory=latest_look.trajectory,
            efficacy_boundary=latest_look.efficacy_boundary,
            info_frac=latest_look.info_frac,
            status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)


class YeastFinalProjector(Projector[YeastFinalReport]):
    """
    Projector for final study summary of YEAST trials.
    """

    def project(self, table: ibis.Expr) -> ProjectionResult[YeastFinalReport]:
        results_df = table.filter(table.payload_type == "LookResult").execute()

        if results_df.empty:
            raise RuntimeError("No LookResult found in ledger for final report.")

        latest_row = results_df.iloc[-1]
        payload = latest_row["payload"]
        if isinstance(payload, str):
            import json

            payload = json.loads(payload)

        latest_look = LookResult.model_validate(payload)

        metrics_traced = Scoreboard(identity="metrics").project(table)
        metrics = metrics_traced.data

        report = YeastFinalReport(
            sample_n=latest_look.sample_n,
            trajectory=latest_look.trajectory,
            is_rejected=latest_look.is_efficacy_crossed,
            final_status=latest_look.status,
            arms={k: v.metrics.model_dump() for k, v in metrics.arms.items()},
        )

        return ProjectionResult(data=report, trace=metrics_traced.trace)
