from typing import Any, Optional

import ibis

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.builtin.YEAST.adapters import BinomialAdapter, ContinuousAdapter
from earlysign.builtin.YEAST.core import BoundaryModel, TrajectoryModel
from earlysign.framework.entity import Entity, Snapshot
from earlysign.framework.projector import ProjectionResult
from earlysign.framework.trace import TraceId
from earlysign.schema.ES3.Binomial import (
    Scoreboard as BinomialScoreboard,
)
from earlysign.schema.ES3.Continuous import (
    Scoreboard as ContinuousScoreboard,
)
from earlysign.schema.ES3.YEAST import Protocol
from earlysign.schema.ES3.YEAST.Log import (
    Boundary as BoundarySchema,
    DecisionStatus,
    LookResult,
)


class Boundary(Entity[BoundarySchema]):
    """
    Entity representing the fixed testing boundary for YEAST.
    """

    data_type = BoundarySchema

    @property
    def initial_value(self) -> BoundarySchema:
        import warnings

        warnings.warn(
            "YEAST boundary value has not been properly set. Returning an ineffective boundary.",
            UserWarning,
        )
        return BoundarySchema(value=None)

    @classmethod
    def calculate(cls, protocol: Protocol) -> float:
        return BoundaryModel.calculate_boundary_value(protocol)

    def compute(
        self,
        snapshot: Optional[Snapshot[BoundarySchema]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[BoundarySchema]:
        """
        Projects the Boundary state from the event stream.
        Logic: Use the latest 'Boundary' payload if available, else retain snapshot.
        """
        # Filter for Boundary updates
        boundary_updates = delta_expr.filter(delta_expr.type == "Boundary")

        # Get latest update
        latest_df = (
            boundary_updates.order_by(ibis.desc("timestamp"))
            .limit(1)
            .select("uuid", "payload")
            .execute()
        )

        if not latest_df.empty:
            # Found a new boundary update
            row = latest_df.iloc[0]
            record_uuid = str(row["uuid"])
            payload = row["payload"]

            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            new_data = BoundarySchema.model_validate(payload)

            # Trace lineage
            trace = [TraceId(record_uuid)]
            if snapshot and snapshot.uuid:
                trace.insert(0, TraceId(str(snapshot.uuid)))

            return ProjectionResult(data=new_data, trace=trace)

        if snapshot:
            # No update, keep existing
            return ProjectionResult(
                data=snapshot.data, trace=[TraceId(str(snapshot.uuid))]
            )

        # No snapshot and no update -> Default state
        return ProjectionResult(data=self.initial_value, trace=[])


class BinomialYEASTEngine:
    """
    Orchestrator for Binomial YEAST execution.
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def run(
        self,
        metrics: BinomialScoreboard,
        boundary: BoundarySchema,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result given current summary statistics.
        """
        # Extract arm names
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError(
                f"YEAST engine requires TwoArmComparison, but got {type(arms_struct).__name__}."
            )

        n_c, n_t, success_c, success_t = BinomialAdapter.extract_stats(
            metrics, control_key, treatment_key
        )
        cumulative_n = n_c + n_t
        raw_diff = float(success_t - success_c)

        # Standardized trajectory
        trajectory = TrajectoryModel.calculate_trajectory(n_c, n_t, raw_diff)

        # Use the passed boundary value
        boundary_val = boundary.value
        is_crossed = False
        if boundary_val is not None:
            is_crossed = trajectory > boundary_val

        status = DecisionStatus.CONTINUE
        if is_crossed:
            status = DecisionStatus.STOP_EFFICACY

        # Check for max N
        if hasattr(self.protocol.method, "expected_num_observations"):
            if cumulative_n >= self.protocol.method.expected_num_observations:
                if status == DecisionStatus.CONTINUE:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            sample_n=cumulative_n,
            info_frac=0.0,  # Placeholder
            trajectory=trajectory,
            raw_difference=raw_diff,
            efficacy_boundary=boundary_val,
            is_efficacy_crossed=is_crossed,
            status=status,
        )


class ContinuousYEASTEngine:
    """
    Orchestrator for Continuous YEAST execution.
    """

    def __init__(self, protocol: Protocol):
        self.protocol = protocol

    def run(
        self,
        metrics: ContinuousScoreboard,
        boundary: BoundarySchema,
        **kwargs: Any,
    ) -> LookResult:
        """
        Computes the test result given current summary statistics.
        """
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError(
                f"YEAST engine requires TwoArmComparison, but got {type(arms_struct).__name__}."
            )

        n_c, n_t, mean_c, mean_t = ContinuousAdapter.extract_stats(
            metrics, control_key, treatment_key
        )
        cumulative_n = n_c + n_t
        raw_diff = (mean_t * n_t) - (mean_c * n_c)

        # Standardized trajectory
        trajectory = TrajectoryModel.calculate_trajectory(n_c, n_t, raw_diff)

        boundary_val = boundary.value
        is_crossed = False
        if boundary_val is not None:
            is_crossed = trajectory > boundary_val

        status = DecisionStatus.CONTINUE
        if is_crossed:
            status = DecisionStatus.STOP_EFFICACY

        if hasattr(self.protocol.method, "expected_num_observations"):
            if cumulative_n >= self.protocol.method.expected_num_observations:
                if status == DecisionStatus.CONTINUE:
                    status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            sample_n=cumulative_n,
            info_frac=0.0,
            trajectory=float(trajectory),
            raw_difference=float(raw_diff),
            efficacy_boundary=boundary_val,
            is_efficacy_crossed=is_crossed,
            status=status,
        )
