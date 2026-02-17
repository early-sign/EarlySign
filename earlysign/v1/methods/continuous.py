from typing import List, Optional, Tuple, Union

import ibis

from earlysign.core.util.ibis_ops import type_filter
from earlysign.schema.ES3.Continuous import (
    ArmMetrics,
    ArmStatus,
    ContinuousArmData,
    Scoreboard as ScoreboardSchema,
)
from earlysign.v1.framework.entity import Entity, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId


class Scoreboard(Entity[ScoreboardSchema]):
    """Projector that tracks the status and metrics of all arms for continuous data.

    Represented as a Scoreboard domain model. Aggregates ArmData across all
    arms and tracks which ones are still active based on decision events.

    Attributes:
        data_type: The Pydantic model type for the scoreboard schema.
    """

    data_type = ScoreboardSchema

    def __init__(self, identity: str, record_type: type = ContinuousArmData):
        super().__init__(identity)
        self.record_type = record_type

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        """Scoreboard depends on its own snapshots + raw ArmData records."""
        deps = super().type_dependencies
        deps.append(self.record_type.__name__)
        return deps

    @property
    def initial_value(self) -> ScoreboardSchema:
        return ScoreboardSchema(arms={})

    def compute(
        self,
        snapshot: Optional[Snapshot[ScoreboardSchema]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[ScoreboardSchema]:
        """Incremental fold for continuous scoreboard.

        Args:
            snapshot: The previous state snapshot.
            delta_expr: New events since the snapshot.
            full_table: The entire event table.

        Returns:
            The updated scoreboard projection.
        """
        # 1. Start with previous state
        current_state = snapshot.data if snapshot else self.initial_value
        current_arms = current_state.arms.copy()

        # 2. Process Delta: Data Ingestion
        batch_table = type_filter(delta_expr, self.record_type)

        from earlysign.core.util.json_ops import extract_json_scalar

        is_arm_data = batch_table.type == "ContinuousArmData"

        # NOTE on BigQuery Robustness:
        # even on non-conforming rows during a union (e.g. Protocol records mixed with data),
        # these extractions use JSON_VALUE + SAFE_CAST to return NULL safely.
        # The ifelse() then provides guaranteed short-circuiting to prevent BigQuery's
        # optimizer from crashing on invalid casts in filtered-out rows.
        total_json = extract_json_scalar(batch_table.payload, "total", "int")
        val_json = extract_json_scalar(batch_table.payload, "value", "float")
        sum_x_json = extract_json_scalar(batch_table.payload, "sum_x", "float")
        sum_x2_json = extract_json_scalar(batch_table.payload, "sum_x2", "float")

        # 1. total: 1 for observations (default), or extracted from ArmData
        total_expr = is_arm_data.ifelse(total_json, 1)

        # 2. sum_x: extracted, or fall back to 'value'
        sum_x_expr = is_arm_data.ifelse(sum_x_json, val_json)

        # 3. sum_x2: extracted, or fall back to value * value
        sum_x2_expr = is_arm_data.ifelse(sum_x2_json, val_json * val_json)

        batch_df = batch_table.select(
            "uuid",
            "payload",
            total=total_expr,
            sum_x=sum_x_expr,
            sum_x2=is_arm_data.ifelse(
                sum_x2_expr, 0.0
            ),  # Final guard for BQ evaluation order
        ).execute()

        # 3. Aggregate deltas into current_arms
        for _, row in batch_df.iterrows():
            arm_name = row["payload"]["arm"]
            if arm_name not in current_arms:
                current_arms[arm_name] = ArmStatus(
                    metrics=ArmMetrics(total=0, mean=0.0, variance=0.0),
                    is_active=True,
                )

            status = current_arms[arm_name]

            # Reconstruct running sums to update easily
            total_old = status.metrics.total
            sum_x_old = status.metrics.mean * total_old
            # Var = E[X^2] - (E[X])^2 => sum_x2 / total - mean^2
            sum_x2_old = (
                (status.metrics.variance + status.metrics.mean**2) * total_old
                if total_old > 0
                else 0.0
            )

            total_new = total_old + int(row["total"])
            sum_x_new = sum_x_old + float(row["sum_x"])
            sum_x2_new = sum_x2_old + float(row["sum_x2"])

            status.metrics.total = total_new
            if total_new > 0:
                status.metrics.mean = sum_x_new / total_new
                # Var = (1/total) * sum(x^2) - mean^2
                status.metrics.variance = (sum_x2_new / total_new) - (
                    status.metrics.mean**2
                )
            else:
                status.metrics.mean = 0.0
                status.metrics.variance = 0.0

        # 4. Lineage Management
        tracked_uuids = [TraceId(str(uid)) for uid in batch_df["uuid"].tolist()]
        if snapshot and snapshot.uuid:
            tracked_uuids.insert(0, TraceId(str(snapshot.uuid)))

        return ProjectionResult(
            data=ScoreboardSchema(arms=current_arms), trace=tracked_uuids
        )
