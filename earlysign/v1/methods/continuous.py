from typing import Optional

import ibis

from earlysign.schema.ES3.Continuous import (
    ArmMetrics,
    ArmStatus,
    Scoreboard as ScoreboardSchema,
)
from earlysign.v1.framework.entity import Entity, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId


class Scoreboard(Entity[ScoreboardSchema]):
    """
    Projector that tracks the status and metrics of all arms for continuous data.
    Represented as a Scoreboard domain model.

    Aggregates ArmData across all arms and tracks which ones are
    still active based on decision events.
    """

    data_type = ScoreboardSchema

    def compute(
        self,
        snapshot: Optional[Snapshot[ScoreboardSchema]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[ScoreboardSchema]:
        # 1. Start with previous state
        current_arms = snapshot.data.arms.copy() if snapshot else {}

        # 2. Process Delta: Data Ingestion
        # Look for both 'Observation' and 'ArmData'
        batch_table = delta_expr.filter(
            (delta_expr.payload_type == "ArmData")
            | (delta_expr.payload_type == "Observation")
        )

        is_arm_data = batch_table.payload_type == "ArmData"

        # Simple iteration for prototype:
        # We need sum_x2 for Observation too
        val = batch_table.payload["value"].cast("float")
        sum_x2_expr = is_arm_data.ifelse(
            batch_table.payload["sum_x2"].cast("float"), val * val
        )

        batch_df = batch_table.select(
            "uuid",
            arm=batch_table.payload["arm"].cast("string").re_replace('^"|"$', ""),
            n=is_arm_data.ifelse(batch_table.payload["n"], 1).cast("int"),
            sum_x=is_arm_data.ifelse(
                batch_table.payload["sum_x"], batch_table.payload["value"]
            ).cast("float"),
            sum_x2=sum_x2_expr.cast("float"),
        ).execute()

        # 3. Aggregate deltas into current_arms
        for _, row in batch_df.iterrows():
            arm_name = row["arm"]
            if arm_name not in current_arms:
                current_arms[arm_name] = ArmStatus(
                    metrics=ArmMetrics(n=0, mean=0.0, variance=0.0),
                    is_active=True,
                )

            status = current_arms[arm_name]

            # Reconstruct running sums to update easily
            n_old = status.metrics.n
            sum_x_old = status.metrics.mean * n_old
            # Var = E[X^2] - (E[X])^2 => sum_x2 / n - mean^2
            sum_x2_old = (
                (status.metrics.variance + status.metrics.mean**2) * n_old
                if n_old > 0
                else 0.0
            )

            n_new = n_old + int(row["n"])
            sum_x_new = sum_x_old + float(row["sum_x"])
            sum_x2_new = sum_x2_old + float(row["sum_x2"])

            status.metrics.n = n_new
            if n_new > 0:
                status.metrics.mean = sum_x_new / n_new
                # Var = (1/n) * sum(x^2) - mean^2
                status.metrics.variance = (sum_x2_new / n_new) - (
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
