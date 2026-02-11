from typing import Optional

import ibis
import numpy as np

from earlysign.schema.ES3.Binomial import (
    ArmMetrics,
    ArmStatus,
    Scoreboard as ScoreboardSchema,
)
from earlysign.v1.framework.entity import Entity, Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId


class Scoreboard(Entity[ScoreboardSchema]):
    """Projector that tracks the status and metrics of all arms for binomial data.

    Represented as a Scoreboard domain model. Aggregates ArmData across all
    arms and tracks which ones are still active based on decision events.

    Attributes:
        data_type: The Pydantic model type for the scoreboard schema.
    """

    data_type = ScoreboardSchema

    @property
    def initial_value(self) -> ScoreboardSchema:
        return ScoreboardSchema(arms={})

    def compute(
        self,
        snapshot: Optional[Snapshot[ScoreboardSchema]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[ScoreboardSchema]:
        """Incremental fold for binomial scoreboard.

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
        # Look for both 'Observation' and 'ArmData' (and legacy BinomialData)
        batch_table = delta_expr.filter(
            (delta_expr.type == "ArmData")
            | (delta_expr.type == "BinomialData")
            | (delta_expr.type == "Observation")
        )

        is_arm_data = (batch_table.type == "ArmData") | (
            batch_table.type == "BinomialData"
        )
        # Simple iteration for prototype:
        batch_df = batch_table.select(
            "uuid",
            "payload",
            n=is_arm_data.ifelse(batch_table.payload["n"], 1).cast("int"),
            success=batch_table.payload["success"].cast("int"),
        ).execute()

        # 3. Aggregate deltas into current_arms
        for _, row in batch_df.iterrows():
            arm_name = row["payload"]["arm"]
            if arm_name not in current_arms:
                current_arms[arm_name] = ArmStatus(
                    metrics=ArmMetrics(n=0, successes=0, p_hat=0.0),
                    is_active=True,
                )

            status = current_arms[arm_name]
            status.metrics.n += int(row["n"])
            status.metrics.successes += int(row["success"])
            if status.metrics.n > 0:
                status.metrics.p_hat = status.metrics.successes / status.metrics.n

        # 4. Lineage Management
        tracked_uuids = [TraceId(str(uid)) for uid in batch_df["uuid"].tolist()]
        if snapshot and snapshot.uuid:
            tracked_uuids.insert(0, TraceId(str(snapshot.uuid)))

        return ProjectionResult(
            data=ScoreboardSchema(arms=current_arms), trace=tracked_uuids
        )


def calculate_binomial_z_statistic(control: ArmMetrics, treatment: ArmMetrics) -> float:
    """
    Computes the standard Z-statistic for two binomial proportions.
    """
    n_c, n_t = control.n, treatment.n
    cumulative_n = n_c + n_t

    if n_c < 2 or n_t < 2:
        return 0.0

    p_pool = (control.successes + treatment.successes) / cumulative_n
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

    if se <= 0:
        return 0.0

    return float((treatment.p_hat - control.p_hat) / se)
