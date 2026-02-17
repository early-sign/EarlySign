from typing import List, Optional, Tuple, Union

import ibis
import numpy as np

from earlysign.core.util.ibis_ops import type_filter
from earlysign.framework.entity import Entity, Snapshot
from earlysign.framework.projector import ProjectionResult
from earlysign.framework.trace import TraceId
from earlysign.schema.ES3.Binomial import (
    ArmMetrics,
    ArmStatus,
    BinomialArmData,
    Scoreboard as ScoreboardSchema,
)


class Scoreboard(Entity[ScoreboardSchema]):
    """Projector that tracks the status and metrics of all arms for binomial data.

    Represented as a Scoreboard domain model. Aggregates ArmData across all
    arms and tracks which ones are still active based on decision events.

    Attributes:
        data_type: The Pydantic model type for the scoreboard schema.
    """

    data_type = ScoreboardSchema

    def __init__(self, identity: str, record_type: type = BinomialArmData):
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
        batch_table = type_filter(delta_expr, self.record_type)

        is_arm_data = batch_table.type == "BinomialArmData"
        # Simple iteration for prototype:
        from earlysign.core.util.json_ops import extract_json_scalar

        # NOTE on BigQuery Robustness:
        # We use ifelse() combined with extract_json_scalar() to guard against
        # BigQuery's non-deterministic evaluation order. Even if a row is filtered
        # out by a WHERE clause later, BigQuery may attempt to evaluate CAST(JSON AS INT)
        # on non-conforming rows (like Protocol records), causing a crash.
        # ifelse() provides guaranteed short-circuiting in BigQuery.
        batch_df = batch_table.select(
            "uuid",
            "payload",
            total=is_arm_data.ifelse(
                extract_json_scalar(batch_table.payload, "total", "int"), 1
            ),
            success=is_arm_data.ifelse(
                extract_json_scalar(batch_table.payload, "success", "int"), 0
            ),
        ).execute()

        # 3. Aggregate deltas into current_arms
        for _, row in batch_df.iterrows():
            arm_name = row["payload"]["arm"]
            if arm_name not in current_arms:
                current_arms[arm_name] = ArmStatus(
                    metrics=ArmMetrics(total=0, successes=0, p_hat=0.0),
                    is_active=True,
                )

            status = current_arms[arm_name]
            status.metrics.total += int(row["total"])
            status.metrics.successes += int(row["success"])
            if status.metrics.total > 0:
                status.metrics.p_hat = status.metrics.successes / status.metrics.total

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
    n_c, n_t = control.total, treatment.total
    cumulative_n = n_c + n_t

    if n_c < 2 or n_t < 2:
        return 0.0

    p_pool = (control.successes + treatment.successes) / cumulative_n
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

    if se <= 0:
        return 0.0

    return float((treatment.p_hat - control.p_hat) / se)
