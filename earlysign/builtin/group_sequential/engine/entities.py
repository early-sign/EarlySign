"""
Interim Analyses Entity for Group Sequential Testing.

This module provides a SequentialEntity for managing the trajectory of
interim analyses in a Group Sequential Test.
"""

from typing import Any, List, Optional, Tuple, Type

import ibis
from pydantic import BaseModel

from earlysign.builtin.group_sequential.schema.logs import LookResult
from earlysign.core.util.json_ops import extract_json_scalar
from earlysign.framework.entity import (
    SequentialEntity,
)
from earlysign.framework.projector import ProjectionResult


class PreComputedBoundaries(BaseModel):
    """
    Stored results of a fixed-boundary calculation.
    """

    constant: float
    schedule: List[float]
    policy_name: str
    drift: Optional[float] = None


class InterimAnalyses(SequentialEntity[int, LookResult]):
    """
    Sequential Entity representing the trajectory of interim analyses.

    This entity tracks the state at each look (interim analysis) in a
    Group Sequential Test. The index is the look number (1, 2, 3, ...),
    and the state at each look is a LookResult containing:
    - z_stat: the test statistic
    - efficacy_boundary / futility_boundary: decision thresholds
    - status: continue, stop_efficacy, stop_futility, etc.

    Examples
    --------
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.framework.session import Session
    >>> from earlysign.builtin.group_sequential.engine.entities import LookResult
    >>> import ibis, duckdb
    >>> con = ibis.duckdb.connect(":memory:")
    >>> ledger = Ledger(con, "events"); ledger.ensure()
    >>> analyses = InterimAnalyses("my-trial")
    >>> # Verify initial state
    >>> with Session(ledger) as sess:
    ...     trajectory = sess.read(analyses)
    >>> trajectory.data
    []
    >>> # Simulate a look result
    >>> res = LookResult(look=1, sample_n=50, info_frac=0.5, z_stat=1.5,
    ...                  efficacy_boundary=2.5, futility_boundary=0.0,
    ...                  is_efficacy_crossed=False, is_futility_crossed=False,
    ...                  status="continue")
    >>> with Session(ledger) as sess:
    ...     sess.commit(res, identity="my-trial")
    UUID(...)
    >>> # Read trajectory back
    >>> with Session(ledger) as sess:
    ...     trajectory = sess.read(analyses)
    >>> len(trajectory.data)
    1
    >>> look_number, state = trajectory.data[0]
    >>> look_number
    1
    >>> state.z_stat
    1.5
    """

    state_type: Type[LookResult] = LookResult
    data_type: Type[List[Tuple[int, LookResult]]] = list
    index_field: str = "look"
    snapshot_strategy = SequentialEntity.SnapshotStrategy.COLLECTIVE

    @property
    def initial_value(self) -> List[Tuple[int, LookResult]]:
        return []

    def project_trajectory(
        self, table: ibis.Expr
    ) -> List[Tuple[int, ProjectionResult[LookResult]]]:
        """
        Collect trajectory by finding all LookResult records in the ledger
        matching this entity's identity.
        """
        from earlysign.framework.trace import TraceId

        # 1. Filter by LookResult type AND entity identity
        identity_expr = extract_json_scalar(
            table.attributes, "entity_identity", "string"
        )
        matched = table.filter(
            (table.type == "LookResult") & (identity_expr == self.identity)
        )

        # 2. Reconstruct trajectory (Latest record for each look index wins)
        # We order by timestamp to ensure we pick the most recent if multiple exist.
        results_df = matched.order_by(ibis.desc("timestamp")).execute()

        trajectory: List[Tuple[int, ProjectionResult[LookResult]]] = []
        seen_looks: set[int] = set()

        for _, row in results_df.iterrows():
            payload = row.get("payload", {})
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            # Reconstruct LookResult
            look_val = payload.get("look")
            idx = look_val if look_val is not None else 0

            if idx not in seen_looks:
                result = LookResult.model_validate(payload)
                trace = [TraceId(str(row["uuid"]))] if "uuid" in row else []
                trajectory.append((idx, ProjectionResult(data=result, trace=trace)))
                seen_looks.add(idx)

        # Sort by look number (0 comes first for pre-look results)
        trajectory.sort(key=lambda x: x[0])

        return trajectory

    def compute(
        self,
        snapshot: Optional[Any],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[List[Tuple[int, LookResult]]]:
        """
        Compute the trajectory from the full table.
        """
        trajectory = self.project_trajectory(full_table)
        data = [(idx, pr.data) for idx, pr in trajectory]
        return ProjectionResult(data=data, trace=[])

    def compute_step(
        self,
        index: int,
        prev_state: Optional[LookResult],
        delta_expr: ibis.Expr,
    ) -> LookResult:
        """
        Compute state at a specific look (not typically used with COLLECTIVE).
        """
        raise NotImplementedError(
            "compute_step is not used with COLLECTIVE snapshot strategy. "
            "Use project() instead."
        )
