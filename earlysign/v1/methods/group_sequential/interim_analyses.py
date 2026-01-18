"""
Interim Analyses Entity for Group Sequential Testing.

This module provides a SequentialEntity for managing the trajectory of
interim analyses in a Group Sequential Test.
"""

from typing import Any, List, Optional, Tuple, Type

import ibis

from earlysign.v1.framework.entity import (
    SequentialEntity,
)
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.methods.group_sequential.binomial import LookResult


class InterimAnalyses(SequentialEntity[int, LookResult]):
    """
    Sequential Entity representing the trajectory of interim analyses.

    This entity tracks the state at each look (interim analysis) in a
    Group Sequential Test. The index is the look number (1, 2, 3, ...),
    and the state at each look is a LookResult containing:
    - z_stat: the test statistic
    - efficacy_boundary / futility_boundary: decision thresholds
    - status: continue, stop_efficacy, stop_futility, etc.

    The full trajectory can be retrieved via project(), which returns
    List[(look, LookResult)] representing the complete analysis history.

    Attributes:
        identity: Unique identifier for this analysis (e.g., experiment ID)
        snapshot_strategy: COLLECTIVE stores full trajectory in one snapshot

    Example:
        analyses = InterimAnalyses(identity="experiment_001")
        # trajectory = sess.Read(analyses)
        # for look, state in trajectory.data:
        #     print(f"Look {look}: z={state.z_stat:.2f}")
    """

    state_type: Type[LookResult] = LookResult
    data_type: Type[List[Tuple[int, LookResult]]] = list  # type: ignore
    index_field: str = "look"
    snapshot_strategy = SequentialEntity.SnapshotStrategy.COLLECTIVE

    def project_trajectory(
        self, table: ibis.Expr
    ) -> List[Tuple[int, ProjectionResult[LookResult]]]:
        """
        Collect trajectory by finding all LookResult records in the ledger.

        Overrides the parent method to search for LookResult payload types
        rather than Snapshot payloads.
        """
        # Look for LookResult records in table
        test_results = table.filter(table.payload_type == "LookResult")
        results_df = test_results.execute()

        trajectory: List[Tuple[int, ProjectionResult[LookResult]]] = []
        seen_looks: set[int] = set()

        for _, row in results_df.iterrows():
            payload = row.get("payload", {})
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            # Reconstruct LookResult
            # Even if look is None (not yet at a milestone), we want to see it in progress
            look_val = payload.get("look")
            idx = look_val if look_val is not None else 0

            if idx not in seen_looks:
                result = LookResult(**payload)
                trajectory.append((idx, ProjectionResult(data=result, trace=[])))
                seen_looks.add(idx)

        # Sort by look number (0 comes first for pre-look results)
        trajectory.sort(key=lambda x: x[0])

        return trajectory

    def compute(  # type: ignore
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
