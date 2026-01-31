from typing import Any, List, Optional, Tuple, Type, TypeVar, cast

import ibis
from pydantic import BaseModel

from earlysign.v1.framework.entity import SequentialEntity
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId

S = TypeVar("S", bound=BaseModel)
Index = TypeVar("Index")


class SimpleSequentialEntity(SequentialEntity[Index, S]):
    """
    A SequentialEntity that simplifies trajectory reconstruction.
    Instead of a complex fold, it treats every record of a specific type
    with a matching identity as a point in the sequence.

    Like SimpleEntity, it doesn't do incremental computation; it simply
    finds all matching records and collects them into a trajectory.
    """

    state_type: Type[S]
    data_type: Type[List[Tuple[Index, S]]] = list

    def __init__(self, state_type: Type[S], identity: str, index_field: str = "look"):
        super().__init__(identity)
        self.state_type = state_type
        self.index_field = index_field
        # We always use POINTWISE for 'Simple' sequential entities by default
        # because the user is just committing events (ArmMetrics) one by one.
        self.snapshot_strategy = SequentialEntity.SnapshotStrategy.POINTWISE

    @property
    def initial_value(self) -> List[Tuple[Index, S]]:
        return []

    def compute(
        self,
        snapshot: Optional[Any],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[List[Tuple[Index, S]]]:
        """
        Reconstruct the full trajectory by collecting all records of data_type.
        """
        trajectory = self.project_trajectory(full_table)
        data = [(idx, pr.data) for idx, pr in trajectory]
        # Trace is the collection of all uuids in the trajectory
        trace = []
        for _, pr in trajectory:
            trace.extend(pr.trace)

        return ProjectionResult(data=data, trace=trace)

    def project_trajectory(
        self, table: ibis.Expr
    ) -> List[Tuple[Index, ProjectionResult[S]]]:
        """
        Collect trajectory by finding all records of state_type matching the identity.
        """
        schema_name = self.state_type.__name__

        # We filter the table for the correct type and identity.
        # We use re_replace to handle potential quotes in the identity label if stored as JSON string.
        matched = table.filter(
            (table.type == schema_name)
            & (table.attributes["entity_identity"].str == self.identity)
        )
        ordered = matched.order_by(ibis.asc("timestamp")).execute()

        trajectory: List[Tuple[Index, ProjectionResult[S]]] = []
        for i, row in ordered.iterrows():
            payload = row.get("payload", {})
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            # Reconstruct the model instance
            data_inst = self.state_type.model_validate(payload)
            trace = [TraceId(str(row["uuid"]))]

            trajectory.append(
                (
                    cast(Index, i),
                    ProjectionResult(data=data_inst, trace=trace),
                )
            )

        return trajectory

    def compute_step(
        self,
        index: Index,
        prev_state: Optional[S],
        delta_expr: ibis.Expr,
    ) -> S:
        """
        Not used for simple pointwise reconstruction.
        """
        raise NotImplementedError(
            "SimpleSequentialEntity uses pointwise reconstruction."
        )
