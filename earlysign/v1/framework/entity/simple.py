"""
Lightweight entities for simple projection-based retrieval.
"""

from typing import List, Optional, Tuple, Type, TypeVar

import ibis

from earlysign.v1.framework.entity.base import BaseEntity
from earlysign.v1.framework.entity.sequential import SequentialEntity
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.trace import TraceId

T = TypeVar("T")
"""Generic type placeholder for the entity's state model"""
S = TypeVar("S")
"""Generic type placeholder for individual states in a trajectory"""
Index = TypeVar("Index")
"""Generic type placeholder for the sequential index"""


class SimpleEntity(BaseEntity[Optional[T]]):
    """A lightweight projection for 'the latest value' of a specific identity.

    Unlike Entity, SimpleEntity does not support incremental computation (fold).
    It simply looks for the latest record of the given schema type matching
    the identity in its labels.
    """

    def __init__(self, data_type: Type[T], identity: str):
        super().__init__(identity)
        self.data_type = data_type

    def project(self, table: ibis.Expr) -> ProjectionResult[Optional[T]]:
        """
        Find the latest record for this identity.
        """
        schema_name = self.data_type.__name__
        matched = table.filter(
            (table.type == schema_name)
            & (table.attributes["entity_identity"].str == self.identity)
        )

        latest = matched.order_by(ibis.desc("timestamp")).limit(1).execute()

        if latest.empty:
            # For SimpleEntity, we might want an initial value or None.
            # Here we follow the Projector interface but it might return None data.
            return ProjectionResult(data=None, trace=[])

        row = latest.iloc[0]
        # Payload is the data
        data_raw = row["payload"]
        data_inst = (
            self.data_type(**data_raw) if isinstance(data_raw, dict) else data_raw
        )

        return ProjectionResult(
            data=data_inst,
            trace=[TraceId(str(row["uuid"]))],
        )


class SimpleSequentialEntity(SequentialEntity[Index, S]):
    """A SequentialEntity that simplifies trajectory reconstruction.

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
