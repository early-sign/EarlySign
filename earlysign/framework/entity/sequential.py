"""
Sequential entities for trajectory tracking.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import ibis

from earlysign.core.util.json_ops import extract_json_scalar
from earlysign.framework.entity.core import Entity
from earlysign.framework.entity.snapshot import Snapshot
from earlysign.framework.projector import ProjectionResult, Projector
from earlysign.framework.trace import TraceId

S = TypeVar("S")
"""Generic type placeholder for individual states in a trajectory"""
Index = TypeVar("Index")
"""Generic type placeholder for the sequential index"""


class LatestStateProjector(Projector[Optional[S]], Generic[Index, S]):
    """A Projector that returns the latest state from a SequentialEntity trajectory.

    This is returned by SequentialEntity.latest property and provides a
    Projector interface for use with `sess.read()`.
    """

    def __init__(self, entity: "SequentialEntity[Index, S]"):
        self.entity = entity

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        """Inherit dependencies from the entity."""
        return self.entity.type_dependencies

    def project(self, table: ibis.Expr) -> ProjectionResult[Optional[S]]:
        """Project only the latest (most recent) state."""
        trajectory = self.entity.project_trajectory(table)
        if not trajectory:
            return ProjectionResult(data=None, trace=[])

        # Get the last element (most recent)
        _, latest_pr = trajectory[-1]
        return ProjectionResult(data=latest_pr.data, trace=latest_pr.trace)


class PointwiseTrajectoryProjector(Projector[List[Tuple[Index, S]]], Generic[Index, S]):
    """
    Projector that reconstructs a full trajectory from individual state snapshots.
    """

    def __init__(self, entity: "SequentialEntity[Index, S]"):
        self.entity = entity

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        """Inherit dependencies from the entity."""
        return self.entity.type_dependencies

    def project(self, table: ibis.Expr) -> ProjectionResult[List[Tuple[Index, S]]]:
        """
        Reconstruct the full trajectory by collecting all individual state snapshots.
        """
        trajectory_prs = self.project_detailed(table)
        data = [(idx, pr.data) for idx, pr in trajectory_prs]
        trace = [t for _, pr in trajectory_prs for t in pr.trace]
        return ProjectionResult(data=data, trace=trace)

    def project_detailed(
        self, table: ibis.Expr
    ) -> List[Tuple[Index, ProjectionResult[S]]]:
        """
        Detailed projection preserving per-item traces.
        """
        # For Pointwise, we query the table for the entity's data_type (or state_type implicitly)

        # Priority: state_type if available (e.g. SimpleSequentialEntity), else data_type
        # SimpleSequentialEntity has data_type=list, so we must use state_type.
        target_cls = getattr(self.entity, "state_type", None) or self.entity.data_type
        schema_name = target_cls.__name__

        identity_expr = extract_json_scalar(
            table.attributes, "entity_identity", "string"
        )
        matched = table.filter(
            # Schema type filter
            (table.type == schema_name)
            # Identity filter
            & (identity_expr == self.entity.identity)
        ).order_by("timestamp")

        # If the entity has a specific state_type defined (preferred for Pointwise), use it.
        # Otherwise fall back to data_type or dict.
        state_cls = getattr(self.entity, "state_type", None)

        trajectory: List[Tuple[Index, ProjectionResult[S]]] = []
        for i, row in matched.execute().iterrows():
            payload = row.get("payload", {})
            if isinstance(payload, str):
                import json

                payload = json.loads(payload)

            data_raw = payload
            data_inst = data_raw

            # Simplified Hydration: Only strict if explicit state_type is provided.
            if (
                state_cls
                and isinstance(data_raw, dict)
                and hasattr(state_cls, "model_validate")
            ):
                try:
                    data_inst = state_cls(**data_raw)
                except Exception:
                    # Allow fallback or re-raise?
                    # For robustness in reading, fallback or error.
                    # Given user feedback "is this really needed?", simple is better.
                    pass

            if data_inst is not None:
                # We assume the user guarantees S compatibility if they use Pointwise
                trajectory.append(
                    (
                        cast(Index, i),
                        ProjectionResult(
                            data=cast(S, data_inst), trace=[TraceId(str(row["uuid"]))]
                        ),
                    )
                )
        return trajectory


class SequentialEntity(Entity[List[Tuple[Index, S]]], Generic[Index, S], ABC):
    """Entity whose state is indexed by a sequential coordinate (look, sample, etc.).

    A Sequential Entity represents a trajectory of states $(S_0, S_1, \\dots, S_n)$
    treated as a single coherent Entity. This is useful for sequential procedures
    like Group Sequential Testing where the entire path of decisions matters.

    Supports two snapshot strategies:
    - COLLECTIVE: Store full trajectory in a single snapshot record
    - POINTWISE: Store one state per snapshot, reconstruct trajectory by collection

    Attributes:
        index_field: The name of the column that contains the sequential index
            (e.g., "look", "sample", "stage").
        snapshot_strategy: How to persist the trajectory (COLLECTIVE or POINTWISE).
        state_type: The Pydantic model type for individual states in the trajectory.

    Examples:
        >>> import ibis
        >>> import json
        >>> from typing import List, Tuple
        >>> from pydantic import BaseModel
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.framework.session import Session
        >>> from earlysign.framework.entity.sequential import SequentialEntity
        >>>
        >>> class SumState(BaseModel):
        ...     value: int
        >>>
        >>> class Add(BaseModel):
        ...     look: int
        ...     amount: int
        >>>
        >>> class SumsEntity(SequentialEntity[int, SumState]):
        ...     data_type = List[Tuple[int, SumState]]
        ...     index_field = "look"
        ...     snapshot_strategy = SequentialEntity.SnapshotStrategy.COLLECTIVE
        ...
        ...     def get_index_expr(self, table: ibis.Expr) -> ibis.Expr:
        ...         return extract_json_scalar(table.attributes, self.index_field, "int64")
        ...
        ...     @property
        ...     def initial_value(self) -> List[Tuple[int, SumState]]:
        ...         return []
        ...
        ...     def compute_step(self, index, prev_state, delta_expr) -> SumState:
        ...         current_val = prev_state.value if prev_state else 0
        ...         adds = delta_expr.filter(delta_expr.type == "Add").execute()
        ...         step_sum = 0
        ...         for _, row in adds.iterrows():
        ...             payload = row["payload"]
        ...             if isinstance(payload, str):
        ...                 payload = json.loads(payload)
        ...             step_sum += payload.get("amount", 0)
        ...         return SumState(value=current_val + step_sum)
        >>>
        >>> # Setup
        >>> con = ibis.duckdb.connect(":memory:")
        >>> ledger = Ledger(con, "events")
        >>> ledger.ensure()
        >>>
        >>> # 1. Insert initial data (Looks 1 and 2)
        >>> ledger.insert(Add(look=1, amount=10), attributes={"look": 1})
        UUID(...)
        >>> ledger.insert(Add(look=2, amount=20), attributes={"look": 2})
        UUID(...)
        >>>
        >>> # 2. First Read
        >>> with Session(ledger) as sess:
        ...     sums = SumsEntity(identity="test_sums")
        ...     result = sess.read(sums)
        ...     print(f"Len: {len(result.data)}")
        ...     print(f"L1: {result.data[0]}")
        ...     print(f"L2: {result.data[1]}")
        ...     sums.save(sess, result)
        Len: 2
        L1: (1, SumState(value=10))
        L2: (2, SumState(value=30))
        >>>
        >>> # 3. Add incremental data (Look 3)
        >>> ledger.insert(Add(look=3, amount=5), attributes={"look": 3})
        UUID(...)
        >>>
        >>> # 4. Second Read (Resume)
        >>> with Session(ledger) as sess2:
        ...     sums2 = SumsEntity(identity="test_sums")
        ...     result2 = sess2.read(sums2)
        ...     print(f"Len: {len(result2.data)}")
        ...     print(f"L3: {result2.data[2]}")
        Len: 3
        L3: (3, SumState(value=35))
    """

    index_field: str = "look"
    state_type: Optional[Type[S]] = None

    class SnapshotStrategy(Enum):
        """Strategy for persisting sequential entity trajectories."""

        COLLECTIVE = "collective"
        """Store the entire trajectory in a single snapshot record."""

        POINTWISE = "pointwise"
        """Store one snapshot per index; reconstruct trajectory by collection."""

    snapshot_strategy: SnapshotStrategy = SnapshotStrategy.COLLECTIVE

    @property
    def type_dependencies(self) -> List[Union[str, Tuple[str, str]]]:
        """
        Sequential entities depend on their collective snapshots (via super)
        AND their individual state elements if state_type is defined.
        """
        deps = super().type_dependencies
        if self.state_type:
            deps.append((self.state_type.__name__, self.identity))
        return deps

    def get_index_expr(self, table: ibis.Expr) -> ibis.Expr:
        """
        Return an Ibis expression for extracting the sequential index.

        Default implementation extracts from attributes[self.index_field] as a string.
        """
        return extract_json_scalar(table.attributes, self.index_field, "string")

    @abstractmethod
    def compute_step(
        self,
        index: Index,
        prev_state: Optional[S],
        delta_expr: ibis.Expr,
    ) -> S:
        """
        Compute the state at a specific index in the trajectory.
        """
        ...

    def compute(
        self,
        snapshot: Optional[Snapshot[List[Tuple[Index, S]]]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[List[Tuple[Index, S]]]:
        """
        Incremental fold for sequential trajectories.
        """
        if self.snapshot_strategy == self.SnapshotStrategy.POINTWISE:
            return PointwiseTrajectoryProjector(self).project(full_table)

        # Default: COLLECTIVE
        return self._compute_collective(snapshot, full_table)

    def _compute_collective(
        self,
        snapshot: Optional[Snapshot[List[Tuple[Index, S]]]],
        full_table: ibis.Expr,
    ) -> ProjectionResult[List[Tuple[Index, S]]]:
        """Perform the incremental fold for the collective trajectory."""
        trajectory: List[Tuple[Index, S]] = []
        if snapshot and isinstance(snapshot.data, list):
            trajectory = list(snapshot.data)
        elif snapshot:
            # Fallback for single item snapshot
            trajectory = [(cast(Index, 0), cast(S, snapshot.data))]

        # 1. Discover indices
        try:
            idx_expr = self.get_index_expr(full_table)
            indices_expr = (
                full_table.filter(idx_expr.notnull()).select(idx=idx_expr).distinct()
            )
            all_indices = [
                cast(Index, row.idx) for row in indices_expr.execute().itertuples()
            ]
            try:
                all_indices.sort(key=lambda x: float(cast(Any, x)))
            except (ValueError, TypeError):
                all_indices.sort()
        except Exception:
            # If default index discovery fails, we can't fold.
            all_indices = []

        # 2. Fold
        current_indices = {idx for idx, _ in trajectory}
        prev_state: Optional[S] = trajectory[-1][1] if trajectory else None

        for idx in all_indices:
            if idx not in current_indices:
                step_delta = full_table.filter(self.get_index_expr(full_table) == idx)
                state = self.compute_step(idx, prev_state, step_delta)
                trajectory.append((idx, state))
                prev_state = state
            else:
                prev_state = next(s for i, s in trajectory if i == idx)

        return ProjectionResult(data=trajectory, trace=[])

    @property
    def latest(self) -> LatestStateProjector[Index, S]:
        """
        Returns a Projector that yields only the latest (most recent) state.
        """
        return LatestStateProjector(self)

    def project_trajectory(
        self, table: ibis.Expr
    ) -> List[Tuple[Index, ProjectionResult[S]]]:
        """
        Project the full trajectory of states.
        """
        if self.snapshot_strategy == self.SnapshotStrategy.COLLECTIVE:
            # Reuse compute logic to get up-to-date trajectory
            snapshot = self._find_latest_snapshot(table)
            self._last_snapshot_uuid = snapshot.uuid if snapshot else None

            # Cast safety for the snapshot type which is generic T in BaseEntity
            snap_typed = snapshot

            # Since compute() relies on fold, we must pass the table as delta/full
            result = self.compute(snap_typed, table, table)

            return [
                (idx, ProjectionResult(data=state, trace=result.trace))
                for idx, state in result.data
            ]

        # POINTWISE
        return PointwiseTrajectoryProjector(self).project_detailed(table)
