"""
Entity module for the framework.

Design Philosophy (CQRS Hybrid):
    In CQRS, "Entity" is a term that traditionally refers to aggregates in the Write Model.
    However, in our Event Sourcing architecture, we use "Entity" to describe a special type
    of projection that:

    1. Has a consistent **identity** across time
    2. Can be **snapshotted** for computational efficiency
    3. Is fundamentally derived from events (like a Read Model / Projection)

    An Entity is NOT a pure Write Model nor a pure Read Model—it's a hybrid concept.
    It represents a consistently-identifiable aggregate whose state is derived from
    event projections but can be cached as Snapshots.

Key Benefits:
    - Entities enable efficient incremental computation via snapshot + delta folding
    - The consistent identity allows Optimistic Concurrency Control (OCC)
    - Snapshots can be safely recomputed from events if needed

See Also:
    - docs/source/explanation/20260115_entity_and_CQRS.md
    - docs/source/explanation/JSS.md
"""

import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import ibis
from pydantic import BaseModel

from earlysign.v1.framework.base_entity import BaseEntity
from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.trace import Traced, TraceId

T = TypeVar("T")
S = TypeVar("S")
Index = TypeVar("Index")


class Snapshot(BaseModel, Generic[T]):
    """A recomputable intermediate fact (Memento).

    Snapshots cache the state of an Entity at a given point in time.
    They can always be recomputed from the underlying events.

    Attributes:
        entity_identity: The unique identity of the entity.
        data: The captured state data.
        timestamp: The ledger's last timestamp at the time of snapshot.
        uuid: Optional record ID for trace reference.
    """

    entity_identity: str
    data: T
    timestamp: Any  # Ledger's Last Timestamp
    uuid: Optional[str] = None  # Record record_id for trace reference


class Entity(BaseEntity[T], ABC):
    """Base class for identifiable aggregates with snapshot caching.

    In CQRS terms, an Entity is a hybrid:
    - Its state is reconstructed from events (Read/Projection)
    - But it has identity and can be snapshotted for efficiency
    - It is NOT a Write Model, but a special "Identifiable Projection"

    Subclasses must define:
    - `data_type`: The Pydantic model type for the entity's state
    - `compute()`: The fold logic to compute state from snapshot + delta

    Examples:
        >>> class MyState(BaseModel):
        ...     count: int = 0
        >>> class MyEntityFact(Entity[MyState]):
        ...     data_type = MyState
        ...     @property
        ...     def initial_value(self): return MyState()
        ...     def compute(self, snapshot, delta_expr, full_table):
        ...         # Fold logic here
        ...         return ProjectionResult(data=MyState(count=1), trace=[])
    """

    data_type: Type[T]

    @property
    @abstractmethod
    def initial_value(self) -> T:
        """
        Return the initial (identity) state for the entity.
        This state is used when no snapshots or events exist.
        """
        ...

    @abstractmethod
    def compute(
        self,
        snapshot: Optional[Snapshot[T]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[T]:
        """
        Implemented by subclasses to perform the actual folding.
        """
        ...

    def project(self, table: ibis.Expr) -> ProjectionResult[T]:
        """
        Coordinates the reconstruction of state from the Ledger.

        This method:
        1. Retrieves the latest snapshot for this entity's identity
        2. Filters for delta (new events since snapshot)
        3. Delegates to `compute()` for the actual fold

        Args:
            table: The Ibis expression representing the event table

        Returns:
            ProjectionResult with the current entity state
        """
        # 1. Retrieve latest snapshot for this identity
        snapshot = self._find_latest_snapshot(table)
        self._last_snapshot_uuid = snapshot.uuid if snapshot else None

        # 2. Filter for delta (new events since snapshot)
        if snapshot:
            delta_expr = table.filter(table.timestamp > snapshot.timestamp)
        else:
            delta_expr = table

        # 3. Delegate realization
        return self.compute(snapshot, delta_expr, table)

    def _find_latest_snapshot(self, table: ibis.Expr) -> Optional[Snapshot[T]]:
        """
        Efficiently find the latest snapshot for this identity.
        """
        # We search specifically for snapshots of this identity
        # Snapshots have the same payload_schema as the entity's data_type
        # but are identified by entity_identity label.
        schema_name = self.data_type.__name__
        matched = table.filter(
            (table.type == schema_name)
            & (table.attributes["entity_identity"].str == self.identity)
        )

        # Get the latest one
        latest = matched.order_by(ibis.desc("timestamp")).limit(1).execute()

        if not latest.empty:
            row = latest.iloc[0]

            # Robust Metadata Parsing
            def _ensure_dict(val: Any) -> Dict[str, Any]:
                if isinstance(val, str):
                    return cast(Dict[str, Any], json.loads(val))
                return dict(val) if val is not None else {}

            payload = _ensure_dict(row.get("payload"))
            # For snapshots, the payload IS the data (since payload_schema matches state)
            data_raw = payload

            if not data_raw:
                raise KeyError(
                    f"Snapshot for {self.identity} is missing data in payload."
                )

            # Hydrate the data into the expected Pydantic model T
            try:
                data_inst = (
                    self.data_type(**data_raw)
                    if isinstance(data_raw, dict)
                    else data_raw
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to hydrate snapshot data for {self.identity}: {e}"
                ) from e

            return Snapshot(
                entity_identity=self.identity,
                data=data_inst,
                timestamp=row["timestamp"],
                uuid=row.get("uuid"),
            )
        return None

    def save(self, session: Session, result: Traced[T]) -> None:
        """
        Standardizes how a new entity snapshot is committed.

        Args:
            session: The current session
            result: The traced result to save as a snapshot
        """
        # Optimization: Skip saving if there is no new information beyond the latest snapshot.
        if (
            hasattr(self, "_last_snapshot_uuid")
            and self._last_snapshot_uuid is not None
        ):
            trace_uuids = [str(t) for t in result.trace]
            if trace_uuids == [str(self._last_snapshot_uuid)]:
                return

        # Snapshot is just the data model, committed with identity
        session.Commit(result.data, identity=self.identity)


class LatestStateProjector(Projector[Optional[S]], Generic[Index, S]):
    """A Projector that returns the latest state from a SequentialEntity trajectory.

    This is returned by SequentialEntity.latest property and provides a
    Projector interface for use with `sess.Read()`.
    """

    def __init__(self, entity: "SequentialEntity[Index, S]"):
        self.entity = entity

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

        # Pointwise usually works with specific state records.
        # We rely on the entity identity filtering.

        attributes = table.attributes
        matched = table.filter(
            # Schema type filter
            (table.type == schema_name)
            # Identity filter
            & (attributes["entity_identity"].str == self.entity.identity)
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

    def get_index_expr(self, table: ibis.Expr) -> ibis.Expr:
        """
        Return an Ibis expression for extracting the sequential index.

        Default implementation extracts from attributes[self.index_field] as a string.
        """
        return table.attributes[self.index_field].cast("string")

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
