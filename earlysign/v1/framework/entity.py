"""
Entity module for the framework.

Design Philosophy (CQRS Hybrid):
================================
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
-- The consistent identity allows Optimistic Concurrency Control (OCC)
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
    """
    A recomputable intermediate fact (Memento).

    Snapshots cache the state of an Entity at a given point in time.
    They can always be recomputed from the underlying events.
    """

    entity_identity: str
    data: T
    timestamp: Any  # Ledger's Last Timestamp
    uuid: Optional[str] = None  # Record record_id for trace reference


class Entity(BaseEntity[T], ABC):
    """
    Base class for identifiable aggregates with snapshot caching.

    In CQRS terms, an Entity is a hybrid:
    - Its state is reconstructed from events (Read/Projection)
    - But it has identity and can be snapshotted for efficiency
    - It is NOT a Write Model, but a special "Identifiable Projection"

    Subclasses must define:
    - `data_type`: The Pydantic model type for the entity's state
    - `compute()`: The fold logic to compute state from snapshot + delta

    Example:
        class MyEntityFact(Entity[MyState]):
            data_type = MyState

            def compute(self, snapshot, delta_expr, full_table):
                # Fold logic here
                ...
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
    """
    A Projector that returns the latest state from a SequentialEntity trajectory.

    This is returned by SequentialEntity.latest property and provides a
    Projector interface for use with sess.Read().
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


class SequentialEntity(Entity[List[Tuple[Index, S]]], Generic[Index, S], ABC):
    """
    Entity whose state is indexed by a sequential coordinate (look, sample, etc.).

    A Sequential Entity represents a trajectory of states $(S_0, S_1, \\ldots, S_n)$
    treated as a single coherent Entity. This is useful for sequential procedures
    like Group Sequential Testing where the entire path of decisions matters.

    Supports two snapshot strategies:
    - COLLECTIVE: Store full trajectory in a single snapshot record
    - POINTWISE: Store one state per snapshot, reconstruct trajectory by collection

    Attributes:
        index_field: The name of the column that contains the sequential index
                     (e.g., "look", "sample", "stage")
        snapshot_strategy: How to persist the trajectory (COLLECTIVE or POINTWISE)

    Example:
        class ZStatisticTrajectory(SequentialEntity[int, ZStatState]):
            data_type = ZStatState
            index_field = "look"
            snapshot_strategy = SequentialEntity.SnapshotStrategy.COLLECTIVE

            def compute_step(self, index, prev_state, delta_expr):
                # Compute state at this index
                ...
    """

    index_field: str = "look"

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

        Default implementation extracts from attributes['look'] as a string.
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

        If a snapshot exists, it resumes from the latest index.
        Otherwise, it builds the trajectory from scratch (index 0).
        """
        trajectory: List[Tuple[Index, S]] = []
        if snapshot:
            trajectory = list(snapshot.data)

        # 1. Identify all sequential indices present in the full_table
        try:
            # Identify unique indices using the overridable expression
            idx_expr = self.get_index_expr(full_table)
            indices_expr = (
                full_table.filter(idx_expr.notnull())
                .select(idx=idx_expr)
                .distinct()
                .order_by("idx")
            )

            all_indices = [
                cast(Index, row.idx) for row in indices_expr.execute().itertuples()
            ]

            # Post-processing: Ensure indices are sorted correctly
            if all_indices:
                try:
                    # Try to sort numerically if they are numbers
                    all_indices.sort(key=lambda x: float(cast(Any, x)))
                except (ValueError, TypeError):
                    all_indices.sort()

        except Exception:
            # Fallback for entities that don't use this specific attribute-based indexing
            # Subclasses can override compute() if they have a custom index discovery.
            raw_trajectory = self.project_trajectory(full_table)
            data = [(idx, pr.data) for idx, pr in raw_trajectory]
            trace = [t for _, pr in raw_trajectory for t in pr.trace]
            return ProjectionResult(data=data, trace=trace)

        # 2. Fold: Compute state for each missing or new index
        current_indices = {idx for idx, _ in trajectory}
        prev_state: Optional[S] = trajectory[-1][1] if trajectory else None

        for idx in all_indices:
            if idx not in current_indices:
                # Filter delta_expr specifically for this index
                # We use the raw value for comparison
                idx_expr = self.get_index_expr(full_table)
                step_delta = full_table.filter(idx_expr == idx)
                state = self.compute_step(idx, prev_state, step_delta)
                trajectory.append((idx, state))
                prev_state = state
            else:
                # Update prev_state for the next iteration
                prev_state = next(s for i, s in trajectory if i == idx)

        return ProjectionResult(data=trajectory, trace=[])

    @property
    def latest(self) -> LatestStateProjector[Index, S]:
        """
        Returns a Projector that yields only the latest (most recent) state.

        Usage:
            with Session(ledger) as sess:
                latest_state = sess.Read(analyses.latest)
        """
        return LatestStateProjector(self)

    def project_trajectory(
        self, table: ibis.Expr
    ) -> List[Tuple[Index, ProjectionResult[S]]]:
        """
        Project the full trajectory of states.

        Returns a list of (index, state) pairs representing the complete
        history of this sequential entity.

        Args:
            table: The Ibis expression representing the event table

        Returns:
            List of (index, ProjectionResult) tuples in order
        """
        if self.snapshot_strategy == self.SnapshotStrategy.COLLECTIVE:
            # In COLLECTIVE mode, the latest snapshot contains the full history
            snapshot = self._find_latest_snapshot(table)
            if snapshot is None:
                return []

            # Check if snapshot.data is a list of (index, state) or just the latest state
            # For COLLECTIVE, usually we want the full trajectory stored in the data field.
            # Subclasses should handle hydration if it's a list.
            if isinstance(snapshot.data, list):
                return [
                    (idx, ProjectionResult(data=item, trace=[]))
                    for idx, item in snapshot.data
                ]

            # Default: single element trajectory
            return [(cast(Index, 0), ProjectionResult(data=snapshot.data, trace=[]))]

        elif self.snapshot_strategy == self.SnapshotStrategy.POINTWISE:
            # Collect all snapshots for this identity and reconstruct trajectory
            # Snapshots have the same payload_schema as the entity's data_type
            schema_name = self.data_type.__name__
            matched = table.filter(
                (table.type == schema_name)
                & (table.attributes["entity_identity"].str == self.identity)
            )
            ordered = matched.order_by(ibis.asc("timestamp")).execute()

            trajectory: List[Tuple[Index, ProjectionResult[S]]] = []
            for i, row in ordered.iterrows():
                payload = row.get("payload", {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                data_raw = payload
                if data_raw:
                    data_inst = (
                        self.data_type(**data_raw)
                        if isinstance(data_raw, dict)
                        else data_raw
                    )
                    trajectory.append(
                        (
                            cast(Index, i),
                            ProjectionResult(data=cast(S, data_inst), trace=[]),
                        )
                    )
            return trajectory

        else:
            raise ValueError(
                f"Unknown snapshot strategy: {self.snapshot_strategy}. "
                f"Expected COLLECTIVE or POINTWISE."
            )


class SimpleEntity(BaseEntity[Optional[T]]):
    """
    A lightweight projection for 'the latest value' of a specific identity.

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
                payload = json.loads(payload)

            # Reconstruct the model instance
            data_inst = (
                self.state_type(**payload) if isinstance(payload, dict) else payload
            )
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
