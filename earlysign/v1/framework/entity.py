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

from earlysign.v1.framework.projector import ProjectionResult, Projector
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.trace import Traced

T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=BaseModel)
Index = TypeVar("Index")


class Snapshot(BaseModel, Generic[T]):
    """
    A recomputable intermediate fact (Memento).

    Snapshots cache the state of an Entity at a given point in time.
    They can always be recomputed from the underlying events.
    """

    identity: str
    data: T
    ts: Any  # Ledger's Last Timestamp
    uuid: Optional[str] = None  # Record uuid for trace reference


class Entity(Projector[T], ABC):
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

    def __init__(self, identity: str):
        """
        Initialize an Entity with a consistent identity.

        Args:
            identity: Unique identifier for this entity instance.
        """
        self.identity = identity

    @abstractmethod
    def compute(
        self,
        snapshot: Optional[Snapshot[T]],
        delta_expr: ibis.Expr,
        full_table: ibis.Expr,
    ) -> ProjectionResult[T]:
        """
        Implemented by subclasses to perform the actual folding.

        Args:
            snapshot: The latest cached snapshot (or None if first computation)
            delta_expr: Events since the snapshot
            full_table: The complete event table

        Returns:
            ProjectionResult containing the computed state and its trace
        """
        pass

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
            delta_expr = table.filter(table.ts > snapshot.ts)
        else:
            delta_expr = table

        # 3. Delegate realization
        return self.compute(snapshot, delta_expr, table)

    def _find_latest_snapshot(self, table: ibis.Expr) -> Optional[Snapshot[T]]:
        """
        Efficiently find the latest snapshot for this identity.
        """
        # We search specifically for snapshots of this identity
        # The identity column is now top-level in the ledger
        snaps = table.filter(table.payload_type == "Snapshot")
        matched = snaps.filter(snaps.identity == self.identity)

        # Get the latest one
        latest = matched.order_by(ibis.desc("ts")).limit(1).execute()

        if not latest.empty:
            row = latest.iloc[0]

            # Robust Metadata Parsing
            def _ensure_dict(val: Any) -> Dict[str, Any]:
                if isinstance(val, str):
                    return cast(Dict[str, Any], json.loads(val))
                return dict(val) if val is not None else {}

            payload = _ensure_dict(row.get("payload"))
            data_raw = payload.get("data")

            if data_raw is None:
                raise KeyError(
                    f"Snapshot for {self.identity} is missing 'data' in payload."
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
                identity=self.identity,
                data=data_inst,
                ts=row["ts"],
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
        # We detect this by checking if the result's trace only consists of the last snapshot's UUID.
        if (
            hasattr(self, "_last_snapshot_uuid")
            and self._last_snapshot_uuid is not None
        ):
            trace_uuids = [str(t) for t in result.trace]
            if trace_uuids == [str(self._last_snapshot_uuid)]:
                return

        # Create a Snapshot record
        # Note: We include the session.horizon_id as the 'ts' anchor
        snap_record = Snapshot(
            identity=self.identity, data=result.data, ts=session.horizon_id
        )

        # We use a custom Commit that ensures identity is set in labels
        session.Commit(snap_record, labels={"identity": self.identity})


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


class SequentialEntity(Entity[S], Generic[Index, S], ABC):
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

    @abstractmethod
    def compute_step(
        self,
        index: Index,
        prev_state: Optional[S],
        delta_expr: ibis.Expr,
    ) -> S:
        """
        Compute the state at a specific index in the trajectory.

        Args:
            index: The sequential index (e.g., look number)
            prev_state: The state at the previous index (or None for index 0)
            delta_expr: Events relevant to this step

        Returns:
            The state at this index
        """
        pass

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
            snaps = table.filter(table.payload_type == "Snapshot")
            matched = snaps.filter(snaps.identity == self.identity)
            ordered = matched.order_by(ibis.asc("ts")).execute()

            trajectory: List[Tuple[Index, ProjectionResult[S]]] = []
            for i, row in ordered.iterrows():
                payload = row.get("payload", {})
                if isinstance(payload, str):
                    payload = json.loads(payload)
                data_raw = payload.get("data")
                if data_raw:
                    data_inst = (
                        self.data_type(**data_raw)
                        if isinstance(data_raw, dict)
                        else data_raw
                    )
                    trajectory.append(
                        (cast(Index, i), ProjectionResult(data=data_inst, trace=[]))
                    )
            return trajectory

        else:
            raise ValueError(
                f"Unknown snapshot strategy: {self.snapshot_strategy}. "
                f"Expected COLLECTIVE or POINTWISE."
            )
