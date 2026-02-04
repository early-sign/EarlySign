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
"""

import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Optional,
    Type,
    TypeVar,
    cast,
)

import ibis

from earlysign.v1.framework.entity.base import BaseEntity
from earlysign.v1.framework.entity.snapshot import Snapshot
from earlysign.v1.framework.projector import ProjectionResult
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.trace import Traced

T = TypeVar("T")


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
        >>> import ibis
        >>> import json
        >>> from pydantic import BaseModel
        >>> from earlysign.core.ledger import Ledger
        >>> from earlysign.v1.framework.session import Session
        >>> from earlysign.v1.framework.trace import TraceId
        >>>
        >>> # 1. Define a simple schema and entity for testing
        >>> class CounterState(BaseModel):
        ...     count: int
        >>>
        >>> class Increment(BaseModel):
        ...     value: int = 1
        >>>
        >>> class CounterEntity(Entity[CounterState]):
        ...     data_type = CounterState
        ...
        ...     @property
        ...     def initial_value(self) -> CounterState:
        ...         return CounterState(count=0)
        ...
        ...     def compute(
        ...         self,
        ...         snapshot: Optional[Snapshot[CounterState]],
        ...         delta_expr: ibis.Expr,
        ...         full_table: ibis.Expr,
        ...     ) -> ProjectionResult[CounterState]:
        ...         current_count = snapshot.data.count if snapshot else 0
        ...
        ...         # Sum up 'Increment' events
        ...         # Note: filter execution might vary by backend, using minimal reproduction
        ...         if "payload" not in delta_expr.columns: # trace-only or empty
        ...             return ProjectionResult(data=CounterState(count=current_count), trace=[])
        ...
        ...         increments = delta_expr.filter(delta_expr.type == "Increment").execute()
        ...         for _, row in increments.iterrows():
        ...             payload = row["payload"]
        ...             # Handle potential string payload from some backends
        ...             if isinstance(payload, str):
        ...                 payload = json.loads(payload)
        ...             current_count += payload.get("value", 1)
        ...
        ...         trace = [TraceId(str(uid)) for uid in increments["uuid"]]
        ...         if snapshot and snapshot.uuid:
        ...             trace.insert(0, TraceId(str(snapshot.uuid)))
        ...
        ...         return ProjectionResult(data=CounterState(count=current_count), trace=trace)
        >>>
        >>> # 2. Setup Ledger and Data
        >>> con = ibis.duckdb.connect(":memory:")
        >>> ledger = Ledger(con, "events")
        >>> ledger.ensure()
        >>>
        >>> ledger.insert(Increment(value=1))
        >>> ledger.insert(Increment(value=2))
        >>>
        >>> # 3. First Session: Read and Snapshot
        >>> with Session(ledger) as sess:
        ...     counter = CounterEntity(identity="my_counter")
        ...     result = sess.Read(counter)
        ...     print(f"Count: {result.data.count}")
        ...     print(f"Trace Length: {len(result.trace)}")
        ...     counter.save(sess, result)
        Count: 3
        Trace Length: 2
        >>>
        >>> # Determine snapshot is saved
        >>> df = ledger.t.execute()
        >>> snapshot_row = df[df["type"] == "CounterState"]
        >>> len(snapshot_row)
        1
        >>>
        >>> # 4. Second Session: Resume from Snapshot
        >>> ledger.insert(Increment(value=10))
        >>> with Session(ledger) as sess2:
        ...     counter2 = CounterEntity(identity="my_counter")
        ...     result2 = sess2.Read(counter2)
        ...     print(f"Count: {result2.data.count}")
        ...     # Trace should be [SnapshotID, NewIncrementID]
        ...     print(f"Trace Length: {len(result2.trace)}")
        Count: 13
        Trace Length: 2
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
