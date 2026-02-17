"""
Session management for scientific analysis.

A session provides the context for executing projections and recording
events within a consistent 'scientific horizon'.
"""

import uuid as uuidlib
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.v1.framework.projector import Projector
from earlysign.v1.framework.trace import Traced, TraceId, extract_traces

T = TypeVar("T")
"""Generic type placeholder for projection results"""
B = TypeVar("B", bound=BaseModel)
"""Generic type placeholder for pydantic models"""


class Session:
    """The primary execution context for the framework.

    A session defines the 'Scientific Horizon' by capturing the state of
    the Ledger at initiation. It also manages 'Implicit Trace Accumulation'
    to automatically track causality.

    Attributes:
        ledger (Ledger): The event ledger instance.
        horizon_ts (Optional[Any]): The captured timestamp defining the scientific horizon.
        _session_trace (List[TraceId]): Accumulator for implicit traces during the session.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self._session_trace: List[TraceId] = []
        self._session_commit_buffer: List[Dict[str, Any]] = []
        self.horizon_ts: Optional[Any] = None  # Captured lazily

    def __enter__(self) -> Self:
        # Capture Scientific Horizon at session start
        self.horizon_ts = self.ledger.latest_ts
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Handle session exit, flushing buffered commits if the session was successful.

        During the flush, all buffered records are assigned a common timestamp
        equal to the actual commit time. This ensures that session-local writes
        do not 'leak' into the past of other concurrent sessions' horizons.

        Note on Immortality and Ordering:
        While records capture a provisional timestamp during buffering for local
        sorting, the final ledger timestamp represents when the data became
        globally visible. Since scientific analysis within this framework
        prioritizes causal ordering (Trace) and consistent horizons over absolute
        wall-clock buffering time, this normalization maintains ledger integrity
        without affecting analysis correctness.
        """
        if exc_type is None and self._session_commit_buffer:
            # Atomic commit to the ledger on success
            commit_ts = datetime.now(timezone.utc)
            for row in self._session_commit_buffer:
                row["timestamp"] = commit_ts

            self.ledger.insert_batch(self._session_commit_buffer)

        self._session_trace = []
        self._session_commit_buffer = []
        self.horizon_ts = None

    @property
    def trace(self) -> List[TraceId]:
        """Returns the current implicit session trace."""
        return list(self._session_trace)

    @property
    def table(self) -> Any:
        """Return a lazy table expression filtered by the Scientific Horizon.

        Returns:
            An Ibis table expression containing records up to the horizon.
        """
        import ibis

        t = self.ledger.t
        if self.horizon_ts is not None:
            t = t.filter(t.timestamp <= self.horizon_ts)

        if not self._session_commit_buffer:
            return t

        # Union with local buffer for 'read-your-writes' consistency
        mem_buffer = ibis.memtable(self._session_commit_buffer)
        # Cast memtable columns to match ledger schema exactly for backend compatibility
        schema = self.ledger.t.schema()

        projections = {}
        for name in schema.names:
            col = mem_buffer[name]
            target_type = schema[name]

            # BigQuery Hack: string -> json cast requires PARSE_JSON
            if (
                self.ledger.connector
                and self.ledger.connector.name == "bigquery"
                and target_type.is_json()
            ):
                from earlysign.core.util.ibis_bigquery import bq_parse_json

                projections[name] = bq_parse_json(col)
            else:
                projections[name] = col.cast(target_type)

        mem_buffer_casted = mem_buffer.select(**projections)
        return t.union(mem_buffer_casted)

    def read(self, projector: Projector[T]) -> Traced[T]:
        """Hydrate data using a Projector and accumulate its lineage.

        Args:
            projector: The projector instance to execute.

        Returns:
            The traced result of the projection.
        """
        # No automatic filtering here to ensure correct projection for entities that fold over multiple types.
        # Projectors are responsible for their own filtering within project().
        table = self.table
        result = projector.project(table)

        # Accumulate trace from the result (Expected to be Traced[T] or ProjectionResult[T])
        if hasattr(result, "trace") and result.trace:
            self._session_trace.extend(result.trace)

        return result

    def commit(
        self,
        record: Any,
        identity: Optional[str] = None,
        trace: Optional[List[TraceId]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> uuidlib.UUID:
        """Record a model into the Ledger with implicit context.

        Automatically attaches the session's scientific horizon and
        implicit trace if no explicit trace is provided.

        Args:
            record: The data record (usually a Pydantic model).
            identity: Optional unique identity for the record (e.g., entity ID).
            trace: Optional explicit parent traces. Defaults to session trace.
            attributes: Optional additional labels for the ledger.
        """
        target_trace = trace if trace is not None else self.trace
        combined_attributes = {}
        if identity:
            combined_attributes["entity_identity"] = identity
        if attributes:
            combined_attributes.update(attributes)

        combined_metadata = {
            "trace": [str(t) for t in target_trace],
            "session_horizon": str(self.horizon_ts),
        }

        row = self.ledger.prepare_row(
            data=record,
            attributes=combined_attributes,
            metadata=combined_metadata,
        )
        self._session_commit_buffer.append(row)
        return uuidlib.UUID(hex=row["uuid"])

    def call_and_commit(
        self,
        result_type: Type[Any],
        func: Callable[..., Any],
        *args: Any,
        identity: Optional[str] = None,
        **kwargs: Any,
    ) -> uuidlib.UUID:
        """
        Executes a function and commits its result with implicit lineage.

        Accepts an explicit 'trace' via kwargs. If not provided, extracts
        traces from args/kwargs or falls back to the session's implicit trace.
        """
        # 1. Resolve target trace
        explicit_trace = kwargs.pop("trace", None)
        if explicit_trace is not None:
            target_trace = explicit_trace
        else:
            arg_traces = extract_traces(*args, **kwargs)
            target_trace = arg_traces if arg_traces is not None else self.trace

        # 3. Execute Logic
        raw_args = [v.data if isinstance(v, Traced) else v for v in args]
        raw_kwargs = {
            k: v.data if isinstance(v, Traced) else v for k, v in kwargs.items()
        }
        result_data = func(*raw_args, **raw_kwargs)

        # 4. Commit the result
        if isinstance(result_data, dict):
            record = result_type(**result_data)
        else:
            record = result_data

        return self.commit(record, identity=identity, trace=target_trace)


class BacktestSession(Session):
    """
    Highly optimized Session for backtesting and historical replay.

    Implements transparent read-through caching for projectors. Caches are
    automatically invalidated upon commit of records that match the projector's
    declared type_dependencies.
    """

    def __init__(
        self,
        ledger: Ledger,
        invariant_projectors: Optional[List[Type[Projector[Any]]]] = None,
    ):
        """
        Initializes the BacktestSession.

        Args:
            ledger: The event ledger.
            invariant_projectors: Optional list of projector classes whose results
                should be pinned for the duration of the session (e.g., ProtocolProjector).
        """
        super().__init__(ledger)
        self._invariant_projector_types = invariant_projectors or []
        # Cache stores: (ProjectorType, Identity) -> (ProjectionResult, dependencies)
        self._projector_cache: Dict[
            Tuple[Type[Any], Optional[str]],
            Tuple[Traced[Any], List[Union[str, Tuple[str, str]]]],
        ] = {}

    def read(self, projector: Projector[T]) -> Traced[T]:
        """
        Read-through cache implementation for backtesting.
        """
        cache_key = (type(projector), getattr(projector, "identity", None))

        # 1. Check Cache
        if cache_key in self._projector_cache:
            result, _ = self._projector_cache[cache_key]
            # Maintain implicit session trace context
            if hasattr(result, "trace") and result.trace:
                self._session_trace.extend(result.trace)
            return cast(Traced[T], result)

        # 2. Cache Miss
        result = super().read(projector)

        # 3. Populate Cache
        deps = getattr(projector, "type_dependencies", [])
        self._projector_cache[cache_key] = (result, deps)

        return result

    def commit(
        self,
        record: Any,
        identity: Optional[str] = None,
        trace: Optional[List[TraceId]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> uuidlib.UUID:
        # Commit to ledger/buffer
        record_id = super().commit(record, identity, trace, attributes)

        # Selective Invalidation
        record_type_name = type(record).__name__
        self._invalidate_caches(record_type_name, identity)

        return record_id

    def _invalidate_caches(
        self, committed_type: str, committed_identity: Optional[str]
    ) -> None:
        """
        Invalidates cached results affected by the new commit.
        """
        to_remove = []
        for key, (_, deps) in self._projector_cache.items():
            proj_type, _ = key

            # Skip invariant projectors
            if proj_type in self._invariant_projector_types:
                continue

            # Check if any dependency matches the committed data
            for dep in deps:
                if isinstance(dep, tuple):
                    dep_type, dep_id = dep
                    if committed_type == dep_type and committed_identity == dep_id:
                        to_remove.append(key)
                        break
                elif committed_type == dep:
                    to_remove.append(key)
                    break

        for key in to_remove:
            del self._projector_cache[key]
