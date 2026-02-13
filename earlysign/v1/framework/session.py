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
    Type,
    TypeVar,
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
        mem_buffer_casted = mem_buffer.select(
            **{name: mem_buffer[name].cast(schema[name]) for name in schema.names}
        )
        return t.union(mem_buffer_casted)

    def read(self, projector: Projector[T]) -> Traced[T]:
        """Hydrate data using a Projector and accumulate its lineage.

        Args:
            projector: The projector instance to execute.

        Returns:
            The traced result of the projection.
        """
        # Execute projection using the unified table view (includes buffer)
        result = projector.project(self.table)

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
        combined_attributes = {"horizon": str(self.horizon_ts)}
        if identity:
            combined_attributes["entity_identity"] = identity
        if attributes:
            combined_attributes.update(attributes)

        row = self.ledger.prepare_row(
            data=record,
            attributes=combined_attributes,
            metadata={"trace": [str(t) for t in target_trace]},
        )
        self._session_commit_buffer.append(row)
        return uuidlib.UUID(hex=row["uuid"])

    def call_and_commit(
        self,
        result_type: Type[Any],
        func: Callable[..., Any],
        *args: Any,
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

        row = self.ledger.prepare_row(
            data=record,
            attributes={"is_result": True},
            metadata={"trace": [str(t) for t in target_trace]},
        )
        self._session_commit_buffer.append(row)
        return uuidlib.UUID(hex=row["uuid"])
