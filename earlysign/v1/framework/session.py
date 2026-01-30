from typing import Any, Callable, Dict, List, Optional, Self, Type, TypeVar

from pydantic import BaseModel

from earlysign.core.ledger import Ledger
from earlysign.v1.framework.projector import Projector
from earlysign.v1.framework.trace import Traced, TraceId, extract_traces
from earlysign.v1.framework.writer import Writer

T = TypeVar("T")
B = TypeVar("B", bound=BaseModel)


class Session:
    """
    The primary execution context for the framework.

    A session defines the 'Scientific Horizon' by capturing the state of
    the Ledger at initiation. It also manages 'Implicit Trace Accumulation'
    to automatically track causality.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.horizon_ts: Optional[Any] = None  # Captured lazily
        self._session_trace: List[TraceId] = []

    def __enter__(self) -> Self:
        # Capture Scientific Horizon at session start
        self.horizon_ts = self.ledger.latest_ts
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._session_trace = []

    @property
    def trace(self) -> List[TraceId]:
        """Returns the current implicit session trace."""
        return list(self._session_trace)

    @property
    def table(self) -> Any:
        """
        Returns a lazy table expression filtered by the Scientific Horizon.
        """
        if self.horizon_ts is None:
            return self.ledger.t
        return self.ledger.t.filter(self.ledger.t.timestamp <= self.horizon_ts)

    def Read(self, projector: Projector[T]) -> Traced[T]:
        """
        Hydrates data using a Projector and accumulates its lineage.
        """
        filtered_data = self.ledger.t
        if self.horizon_ts is not None:
            filtered_data = filtered_data.filter(
                self.ledger.t.timestamp <= self.horizon_ts
            )

        # Execute projection
        result = projector.project(filtered_data)

        # Accumulate trace from the result (Expected to be Traced[T] or ProjectionResult[T])
        if hasattr(result, "trace") and result.trace:
            self._session_trace.extend(result.trace)

        return result

    def Commit(
        self,
        record: Any,
        identity: Optional[str] = None,
        trace: Optional[List[TraceId]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Records a model into the Ledger with implicit context.

        Automatically attaches the session's scientific horizon and
        implicit trace if no explicit trace is provided.
        """
        target_trace = trace if trace is not None else self.trace
        combined_attributes = {"horizon": str(self.horizon_ts)}
        if attributes:
            combined_attributes.update(attributes)

        Writer.Commit(
            self,
            record,
            identity=identity,
            trace=target_trace,
            attributes=combined_attributes,
        )

    def CallAndCommit(
        self,
        result_type: Type[Any],
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
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

        Writer.CallAndCommit(
            self,
            result_type,
            func,
            *args,
            trace=target_trace,
            **kwargs,
        )
