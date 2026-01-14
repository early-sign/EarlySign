from typing import TYPE_CHECKING, Any, List, TypeVar

from earlysign.v1.framework.projector import Projector
from earlysign.v1.framework.trace import Traced, TraceId

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger

T = TypeVar("T")


class Session:
    """
    The primary execution context for the framework.

    A session defines the 'Scientific Horizon' by capturing the state of
    the Ledger at initiation. It also manages 'Implicit Trace Accumulation'
    to automatically track causality.
    """

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger
        self.horizon_id = ledger.latest_ts  # Define the Scientific Horizon
        self._session_trace: List[TraceId] = []

    def __enter__(self) -> "Session":
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
        # Filter ledger by the horizon timestamp captured at __init__
        return self.ledger.t.filter(self.ledger.t.ts <= self.horizon_id)

    def Read(self, projector: Projector[T]) -> Traced[T]:
        """
        Hydrates data using a Projector and accumulates its lineage.
        """
        # Get data filtered by the Scientific Horizon
        filtered_data = self.ledger.t.filter(self.ledger.t.ts <= self.horizon_id)

        # Execute projection
        result = projector.project(filtered_data)

        # Accumulate trace
        self._session_trace.extend(result.trace)

        return Traced(data=result.data, trace=result.trace)
