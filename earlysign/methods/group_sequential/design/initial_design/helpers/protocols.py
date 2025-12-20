"""
Protocols and lightweight data structures for the group-sequential
initial-design refactor.

The types defined here are shared by both the in-memory and template-backed
paths so that scenario code can remain agnostic to the concrete procedure
implementations.

Examples
--------
>>> class _DummyProcedure:
...     def reset(self) -> None: ...
...     def ingest(self, cumulative: dict[str, int]) -> None: ...
...     def should_stop(self, look: int) -> dict[str, bool] | None:
...         return {"reject": True} if look > 0 else None
...     def snapshot_metadata(self) -> dict[str, int]:
...         return {"looks": 1}
>>> isinstance(_DummyProcedure(), ProcedureLike)
True
"""

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


@runtime_checkable
class ProcedureLike(Protocol):
    """Protocol implemented by procedures used in Monte-Carlo simulations."""

    def reset(self) -> None: ...

    def ingest(self, cumulative: Mapping[str, Any]) -> None: ...

    def should_stop(self, look: int) -> Optional[Dict[str, Any]]: ...

    def snapshot_metadata(self) -> Dict[str, Any]: ...


ProcedureFactory = Callable[
    [Sequence[float], int, Optional[Mapping[str, Any]], Optional[int]],
    ProcedureLike,
]


@dataclass(slots=True)
class ASNResult:
    """Summary statistics returned by ASN evaluation."""

    expected_n: float
    sd_n: float
    power: float
    details: Dict[str, Any]
