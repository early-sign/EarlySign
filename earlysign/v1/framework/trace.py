"""
Trace primitives for scientific provenance tracking.

Trace = list of parent record UUIDs that contributed to a computation.
"""

from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    List,
    NewType,
    Optional,
    TypeVar,
)

T = TypeVar("T", covariant=True)

# Strong typing for trace identifiers (Ledger UUIDs)
TraceId = NewType("TraceId", str)


@dataclass(frozen=True)
class Traced(Generic[T]):
    """
    A container that wraps data with its scientific provenance (trace).

    The framework uses this to track causality through an analysis session.
    The trace is a list of parent record UUIDs.

    >>> t = Traced(data=42, trace=[TraceId("abc123")])
    >>> t.data
    42
    >>> t.trace
    ['abc123']
    """

    data: T
    trace: List[TraceId]


def extract_traces(*args: Any, **kwargs: Any) -> Optional[List[TraceId]]:
    """
    Recursively extracts and flattens TraceId values from Traced containers.
    Returns None if no Traced containers were encountered.
    Returns an empty list [] if Traced containers were encountered but they had empty traces.

    >>> t1 = Traced(10, [TraceId("h1")])
    >>> t2 = Traced(20, [TraceId("h2"), TraceId("h3")])
    >>> sorted(extract_traces(t1, "normal_value", t2, extra=t1))
    ['h1', 'h1', 'h2', 'h3']
    >>> extract_traces(1, 2, 3) is None
    True
    >>> extract_traces(Traced(10, []))
    []
    """
    found_traced = False
    traces: List[TraceId] = []

    for item in list(args) + list(kwargs.values()):
        if isinstance(item, Traced):
            found_traced = True
            traces.extend(item.trace)
        elif isinstance(item, (list, tuple)):
            sub = extract_traces(*item)
            if sub is not None:
                found_traced = True
                traces.extend(sub)
        elif isinstance(item, dict):
            sub = extract_traces(**item)
            if sub is not None:
                found_traced = True
                traces.extend(sub)

    return traces if found_traced else None
