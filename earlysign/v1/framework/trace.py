import hashlib
import json
from dataclasses import dataclass
from typing import Any, Generic, List, NewType, Optional, TypeVar

T = TypeVar("T")

# Strong typing for hashes to ensure scientific provenance
TraceHash = NewType("TraceHash", str)


@dataclass(frozen=True)
class Traced(Generic[T]):
    """
    A container that wraps data with its scientific provenance (trace).

    The framework uses this to track causality through an analysis session.

    >>> t = Traced(data=42, trace=[TraceHash("hash_1")])
    >>> t.data
    42
    >>> t.trace
    ['hash_1']
    """

    data: T
    trace: List[TraceHash]


def stable_hash(*args: Any) -> TraceHash:
    """
    Computes a deterministic MD5 hash for any set of serializable arguments.
    Arguments are converted to a sorted JSON string to ensure stability.

    >>> stable_hash("a", 1, {"b": 2})
    '455c36df2a5fd63b77d94a81312d81c8'
    >>> stable_hash({"x": 1, "y": 2}) == stable_hash({"y": 2, "x": 1})
    True
    """

    def serialize(obj: Any) -> Any:
        if isinstance(obj, (list, tuple)):
            return [serialize(i) for i in obj]
        if isinstance(obj, dict):
            return {str(k): serialize(v) for k, v in obj.items()}
        if hasattr(obj, "model_dump"):  # Handle Pydantic models
            return obj.model_dump()
        try:
            import pandas as pd

            if pd.isna(obj):
                return None
        except ImportError:
            pass
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj) if hasattr(obj, "__dict__") else obj

    # We use a custom serializer to handle sets/dicts/etc deterministically
    serialized_args = json.dumps(serialize(args), sort_keys=True)
    return TraceHash(hashlib.md5(serialized_args.encode("utf-8")).hexdigest())


def extract_traces(*args: Any, **kwargs: Any) -> Optional[List[TraceHash]]:
    """
    Recursively extracts and flattens TraceHash values from Traced containers.
    Returns None if no Traced containers were encountered.
    Returns an empty list [] if Traced containers were encountered but they had empty traces.

    >>> t1 = Traced(10, [TraceHash("h1")])
    >>> t2 = Traced(20, [TraceHash("h2"), TraceHash("h3")])
    >>> sorted(extract_traces(t1, "normal_value", t2, extra=t1))
    ['h1', 'h1', 'h2', 'h3']
    >>> extract_traces(1, 2, 3) is None
    True
    >>> extract_traces(Traced(10, []))
    []
    """
    found_traced = False
    traces: List[TraceHash] = []

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
