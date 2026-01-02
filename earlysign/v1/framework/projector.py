import json
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Protocol, Type, TypeVar, cast

import ibis
from pydantic import BaseModel

from earlysign.v1.framework.trace import TraceHash

T = TypeVar("T", covariant=True)
P = TypeVar("P", bound=BaseModel)


@dataclass(frozen=True)
class ProjectionResult(Generic[T]):
    """
    The output of a scientific projection.
    Wraps the hydrated state with the specific evidentiary trace that produced it.
    """

    data: T
    trace: List[TraceHash]


class Projector(Protocol, Generic[T]):
    """
    Defines the 'State-as-a-Fold' interface.
    Projectors are responsible for transforming raw event streams (Ibis tables)
    into structured scientific contexts.

    The projector MUST identify exactly which events (via TraceHash) constitute
    the resulting state.
    """

    def project(self, data: ibis.Expr) -> ProjectionResult[T]:
        """
        Hydrates data from the provided Ibis expression and identifies its trace.
        """
        ...


class ProtocolProjector(Projector[P]):
    """
    Standard projector for retrieving the most recent protocol from the ledger.
    """

    def __init__(self, protocol_type: Type[P]):
        self.protocol_type = protocol_type

    def project(self, table: ibis.Expr) -> ProjectionResult[P]:
        # Payload type in Pattern G is the class name
        type_name = self.protocol_type.__name__
        matched = table.filter(table.payload_type == type_name)

        # Get the latest one recorded in history
        latest = matched.order_by(ibis.desc("ts")).limit(1).execute()

        if latest.empty:
            raise RuntimeError(f"No protocol of type {type_name} found in ledger")

        row = latest.iloc[0]

        # Robust Metadata Parsing (Support both dict and JSON string backends)
        def _ensure_dict(val: Any) -> Dict[str, Any]:
            if isinstance(val, str):
                return cast(Dict[str, Any], json.loads(val))
            return dict(val) if val is not None else {}

        payload = _ensure_dict(row.get("payload"))
        labels = _ensure_dict(row.get("labels"))

        # Reconstruct into the Pydantic model
        try:
            data = self.protocol_type(**payload)
        except Exception as e:
            raise RuntimeError(
                f"Failed to hydrate protocol {type_name} from ledger payload: {e}"
            ) from e

        # Extract the trace hash (Scientific Lineage)
        raw_trace_hash = labels.get("trace_hash")
        trace = [TraceHash(str(raw_trace_hash))] if raw_trace_hash else []

        return ProjectionResult(data=data, trace=trace)
