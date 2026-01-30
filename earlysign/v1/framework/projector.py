import json
from dataclasses import dataclass
from typing import Any, Dict, Generic, Protocol, Type, TypeVar, cast

import ibis
from pydantic import BaseModel

from earlysign.v1.framework.trace import Traced, TraceId

T = TypeVar("T", covariant=True)
P = TypeVar("P", bound=BaseModel)


@dataclass(frozen=True)
class ProjectionResult(Traced[T]):
    """
    The output of a scientific projection.
    Wraps the hydrated state with the specific evidentiary trace that produced it.
    Inherits from Traced[T].
    """

    pass


class Projector(Protocol, Generic[T]):
    """
    Defines the 'State-as-a-Fold' interface.
    Projectors are responsible for transforming raw event streams (Ibis tables)
    into structured scientific contexts.

    The projector MUST identify exactly which events (via TraceId/uuid) constitute
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
        # Payload type is the class name
        type_name = self.protocol_type.__name__
        matched = table.filter(table.type == type_name)

        # Get the latest one recorded in history
        latest = matched.order_by(ibis.desc("timestamp")).limit(1).execute()

        if latest.empty:
            raise RuntimeError(f"No protocol of type {type_name} found in ledger")

        row = latest.iloc[0]

        # Robust Metadata Parsing (Support both dict and JSON string backends)
        def _ensure_dict(val: Any) -> Dict[str, Any]:
            if isinstance(val, str):
                return cast(Dict[str, Any], json.loads(val))
            return dict(val) if val is not None else {}

        payload = _ensure_dict(row.get("payload"))

        # Reconstruct into the Pydantic model
        try:
            data = self.protocol_type(**payload)
        except Exception as e:
            raise RuntimeError(
                f"Failed to hydrate protocol {type_name} from ledger payload: {e}"
            ) from e

        # Extract the uuid as the trace (Scientific Lineage)
        row_id = row.get("uuid")
        trace = [TraceId(str(row_id))] if row_id else []

        return ProjectionResult(data=data, trace=trace)
