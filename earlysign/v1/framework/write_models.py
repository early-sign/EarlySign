"""
Write models for the framework.

Design Philosophy:
==================
In event sourcing, ALL state and lineage flows through Read operations (Projectors).
Write operations (Commit, Ingest) are "fire and forget" - they record events to the
ledger but do not return identifiers. If you need to reference data after writing,
you Read it back via a Projector, which provides Traced[T] with proper lineage.

This design ensures:
1. All trace information comes from the ledger itself (via Projections)
2. No out-of-band state passing through return values
3. Clear separation: Write = record events, Read = reconstruct state and facts with lineage
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from earlysign.v1.framework.trace import (
    Traced,
    TraceId,
    extract_traces,
)

if TYPE_CHECKING:
    from earlysign.v1.framework.session import Session

B = TypeVar("B", bound=BaseModel)


class WriteModel:
    """
    Fundamental operations for asserting events into the Ledger.

    These operations record events with their scientific trace (parent uuids).
    They do NOT return identifiers - trace flows through Read, not Write.
    """

    @staticmethod
    def Commit(
        session: "Session",
        record: BaseModel,
        trace: Optional[List[TraceId]] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Records a Pydantic model into the Ledger with its scientific trace.

        Note: This method intentionally returns nothing. If you need to
        reference this data later, Read it back via a Projector.
        """
        target_trace = trace if trace is not None else session.trace

        combined_labels = {
            "horizon": str(session.horizon_id),
        }
        if labels:
            combined_labels.update(labels)

        session.ledger.insert(
            payload_type=record.__class__.__name__,
            payload=record.model_dump(),
            labels=combined_labels,
            trace=[str(t) for t in target_trace],
        )

    @staticmethod
    def CallAndCommit(
        session: "Session",
        result_type: Type[B],
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Executes a function and commits its result with scientific lineage.

        The trace is extracted from Traced inputs or defaults to session.trace.
        This method intentionally returns nothing - if you need the result,
        Read it back via a Projector.
        """
        # 1. Extract traces from arguments
        arg_traces = extract_traces(*args, **kwargs)

        # 2. Resolve target trace
        # If no Traced containers were found in args/kwargs, it's 'unspecified'
        # and we default to the union of all Reads in the session.
        target_trace = arg_traces if arg_traces is not None else session.trace

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

        session.ledger.insert(
            payload_type=f"Result.{result_type.__name__}",
            payload=record.model_dump(),
            labels={"is_result": True},
            trace=[str(t) for t in target_trace],
        )
