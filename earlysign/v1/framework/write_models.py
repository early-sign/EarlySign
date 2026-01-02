from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from earlysign.v1.framework.trace import Traced, TraceHash, extract_traces, stable_hash

if TYPE_CHECKING:
    from earlysign.v1.framework.session import Session

B = TypeVar("B", bound=BaseModel)


class WriteModel:
    """
    Fundamental operations for asserting facts into the Ledger.
    These operations ensure scientific provenance and idempotency.
    """

    @staticmethod
    def Commit(
        session: "Session",
        record: BaseModel,
        trace: Optional[List[TraceHash]] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> TraceHash:
        """
        Records a Pydantic model into the Ledger with its scientific trace.
        """
        target_trace = trace if trace is not None else session.trace

        # Compute the Trace Hash (The identity of this fact)
        trace_hash = stable_hash(
            session.horizon_id,
            target_trace,
            record.__class__.__name__,
            record.model_dump(),
        )

        # Write to physical ledger
        combined_labels = {
            "trace_hash": str(trace_hash),
            "horizon": str(session.horizon_id),
        }
        if labels:
            combined_labels.update(labels)

        session.ledger.insert(
            payload_type=record.__class__.__name__,
            payload=record.model_dump(),
            labels=combined_labels,
        )

        return trace_hash

    @staticmethod
    def CommitCallResult(
        session: "Session",
        result_type: Type[B],
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> B:
        """
        Executes a function and commits its result, keyed by scientific lineage.
        """
        # 1. Extract traces from arguments
        arg_traces = extract_traces(*args, **kwargs)

        # 2. Resolve target trace
        # If no Traced containers were found in args/kwargs, it's 'unspecified'
        # and we default to the union of all Reads in the session.
        target_trace = arg_traces if arg_traces is not None else session.trace

        # 3. Compute hash
        compute_hash = stable_hash(
            session.horizon_id,
            target_trace,
            result_type.__name__,
            str(args),
            str(kwargs),
        )

        # 4. Execute Logic
        raw_args = [v.data if isinstance(v, Traced) else v for v in args]
        raw_kwargs = {
            k: v.data if isinstance(v, Traced) else v for k, v in kwargs.items()
        }
        result_data = func(*raw_args, **raw_kwargs)

        # 5. Wrap and Commit
        if isinstance(result_data, dict):
            record = result_type(**result_data)
        else:
            record = result_data

        session.ledger.insert(
            payload_type=f"Result.{result_type.__name__}",
            payload=record.model_dump(),
            labels={"trace_hash": str(compute_hash), "is_result": True},
        )

        return record
