"""
earlysign.core.traits
=====================

Trait (mixin) that attaches a small, typed DSL to any `Ledger` implementation.

This mixin assumes the host implements `Ledger.append`, `Ledger.emit_signal`,
and `Ledger.reader`. By inheriting `LedgerOps`, concrete ledgers gain:

- `write_event()` : append a record with typed parameters
- `emit()`        : append a signal record
- `latest()` / `iter_ns()` : typed convenience readers

Doctest (smoke with a tiny in-memory host):

>>> from datetime import datetime, timezone
>>> from earlysign.core.ledger import Row, LedgerReader
>>> from earlysign.core.names import Namespace
>>> class Host(LedgerOps):
...     def __init__(self): self.rows=[]
...     def append(self, **kw):
...         self.rows.append(Row(uuid="u", time_index=kw["time_index"], ts=kw["ts"],
...           namespace=str(kw["namespace"]), kind=kw["kind"], entity=kw["entity"],
...           snapshot_id=kw["snapshot_id"], tag=kw.get("tag"),
...           payload_type=kw["payload_type"], payload=kw["payload"]))
...         return self
...     def emit_signal(self, **kw):
...         return self.append(time_index=kw["time_index"], ts=kw["ts"],
...             namespace=kw.get("namespace", Namespace.SIGNALS.value), kind=kw.get("kind","emitted"),
...             entity=kw["entity"], snapshot_id=kw["snapshot_id"],
...             payload_type="Signal", payload={"topic": kw["topic"], "body": kw["body"]},
...             tag=kw.get("tag","signal"))
...     class R(LedgerReader):
...         def __init__(self, host): self.host=host
...         def iter_rows(self, **f): return iter(self.host.rows)
...         def latest(self, **f): return self.host.rows[-1] if self.host.rows else None
...         def count(self, **f): return len(self.host.rows)
...     def reader(self): return Host.R(self)
>>> L = Host()
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":5,"nB":5,"mA":1,"mB":0}, tag="obs")
>>> L.latest(namespace=Namespace.OBS).payload_type
'TwoPropObsBatch'
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Union

from earlysign.core.ledger import Ledger, LedgerReader, Row
from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex
from earlysign.core.ledger import NamespaceLike


class LedgerOps(Ledger):
    """A trait that attaches a small, typed DSL onto a `Ledger` implementation."""

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    # ---- writers ----

    def write_event(
        self,
        *,
        time_index: Union[TimeIndex, str],
        namespace: NamespaceLike,
        kind: str,
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        payload_type: str,
        payload: Dict[str, Any],
        tag: Optional[str] = None,
        ts: Optional[datetime] = None,
    ) -> None:
        """Append a typed event to the ledger."""
        self.append(
            time_index=str(time_index),
            ts=ts or self._now(),
            namespace=(
                namespace.value if isinstance(namespace, Namespace) else str(namespace)
            ),
            kind=kind,
            entity=str(experiment_id),
            snapshot_id=str(step_key),
            payload_type=payload_type,
            payload=payload,
            tag=tag,
        )

    def emit(
        self,
        *,
        time_index: Union[TimeIndex, str],
        experiment_id: Union[ExperimentId, str],
        step_key: Union[StepKey, str],
        topic: str,
        body: Dict[str, Any],
        tag: str = "signal",
        namespace: NamespaceLike = Namespace.SIGNALS,
        ts: Optional[datetime] = None,
    ) -> None:
        """Append a typed *signal* to the ledger."""
        self.emit_signal(
            time_index=str(time_index),
            ts=ts or self._now(),
            entity=str(experiment_id),
            snapshot_id=str(step_key),
            topic=topic,
            body=body,
            tag=tag,
            namespace=(
                namespace.value if isinstance(namespace, Namespace) else str(namespace)
            ),
        )

    # ---- readers ----

    def latest(
        self,
        *,
        namespace: Optional[NamespaceLike] = None,
        kind: Optional[str] = None,
        experiment_id: Optional[Union[ExperimentId, str]] = None,
        tag: Optional[str] = None,
    ) -> Optional[Row]:
        """Return latest row for given filters (or None)."""
        ns = (
            (namespace.value if isinstance(namespace, Namespace) else namespace)
            if namespace is not None
            else None
        )
        return self.reader().latest(
            namespace=ns,
            kind=kind,
            entity=str(experiment_id) if experiment_id else None,
            tag=tag,
        )

    def iter_ns(
        self,
        *,
        namespace: NamespaceLike,
        experiment_id: Optional[Union[ExperimentId, str]] = None,
    ) -> Iterable[Row]:
        """Iterate rows in a namespace (optionally filtered by experiment_id)."""
        ns = namespace.value if isinstance(namespace, Namespace) else str(namespace)
        return self.reader().iter_rows(
            namespace=ns, entity=str(experiment_id) if experiment_id else None
        )
