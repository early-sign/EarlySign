"""
earlysign.core.ledger
=====================

Core *backend-agnostic* ledger interfaces and shared types.

- `Row`: a typed record returned by readers
- `PayloadRegistry`: optional decoders (payload_type -> dict -> typed object)
- `Ledger` / `LedgerReader`: minimal protocols used by traits/components
- `NamespaceLike`: `Namespace | str` for ergonomic, mypy-friendly APIs

This module intentionally knows nothing about any storage backend.

Examples:
---------
>>> from datetime import datetime, timezone
>>> from earlysign.core.names import Namespace
>>> class FakeReader:
...     def __init__(self, rows): self._rows = rows
...     def iter_rows(self, **f): return iter(self._rows)
...     def latest(self, **f): return self._rows[-1] if self._rows else None
...     def count(self, **f): return len(self._rows)
>>> class FakeLedger:
...     def __init__(self): self.rows=[]
...     def append(self, **kw):
...         self.rows.append(Row(uuid="u", time_index=kw["time_index"], ts=kw["ts"],
...             namespace=str(kw["namespace"]), kind=kw["kind"], entity=kw["entity"],
...             snapshot_id=kw["snapshot_id"], tag=kw.get("tag"),
...             payload_type=kw["payload_type"], payload=kw["payload"]))
...         return self
...     def emit_signal(self, **kw):
...         return self.append(time_index=kw["time_index"], ts=kw["ts"],
...             namespace=kw.get("namespace", Namespace.SIGNALS.value), kind=kw.get("kind","emitted"),
...             entity=kw["entity"], snapshot_id=kw["snapshot_id"],
...             payload_type="Signal", payload={"topic": kw["topic"], "body": kw["body"]},
...             tag=kw.get("tag","signal"))
...     def reader(self): return FakeReader(self.rows)
>>> L = FakeLedger()
>>> _ = L.append(time_index="t1", ts=datetime.now(timezone.utc), namespace=Namespace.OBS,
...              kind="observation", entity="exp#1", snapshot_id="s1",
...              payload_type="TwoPropObsBatch", payload={"nA":10,"nB":10,"mA":1,"mB":2})
>>> L.reader().count()
1
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Protocol, Union, TYPE_CHECKING

from earlysign.core.names import Namespace, ExperimentId, StepKey, TimeIndex

if TYPE_CHECKING:
    from earlysign.core.traits import LedgerOps


# Accept either Enum or plain string for flexibility with mypy/IDE help.
NamespaceLike = Union[Namespace, str]


@dataclass(frozen=True)
class Row:
    """A single ledger record with decoded payload (if decoder present)."""

    uuid: str
    time_index: str
    ts: datetime
    namespace: str
    kind: str
    entity: str
    snapshot_id: str
    tag: Optional[str]
    payload_type: str
    payload: Any  # decoded dict or typed object


class PayloadRegistry:
    """Optional decoders: payload_type -> callable(dict) -> typed object."""

    _decoders: Dict[str, Any] = {}

    @classmethod
    def register(cls, payload_type: str, decoder: Any) -> None:
        cls._decoders[payload_type] = decoder

    @classmethod
    def decode(cls, payload_type: str, payload: Any) -> Any:
        dec = cls._decoders.get(payload_type)
        if dec and isinstance(payload, dict):
            return dec(payload)
        return payload


class LedgerReader(Protocol):
    """Reader side of a ledger; supports filtered iteration and convenience queries."""

    def iter_rows(
        self,
        *,
        namespace: Optional[NamespaceLike] = None,
        kind: Optional[str] = None,
        entity: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Iterable[Row]: ...
    def latest(
        self,
        *,
        namespace: Optional[NamespaceLike] = None,
        kind: Optional[str] = None,
        entity: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Optional[Row]: ...
    def count(self, **filters: Any) -> int: ...


class LedgerBase(Protocol):
    """Base append-only ledger protocol used throughout the framework."""

    def append(
        self,
        *,
        time_index: str,
        ts: datetime,
        namespace: NamespaceLike,
        kind: str,
        entity: str,
        snapshot_id: str,
        payload_type: str,
        payload: Dict[str, Any],
        tag: Optional[str] = None,
    ) -> "LedgerBase": ...
    def emit_signal(
        self,
        *,
        time_index: str,
        ts: datetime,
        entity: str,
        snapshot_id: str,
        topic: str,
        body: Dict[str, Any],
        tag: str = "signal",
        namespace: NamespaceLike = Namespace.SIGNALS,
        kind: str = "emitted",
    ) -> "LedgerBase": ...
    def reader(self) -> LedgerReader: ...


class Ledger(LedgerBase, Protocol):
    """Ledger protocol with LedgerOps capabilities."""

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
    ) -> None: ...

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
    ) -> None: ...

    def latest(
        self,
        *,
        namespace: Optional[NamespaceLike] = None,
        kind: Optional[str] = None,
        experiment_id: Optional[Union[str, Any]] = None,
        tag: Optional[str] = None,
    ) -> Optional[Row]: ...

    def iter_ns(
        self,
        *,
        namespace: NamespaceLike,
        experiment_id: Optional[Union[ExperimentId, str]] = None,
    ) -> Iterable[Row]: ...
