"""
earlysign.backends.polars.ledger
================================

A concrete **Polars-backed** ledger with JSON-UTF8 payload.
No persistence here (see `earlysign.backends.polars.io`).

- Inherits `Ledger` to expose the typed DSL as native methods.
- Implements `append()`, `emit_signal()`, and a `LedgerReader`.

Examples
--------
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> L = PolarsLedger()
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":5,"nB":6,"mA":1,"mB":0}, tag="obs")
>>> L.reader().count(namespace=Namespace.OBS.value)
1
"""

from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, cast, Iterator

import polars as pl

from earlysign.core.ledger import (
    LedgerReader,
    Row,
    PayloadRegistry,
    NamespaceLike,
    Ledger,
)
from earlysign.core.names import Namespace
from earlysign.core.traits import LedgerOps


class PolarsLedger(LedgerOps, Ledger):
    """Polars-backed append-only ledger with JSON-UTF8 payload column."""

    _SCHEMA = {
        "uuid": pl.Utf8,
        "time_index": pl.Utf8,
        "ts": pl.Datetime(time_unit="us", time_zone="UTC"),
        "namespace": pl.Utf8,
        "kind": pl.Utf8,
        "entity": pl.Utf8,  # experiment_id
        "snapshot_id": pl.Utf8,  # step_key
        "tag": pl.Utf8,
        "payload_type": pl.Utf8,
        "payload": pl.Utf8,  # JSON string
    }

    def __init__(self, df: Optional[pl.DataFrame] = None) -> None:
        self._df = (
            df if df is not None else pl.DataFrame(schema=cast(Any, self._SCHEMA))
        )

    # ---- Ledger interface ----

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
    ) -> "PolarsLedger":
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        row = pl.DataFrame(
            {
                "uuid": [str(uuid.uuid4())],
                "time_index": [time_index],
                "ts": [ts],
                "namespace": [
                    str(
                        namespace.value
                        if isinstance(namespace, Namespace)
                        else namespace
                    )
                ],
                "kind": [kind],
                "entity": [entity],
                "snapshot_id": [snapshot_id],
                "tag": [tag],
                "payload_type": [payload_type],
                "payload": [json.dumps(payload, separators=(",", ":"))],
            }
        )
        self._df = pl.concat([self._df, row], how="vertical_relaxed")
        return self

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
    ) -> "PolarsLedger":
        return self.append(
            time_index=time_index,
            ts=ts,
            namespace=namespace,
            kind=kind,
            entity=entity,
            snapshot_id=snapshot_id,
            payload_type="Signal",
            payload={"topic": topic, "body": body},
            tag=tag,
        )

    class _Reader(LedgerReader):
        def __init__(self, df: pl.DataFrame) -> None:
            self.df = df

        def _filter(
            self,
            *,
            namespace: Optional[Any] = None,
            kind: Optional[str] = None,
            entity: Optional[str] = None,
            tag: Optional[str] = None,
        ) -> pl.DataFrame:
            q = self.df
            if namespace is not None:
                ns = namespace.value if isinstance(namespace, Namespace) else namespace
                q = q.filter(pl.col("namespace") == str(ns))
            if kind is not None:
                q = q.filter(pl.col("kind") == kind)
            if entity is not None:
                q = q.filter(pl.col("entity") == entity)
            if tag is not None:
                q = q.filter(pl.col("tag") == tag)
            return q

        def iter_rows(
            self,
            *,
            namespace: Optional[Any] = None,
            kind: Optional[str] = None,
            entity: Optional[str] = None,
            tag: Optional[str] = None,
        ) -> Iterator[Row]:
            q = self._filter(namespace=namespace, kind=kind, entity=entity, tag=tag)
            for rec in q.iter_rows(named=True):
                try:
                    payload = json.loads(rec["payload"]) if rec["payload"] else {}
                except Exception:
                    payload = {}
                yield Row(
                    uuid=rec["uuid"],
                    time_index=rec["time_index"],
                    ts=rec["ts"],
                    namespace=rec["namespace"],
                    kind=rec["kind"],
                    entity=rec["entity"],
                    snapshot_id=rec["snapshot_id"],
                    tag=rec["tag"],
                    payload_type=rec["payload_type"],
                    payload=PayloadRegistry.decode(rec["payload_type"], payload),
                )

        def latest(
            self,
            *,
            namespace: Optional[Any] = None,
            kind: Optional[str] = None,
            entity: Optional[str] = None,
            tag: Optional[str] = None,
        ) -> Optional[Row]:
            q = self._filter(namespace=namespace, kind=kind, entity=entity, tag=tag)
            if q.height == 0:
                return None
            rec = q.tail(1).to_dicts()[0]
            try:
                payload = json.loads(rec["payload"]) if rec["payload"] else {}
            except Exception:
                payload = {}
            return Row(
                uuid=rec["uuid"],
                time_index=rec["time_index"],
                ts=rec["ts"],
                namespace=rec["namespace"],
                kind=rec["kind"],
                entity=rec["entity"],
                snapshot_id=rec["snapshot_id"],
                tag=rec["tag"],
                payload_type=rec["payload_type"],
                payload=PayloadRegistry.decode(rec["payload_type"], payload),
            )

        def count(self, **filters: Any) -> int:
            return int(self._filter(**filters).height)

    def reader(self) -> LedgerReader:
        return PolarsLedger._Reader(self._df)

    # ---- frame helpers (no I/O) ----
    def frame(self) -> pl.DataFrame:
        """Return a copy of the underlying Polars DataFrame."""
        return self._df.clone()

    def replace_with_frame(self, df: pl.DataFrame) -> None:
        """Replace the internal frame (schema will be normalized)."""
        for c, t in self._SCHEMA.items():
            if c not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=cast(Any, t)).alias(c))
        self._df = df.select(list(self._SCHEMA.keys()))
