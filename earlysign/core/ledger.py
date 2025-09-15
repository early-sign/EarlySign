"""
High level Ledger facade over LedgerDF.

- Keeps a *thin* surface and delegates all query ops to Ibis via LedgerDF.
- Default persistence is JsonStrategy; can be switched to TypedStrategy.
- Handlers (LedgerDataHandler) define typed payloads; in TypedStrategy we
  materialize them and auto-join so that JSON access pattern stays identical.

Doctests include:
  * Unit-level: handler creation, basic write paths
  * End-to-end usage example mirroring your workflow expectations

>>> import ibis, duckdb  # noqa: F401
>>> from typing import TypedDict
>>> from earlysign.core.ledger_df import (
...     LEDGER_SCHEMA, LedgerDF, LedgerDataHandler, JsonStrategy, TypedStrategy
... )

# --- Basic JSON-mode roundtrip (no typed tables) ----------------------
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> L = Ledger().set_connector(con).use_default_table("events")
>>> L.ensure()  # JsonStrategy by default
>>> _ = L.insert_events([
...   dict(payload={"nA": 100, "mA": 38, "nB": 120, "mB": 51}, labels={"kind":"observation","batch":1}),
...   dict(payload={"nA":  80, "mA": 22, "nB":  90, "mB": 30}, labels={"kind":"observation","batch":2}),
... ])

# Query using Ibis JSON API (delegate end-to-end)
>>> obs = (
...   L.df
...     .select(
...       "uuid",
...       nA=L.df.payload["nA"].cast("int64"),
...       mA=L.df.payload["mA"].cast("int64"),
...       nB=L.df.payload["nB"].cast("int64"),
...       mB=L.df.payload["mB"].cast("int64"),
...     )
...     .execute()
... )
>>> set({"uuid","nA","mA","nB","mB"}) <= set(obs.columns)
True

# --- Typed handler registration + switch to TypedStrategy --------------
>>> class TwoPropObsBatch(TypedDict):
...     nA: int; nB: int; mA: int; mB: int
...
>>> handler = LedgerDataHandler.from_typeddict("TwoPropObsBatch", TwoPropObsBatch)
>>> _ = L.register_handler(handler)
>>> _ = L.set_strategy(TypedStrategy())
>>> L.ensure()  # creates typed table alongside base
>>> _ = L.insert_event(
...     payload_type="TwoPropObsBatch",
...     payload={"nA":150,"mA":60,"nB":140,"mB":48},
...     labels={"kind":"observation","batch":3}
... )

# JSON access keeps working the same way (joined or JSON)
>>> q2 = (
...   L.df
...     .filter(L.df.t.payload_type == "TwoPropObsBatch")
...     .select(n_treat=L.df.payload["nA"].cast("int64"))
... )
>>> rows = q2.execute().to_dict("records")
>>> rows[0]["n_treat"]
150

# --- Minimal "compat" scenario (namespaces & arbitrary labels) --------
>>> from enum import Enum
>>> class Namespace(str, Enum):
...     OBS = "obs"; STATS = "stats"; CRITERIA = "criteria"; SIGNALS = "signals"; DESIGN = "design"
...
>>> conn = ibis.duckdb.connect(":memory:")
>>> _ = conn.create_table("ledger_compat", schema=LEDGER_SCHEMA)
>>> ledger = Ledger().set_connector(conn).use_default_table("ledger_compat")
>>> ledger.ensure()
>>> _ = ledger.insert_event(
...     payload_type="TwoProportion",
...     payload={"n_treatment": 100, "n_control": 95},
...     labels={
...         "namespace": Namespace.OBS.value,
...         "kind": "observation",
...         "experiment_id": "exp1",
...         "step_key": "s1",
...         "tag": "demo",
...     },
... )
>>> query = ledger.df.filter(ledger.df.t.payload_type == "TwoProportion")
>>> results = query.execute()
>>> len(results) >= 1
True
>>> qn = (
...   ledger.df
...     .filter(ledger.df.t.payload_type == "TwoProportion")
...     .select(n_treatment=ledger.df.payload["n_treatment"].cast("int64"))
... )
>>> r2 = qn.execute().to_dict("records")
>>> r2[0]["n_treatment"]
100
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Literal

import ibis
from ibis.expr.types import Table as TableExpr

from earlysign.core.ledger_df import (
    LEDGER_SCHEMA,
    LedgerDF,
    LedgerDataHandler,
    JsonStrategy,
    TypedStrategy,
)


@dataclass
class Ledger:
    """High-level convenience API over LedgerDF.

    - Keeps default JsonStrategy until caller switches.
    - Delegates all read/query operations to Ibis via `df` (LedgerDF).
    - Provides light write/save helpers.
    """

    connector: Optional[ibis.BaseBackend] = None
    table_name: str = "events"
    _df: Optional[LedgerDF] = field(default=None, repr=False)

    # Setup --------------------------------------------------------------
    def set_connector(self, connector: ibis.BaseBackend) -> "Ledger":
        self.connector = connector
        self._df = None  # rebuild on next access
        return self

    def use_default_table(self, name: str) -> "Ledger":
        self.table_name = name
        self._df = None
        return self

    @property
    def df(self) -> LedgerDF:
        """Build a LedgerDF lazily; default JsonStrategy unless changed later."""
        if not self.connector:
            raise ValueError("Connector not set")
        if self._df is None:
            self._df = LedgerDF(self.connector, self.table_name)
            self._df.set_strategy(JsonStrategy())
        return self._df

    # Strategy / handler configuration ----------------------------------
    def set_strategy(self, strategy) -> "Ledger":
        self.df.set_strategy(strategy)
        return self

    def register_handler(self, handler: LedgerDataHandler) -> "Ledger":
        self.df.register_handler(handler)
        return self

    def set_handlers(self, handlers: Mapping[str, LedgerDataHandler]) -> "Ledger":
        for h in handlers.values():
            self.register_handler(h)
        return self

    # Internals ----------------------------------------------------------
    def _now(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    def _normalize(self, e: Mapping[str, Any]) -> dict:
        """Normalize a user event to the base row shape. No dtype decisions here."""
        import uuid as _uuid

        ts = e.get("ts") or self._now()
        return {
            "uuid": e.get("uuid") or _uuid.uuid4().hex,
            "ts": ts,
            "payload_type": e.get("payload_type", "json"),
            "labels": dict(e.get("labels") or {}),
            "payload": dict(e.get("payload") or {}),
        }

    # Write API ----------------------------------------------------------
    def ensure(self) -> None:
        """Ensure base (and typed) tables exist per the active strategy."""
        self.df.ensure()

    def insert_event(
        self,
        *,
        payload_type: str = "json",
        payload: Mapping[str, Any] | None = None,
        labels: Mapping[str, Any] | None = None,
        ts: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> str:
        row = self._normalize(
            {
                "uuid": uuid,
                "ts": ts,
                "payload_type": payload_type,
                "payload": payload or {},
                "labels": labels or {},
            }
        )
        self.ensure()
        self.df.append_rows([row])
        return row["uuid"]

    def insert_events(self, events: List[Mapping[str, Any]]) -> Dict[str, Any]:
        rows = [self._normalize(e) for e in events]
        self.ensure()
        self.df.append_rows(rows)
        return {
            "inserted": len(rows),
            "failed": 0,
            "errors": [],
            "uuids": [r["uuid"] for r in rows],
        }

    def save(
        self,
        records: Optional[Iterable[Mapping[str, Any]]] | None = None,
        *,
        mode: Literal["append", "replace", "upsert"] = "append",
        match_keys: Optional[List[str]] = None,
        dest_connector: Optional[ibis.BaseBackend] = None,
        dest_strategy: Optional[Any] = None,
        dest_table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save the current view (or provided records) into a destination.

        If dest_* are omitted, saves back into this ledger.
        We do not choose types here; we only hand rows to the destination DF/strategy.
        """
        # Materialize source rows
        if records is None:
            cur = self.df.t.execute()
            recs = cur.to_dict("records") if hasattr(cur, "to_dict") else list(cur)
        else:
            recs = list(records)
        rows = [self._normalize(r) for r in recs]

        # Destination selection
        if dest_connector or dest_table_name or dest_strategy:
            dest_con = dest_connector or self.connector
            if not dest_con:
                raise ValueError("Destination connector not set")
            dest_tbl = dest_table_name or self.table_name
            dest_df = LedgerDF(dest_con, dest_tbl)
            if dest_strategy is not None:
                dest_df.set_strategy(dest_strategy)
            else:
                # Mirror current style (typed or json) without guessing types
                if isinstance(self.df._strategy, TypedStrategy):  # type: ignore[attr-defined]
                    dest_df.set_strategy(TypedStrategy())
                else:
                    dest_df.set_strategy(JsonStrategy())
            for h in self.df._handlers.values():  # type: ignore[attr-defined]
                dest_df.register_handler(h)
            dest_df.ensure()
            target = dest_df
        else:
            self.ensure()
            target = self.df

        # Execute write
        if mode == "append":
            target.append_rows(rows)
            return {"written": len(rows), "replaced": 0, "upserted": 0, "errors": []}
        if mode == "replace":
            prev = target.replace_all(rows)
            return {"written": len(rows), "replaced": prev, "upserted": 0, "errors": []}
        if mode == "upsert":
            keys = match_keys or ["uuid"]  # reserved for future smarter policies
            _ = keys  # keep signature stable; implementation is naive by uuid
            res = target.upsert_rows(rows, keys)
            return {
                "written": len(rows),
                "replaced": 0,
                "upserted": res.get("inserted", 0) + res.get("updated", 0),
                "errors": [],
            }
        raise ValueError("mode must be append|replace|upsert")
