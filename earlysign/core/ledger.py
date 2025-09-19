"""
Thin, backend-agnostic Ibis wrapper for an append-only event ledger.

Design goals
------------
- Keep this layer *thin*. Defer to Ibis for expressions, typing, JSON access.
- Use JSON columns for payload and labels storage.
  Users can do: ledger.t.select(ledger.t.uuid, ledger.t.payload["x"].cast("int64"))

- Methods like select/filter/limit delegate directly to Ibis (no wrappers).
- Table schema is fixed and owned by this module.

Table contract
--------------
Base table (`table_name`) has:
  - uuid: string (auto-generated)
  - ts:   timestamp("UTC") (auto-generated)
  - payload_type: string
  - labels: json
  - payload: json

Doctests
--------
>>> import ibis
>>> con = ibis.duckdb.connect(":memory:")
>>> from earlysign.core.ledger import Ledger
>>> ledger = Ledger().set_connector(con).use_default_table("events")
>>> ledger.ensure()
>>> row_id = ledger.insert(
...     payload_type="TwoProportion",
...     payload={"n_treatment": 100, "n_control": 95},
...     labels={"experiment_id": "exp1"},
... )
>>> isinstance(row_id, str)
True
>>> res = ledger.t.select(
...     ledger.t.payload["n_treatment"].cast("int64").name("n_treatment")
... ).execute()
>>> res.to_dict("records")[0]["n_treatment"]
100
"""

from dataclasses import dataclass
from typing import Any, Mapping
import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import Table as TableExpr


# ---------- Fixed schema ----------

LEDGER_SCHEMA = sch.schema(
    dict(
        uuid=dt.string,
        ts=dt.timestamp(timezone="UTC"),
        payload_type=dt.string,
        payload=dt.json,
        labels=dt.json,
    )
)


# ---------- Main Ledger facade ----------


@dataclass
class Ledger:
    connector: ibis.BaseBackend | None = None
    table_name: str = "events"

    def set_connector(self, connector: ibis.BaseBackend) -> "Ledger":
        self.connector = connector
        return self

    def use_default_table(self, name: str = "events") -> "Ledger":
        self.table_name = name
        return self

    @property
    def t(self) -> TableExpr:
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        return self.connector.table(self.table_name)

    def ensure(self) -> None:
        """Ensure the ledger table exists with the standard schema."""
        if self.connector is None:
            raise RuntimeError("Ledger connector not set")
        if self.table_name in self.connector.list_tables():
            return
        self.connector.create_table(self.table_name, schema=LEDGER_SCHEMA)

    def insert(
        self,
        *,
        payload_type: str,
        payload: Mapping[str, Any],
        labels: Mapping[str, Any] = {},
    ) -> str:
        """Insert a row into the ledger with auto-generated uuid and ts."""
        import uuid as uuidlib
        from datetime import datetime, timezone

        if self.connector is None:
            raise RuntimeError("Ledger connector not set")

        row_id = str(uuidlib.uuid4())
        ts = datetime.now(timezone.utc)

        row = {
            "uuid": row_id,
            "ts": ts,
            "payload_type": payload_type,
            "payload": dict(payload),
            "labels": dict(labels),
        }
        self.connector.insert(self.table_name, [row])
        return row_id
