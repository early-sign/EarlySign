"""
Thin, backend-agnostic Ibis wrapper for an append-only event ledger.

Design goals
------------
- Keep this layer *thin*. Defer to Ibis for expressions, typing, JSON access.
- Use JSON columns for payload and labels storage.
  Users can do: df.select(df.t.uuid, df.payload["x"].cast("int64"))

- Methods like select/filter/limit delegate directly to Ibis (no wrappers that
  re-implement Ibis behavior). We simply return TableExpr.

Table contract
--------------
Base table (`table_name`) has:
  - uuid: string
  - ts: string (ISO 8601; we keep string to avoid tz/precision drift across backends)
  - payload_type: string
  - labels: json
  - payload: json

Notes
-----
- We do *not* hand-roll JSON extract helpers. Access JSON exactly as Ibis exposes:
  df.payload["key"], df.labels["key"] (+ .cast(...) if the user wants).
- All persistence operations work directly with JSON columns in the base table.

Doctests
--------
Small unit doctests for access and delegation live in this module.
Higher-level end-to-end doctests are in ledger.py.

>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger_df import LedgerDF, LEDGER_SCHEMA
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> df = LedgerDF(connector=con, table_name="events")
>>> # base selection delegates to ibis; JSON extraction keeps Ibis API
>>> q = df.select(df.t.uuid, df.payload["x"].name("x"))
>>> out = q.execute()
>>> list(out.columns)
['uuid', 'x']

Backend-agnostic JSON access example (no custom helper):
>>> import ibis
>>> con = ibis.duckdb.connect(":memory:")  # or any other backend
>>> _ = con.create_table("ev2", schema=LEDGER_SCHEMA)
>>> df2 = LedgerDF(connector=con, table_name="ev2")
>>> q2 = df2.select(
...     df2.t.uuid,
...     # Use Ibis JSON extraction API as-is:
...     df2.payload["x"].name("x"),
... )
>>> _ = q2.execute()  # empty okay

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional

import ibis
from ibis.expr.types import Table as TableExpr


# Public base schema used by default tables: JSON for payload & labels
LEDGER_SCHEMA = ibis.schema(
    {
        "uuid": "string",
        "ts": "string",  # ISO8601 string (avoid tz portability issues)
        "payload_type": "string",
        "labels": "json",
        "payload": "json",
    }
)


# ---------- Strategies (persistence policy) ----------


def _table_names(be: ibis.BaseBackend) -> set[str]:
    """Return a set of table names across backends.

    `list_tables()` may return:
      - list[str]
      - list[dict] with a "name" key
      - list[objects] with `.name`
    We normalize to a set[str].
    """
    items = list(be.list_tables())
    names: set[str] = set()
    for it in items:
        if isinstance(it, str):
            names.add(it)
        elif isinstance(it, dict) and "name" in it:
            names.add(str(it["name"]))
        else:
            name = getattr(it, "name", None)
            if name is not None:
                names.add(str(name))
    return names


# ---------- Main DF wrapper (thin) ----------


@dataclass
class LedgerDF:
    """Thin wrapper around an Ibis table with JSON-based persistence.

    - `t` exposes the Ibis TableExpr.
    - `payload`/`labels` expose the JSON columns as-is: `df.payload["k"]`.
    - All select/filter/limit simply delegate to Ibis and return TableExpr.
    - All persistence operations work directly with JSON columns.

    Unit doctests:

    >>> import ibis, duckdb  # noqa: F401
    >>> con = ibis.duckdb.connect(":memory:")
    >>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
    >>> df = LedgerDF(connector=con, table_name="events")
    >>> isinstance(df.t, ibis.Table)
    True
    >>> q = df.select(df.t.uuid, df.payload["x"].name("x")).limit(5)
    >>> isinstance(q, ibis.Table)
    True
    >>> # Delegation: filter/select return Ibis expressions
    >>> r = df.filter(df.t.payload_type == "Any").select("uuid")
    >>> isinstance(r, ibis.Table)
    True
    """

    connector: ibis.BaseBackend
    table_name: str
    _t_cache: Optional[TableExpr] = field(default=None, repr=False)

    # ---- Core table exposure & dynamic delegation ----
    @property
    def t(self) -> TableExpr:
        """Legacy property for backward compatibility. Use direct column access instead."""
        return self._get_table()

    def _get_table(self) -> TableExpr:
        """Internal method to get the actual table expression."""
        if self._t_cache is not None:
            return self._t_cache
        base = self.connector.table(self.table_name)
        # With JSON-only approach, we always use the base table
        self._t_cache = base
        return self._t_cache

    @property
    def payload(self) -> Any:
        # Expose JSON column exactly as Ibis does
        return self._get_table().payload

    @property
    def labels(self) -> Any:
        # Expose JSON column exactly as Ibis does
        return self._get_table().labels

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes/methods to the underlying Ibis table.
        This keeps the wrapper thin and future-proof to Ibis API changes.
        """
        try:
            return getattr(self._get_table(), name)
        except AttributeError:
            raise

    def __dir__(self) -> List[str]:
        """
        Improve IDE completion by merging our attributes with Ibis table attributes.
        """
        return sorted(set(super().__dir__()) | set(dir(self._get_table())))

    # Persistence operations -----------------------------------
    def ensure(self) -> None:
        """Create base table if missing using LEDGER_SCHEMA."""
        names = _table_names(self.connector)
        if self.table_name not in names:
            self.connector.create_table(self.table_name, schema=LEDGER_SCHEMA)

    def append_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Insert rows directly into base table."""
        rows_list = list(rows)
        if not rows_list:
            return

        # Use getattr to access insert() method that may vary by backend
        insert_method = getattr(self.connector, "insert", None)
        if insert_method:
            insert_method(self.table_name, rows_list)
        else:
            # Fallback for backends without insert method
            raise NotImplementedError(
                f"Backend {type(self.connector)} doesn't support insert()"
            )

        # Invalidate cache since table contents changed
        self._t_cache = None

    def replace_all(self, rows: Iterable[Mapping[str, Any]]) -> int:
        """Truncate table and insert new rows."""
        # Use getattr to access raw_sql() method that may vary by backend
        raw_sql_method = getattr(self.connector, "raw_sql", None)
        if raw_sql_method:
            raw_sql_method(f'DELETE FROM "{self.table_name}"')
        else:
            raise NotImplementedError(
                f"Backend {type(self.connector)} doesn't support raw_sql()"
            )

        data = list(rows)
        if data:
            self.append_rows(data)
        self._t_cache = None
        return 0

    def upsert_rows(
        self, rows: Iterable[Mapping[str, Any]], keys: List[str]
    ) -> Dict[str, Any]:
        """Naive uuid-based upsert: delete matching uuids then insert."""
        # We only implement naive uuid-based upsert; `keys` kept for compatibility/logging
        data = list(rows)
        if not data:
            return {"deleted": 0, "inserted": 0, "updated": 0, "upserted": 0}

        uuids = [r["uuid"] for r in data if "uuid" in r]
        if uuids:
            inlist = ",".join(f"'{u}'" for u in uuids)
            raw_sql_method = getattr(self.connector, "raw_sql", None)
            if raw_sql_method:
                raw_sql_method(
                    f'DELETE FROM "{self.table_name}" WHERE uuid IN ({inlist})'
                )
            else:
                raise NotImplementedError(
                    f"Backend {type(self.connector)} doesn't support raw_sql()"
                )

        self.append_rows(data)
        inserted = len(data)
        updated = 0
        self._t_cache = None
        return {
            "deleted": len(uuids),
            "inserted": inserted,
            "updated": updated,
            "upserted": inserted + updated,
        }
