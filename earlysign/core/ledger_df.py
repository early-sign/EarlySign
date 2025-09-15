"""
Thin, backend-agnostic Ibis wrapper for an append-only event ledger.

Design goals
------------
- Keep this layer *thin*. Defer to Ibis for expressions, typing, JSON access.
- Support two persistence strategies:
  1) JsonStrategy: base table stores `payload` and `labels` as JSON columns.
     Users can do: df.select(df.t.uuid, df.payload["x"].cast("int64"))
  2) TypedStrategy: in addition to base table, materialize typed side tables
     from user-provided schemas/TypedDicts and JOIN them automatically so that
     the *same* access pattern works (df.payload["x"]) regardless of layout.

- Methods like select/filter/limit delegate directly to Ibis (no wrappers that
  re-implement Ibis behavior). We simply return TableExpr.

Table contract
--------------
Base table (`table_name`) has at least:
  - uuid: string
  - ts: string (ISO 8601; we keep string to avoid tz/precision drift across backends)
  - payload_type: string
  - labels: json
  - payload: json

Notes
-----
- We do *not* hand-roll JSON extract helpers. Access JSON exactly as Ibis exposes:
  df.payload["key"], df.labels["key"] (+ .cast(...) if the user wants).
- Strategies own persistence details (ensure/append/replace/upsert/materialize).
- Handlers declare typed schemas and (optionally) target table names.

Doctests
--------
Small unit doctests for access and delegation live in this module.
Higher-level end-to-end doctests are in ledger.py.

>>> import ibis, duckdb  # noqa: F401
>>> from earlysign.core.ledger_df import LedgerDF, LEDGER_SCHEMA, JsonStrategy
>>> con = ibis.duckdb.connect(":memory:")
>>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
>>> df = LedgerDF(connector=con, table_name="events").set_strategy(JsonStrategy())
>>> # base selection delegates to ibis; JSON extraction keeps Ibis API
>>> q = df.select(df.t.uuid, df.payload["x"].name("x"))
>>> out = q.execute()
>>> list(out.columns)
['uuid', 'x']

Backend-agnostic JSON access example (no custom helper):
>>> import ibis
>>> con = ibis.duckdb.connect(":memory:")  # or any other backend
>>> _ = con.create_table("ev2", schema=LEDGER_SCHEMA)
>>> df2 = LedgerDF(connector=con, table_name="ev2").set_strategy(JsonStrategy())
>>> q2 = df2.select(
...     df2.t.uuid,
...     # Use Ibis JSON extraction API as-is:
...     df2.payload["x"].name("x"),
... )
>>> _ = q2.execute()  # empty okay

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, get_type_hints

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


# ---------- Handlers (describe typed payloads) ----------

class LedgerDataHandler:
    """Describe a typed payload and (optionally) an external typed table.

    This class does not decide types; it simply carries what the caller gives us.
    Ibis/backends remain the single source of truth for dtypes and JSON behavior.
    """

    def __init__(
        self,
        name: str,
        schema: ibis.Schema,
        payload_cols: List[str],
        *,
        labels_cols: Optional[List[str]] = None,
        typed_table_name: Optional[str] = None,
        fk_name: str = "event_uuid",
    ) -> None:
        self.name = name
        self._schema = schema
        self._payload_cols = payload_cols
        self._labels_cols = labels_cols or []
        self._typed_table_name = typed_table_name
        self._fk_name = fk_name

    def type_name(self) -> str:
        return self.name

    def typed_table_name(self, base_table_name: str) -> str:
        # Deterministic default if not explicitly given
        return self._typed_table_name or f"{base_table_name}__{self.name}"

    def typed_schema(self) -> ibis.Schema:
        return self._schema

    def payload_fields(self) -> List[str]:
        return list(self._payload_cols)

    def join_to_base(self, base: TableExpr, typed: TableExpr) -> TableExpr:
        # Thin default: inner join on FK; strategies may choose a different join
        return base.join(typed, base.uuid == typed[self._fk_name])  # type: ignore[index]

    @classmethod
    def from_schema(
        cls,
        name: str,
        schema: ibis.Schema,
        payload_cols: List[str],
        *,
        typed_table_name: Optional[str] = None,
        fk_name: str = "event_uuid",
    ) -> "LedgerDataHandler":
        return cls(
            name=name,
            schema=schema,
            payload_cols=payload_cols,
            labels_cols=[],
            typed_table_name=typed_table_name,
            fk_name=fk_name,
        )

    @classmethod
    def from_typeddict(
        cls,
        name: str,
        typed_dict: Type,
        *,
        typed_table_name: Optional[str] = None,
        fk_name: str = "event_uuid",
    ) -> "LedgerDataHandler":
        hints: Dict[str, Any] = get_type_hints(typed_dict, include_extras=True)
        # Prepend FK for joins; we keep it simple: FK is string uuid
        schema = ibis.schema({fk_name: "string", **{k: v for k, v in hints.items()}})
        payload_cols = list(hints.keys())
        return cls(
            name=name,
            schema=schema,
            payload_cols=payload_cols,
            labels_cols=[],
            typed_table_name=typed_table_name,
            fk_name=fk_name,
        )

    def __repr__(self) -> str:
        return f"LedgerDataHandler(name={self.name!r}, fields={self._payload_cols!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LedgerDataHandler):
            return False
        return (
            self.name == other.name
            and self._schema == other._schema
            and self._payload_cols == other._payload_cols
            and self._typed_table_name == other._typed_table_name
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


class PersistStrategy:
    """Strategy base.

    A strategy *owns* how rows are persisted and how typed tables (if any) are
    ensured/updated. It must not change the user-facing Ibis interface.
    """

    def ensure(self, df: "LedgerDF") -> None:
        raise NotImplementedError

    def append(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> None:
        raise NotImplementedError

    def replace_all(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> int:
        raise NotImplementedError

    def upsert(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
        raise NotImplementedError


class JsonStrategy(PersistStrategy):
    """Persist all rows in the base table (payload/labels as JSON).

    - `ensure` creates base table if missing using LEDGER_SCHEMA.
    - `append` inserts rows directly into base.
    - `replace_all` truncates and inserts.
    - `upsert` (naive) by uuid: delete matching uuids then insert.
    """

    def ensure(self, df: "LedgerDF") -> None:
        be = df.connector
        names = _table_names(be)
        if df.table_name not in names:
            be.create_table(df.table_name, schema=LEDGER_SCHEMA)

    def _insert(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> None:
        be = df.connector
        be.insert(df.table_name, rows)

    def append(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> None:
        self._insert(df, rows)

    def replace_all(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> int:
        be = df.connector
        be.raw_sql(f'DELETE FROM "{df.table_name}"')
        data = list(rows)
        if data:
            be.insert(df.table_name, data)
        return 0

    def upsert(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
        data = list(rows)
        if not data:
            return {"deleted": 0, "inserted": 0, "updated": 0}
        be = df.connector
        uuids = [r["uuid"] for r in data if "uuid" in r]
        if uuids:
            inlist = ",".join(f"'{u}'" for u in uuids)
            be.raw_sql(f'DELETE FROM "{df.table_name}" WHERE uuid IN ({inlist})')
        be.insert(df.table_name, data)
        return {"deleted": len(uuids), "inserted": len(data), "updated": 0}


class TypedStrategy(JsonStrategy):
    """Json + typed side tables.

    - Base table is the same JSON table as JsonStrategy (so JSON access always works).
    - Additionally, for every registered handler:
        * ensure() creates the typed table if missing
        * append()/replace_all()/upsert() maintain typed tables (append-only demo)
    - `LedgerDF.t` automatically joins typed tables so that
        df.payload["field"] works identically (via JSON or via joined columns).
      (We do not rewrite expressions: we only JOIN typed columns; JSON access remains).
    """

    def ensure(self, df: "LedgerDF") -> None:
        super().ensure(df)
        be = df.connector
        names = _table_names(be)
        for h in df._handlers.values():
            tname = h.typed_table_name(df.table_name)
            if tname not in names:
                be.create_table(tname, schema=h.typed_schema())

    def append(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        # Write into base
        super().append(df, rows)
        # Split and write to typed tables
        be = df.connector
        for h in df._handlers.values():
            tname = h.typed_table_name(df.table_name)
            payload_cols = h.payload_fields()
            typed_rows = []
            for r in rows:
                if r.get("payload_type") == h.type_name():
                    pl = dict(r.get("payload") or {})
                    # Build row for typed table: FK + payload cols (missing -> None)
                    out = {"event_uuid": r.get("uuid")}
                    out.update({k: pl.get(k) for k in payload_cols})
                    typed_rows.append(out)
            if typed_rows:
                be.insert(tname, typed_rows)

    def replace_all(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> int:
        be = df.connector
        be.raw_sql(f'DELETE FROM "{df.table_name}"')
        for h in df._handlers.values():
            be.raw_sql(f'DELETE FROM "{h.typed_table_name(df.table_name)}"')
        data = list(rows)
        if data:
            self.append(df, data)
        return 0

    def upsert(self, df: "LedgerDF", rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
        data = list(rows)
        if not data:
            return {"deleted": 0, "inserted": 0, "updated": 0}
        be = df.connector
        uuids = [r["uuid"] for r in data if "uuid" in r]
        if uuids:
            inlist = ",".join(f"'{u}'" for u in uuids)
            be.raw_sql(f'DELETE FROM "{df.table_name}" WHERE uuid IN ({inlist})')
            for h in df._handlers.values():
                be.raw_sql(
                    f'DELETE FROM "{h.typed_table_name(df.table_name)}" WHERE event_uuid IN ({inlist})'
                )
        self.append(df, data)
        return {"deleted": len(uuids), "inserted": len(data), "updated": 0}


# ---------- Main DF wrapper (thin) ----------

@dataclass
class LedgerDF:
    """Thin wrapper around an Ibis table.

    - `t` exposes the Ibis TableExpr (possibly joined in TypedStrategy).
    - `payload`/`labels` expose the JSON columns as-is: `df.payload["k"]`.
    - All select/filter/limit simply delegate to Ibis and return TableExpr.
    - Persistence is delegated to a configured strategy (JsonStrategy is default).

    Unit doctests:

    >>> import ibis, duckdb  # noqa: F401
    >>> con = ibis.duckdb.connect(":memory:")
    >>> _ = con.create_table("events", schema=LEDGER_SCHEMA)
    >>> df = LedgerDF(connector=con, table_name="events").set_strategy(JsonStrategy())
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
    _strategy: Optional[PersistStrategy] = field(default=None, repr=False)
    _handlers: Dict[str, LedgerDataHandler] = field(default_factory=dict, repr=False)
    _t_cache: Optional[TableExpr] = field(default=None, repr=False)

    # Strategy/handlers -------------------------------------------------
    def set_strategy(self, strategy: PersistStrategy) -> "LedgerDF":
        self._strategy = strategy
        # Invalidate cached table view; join shape might change
        self._t_cache = None
        return self

    def register_handler(self, handler: LedgerDataHandler) -> "LedgerDF":
        self._handlers[handler.type_name()] = handler
        # View can change (extra joins), invalidate cache
        self._t_cache = None
        return self

    # Core table exposure -----------------------------------------------
    @property
    def t(self) -> TableExpr:
        if self._t_cache is not None:
            return self._t_cache
        base = self.connector.table(self.table_name)
        # In TypedStrategy, join side tables so extra columns are available.
        if isinstance(self._strategy, TypedStrategy) and self._handlers:
            for h in self._handlers.values():
                tname = h.typed_table_name(self.table_name)
                # Be tolerant if the typed table is not created yet.
                try:
                    typed_tbl = self.connector.table(tname)
                except Exception:
                    # Skip joining when the typed table is not available yet.
                    continue
                base = h.join_to_base(base, typed_tbl)
        self._t_cache = base
        return self._t_cache

    # Convenience accessors to match Ibis JSON API (no custom JSON helper)
    @property
    def payload(self):
        return self.t.payload  # type: ignore[attr-defined]

    @property
    def labels(self):
        return self.t.labels  # type: ignore[attr-defined]

    # Delegations to Ibis (return TableExpr directly) -------------------
    def select(self, *args, **kwargs) -> TableExpr:
        return self.t.select(*args, **kwargs)

    def filter(self, *preds) -> TableExpr:
        return self.t.filter(*preds)

    def limit(self, n: int) -> TableExpr:
        return self.t.limit(n)

    def expr(self) -> TableExpr:
        """Return the underlying Ibis expression unmodified."""
        return self.t

    # Persistence (owned by strategy) -----------------------------------
    def ensure(self) -> None:
        (self._strategy or JsonStrategy()).ensure(self)

    def append_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return
        (self._strategy or JsonStrategy()).append(self, rows_list)
        # Invalidate cache since table contents changed (not strictly required)
        self._t_cache = None

    def replace_all(self, rows: Iterable[Mapping[str, Any]]) -> int:
        n = (self._strategy or JsonStrategy()).replace_all(self, rows)
        self._t_cache = None
        return n

    def upsert_rows(self, rows: Iterable[Mapping[str, Any]], keys: List[str]) -> Dict[str, Any]:
        # We only implement naive uuid-based upsert; `keys` kept for compatibility/logging
        res = (self._strategy or JsonStrategy()).upsert(self, rows)
        self._t_cache = None
        return res

    # Labels/payload convenience (no magic; still Ibis expressions) -----
    def data(self) -> TableExpr:
        """Alias for the base expression for readability."""
        return self.t

    def __getattr__(self, name: str):
        # Allow df.uuid, df.ts, etc. by forwarding to self.t
        try:
            return getattr(self.t, name)
        except AttributeError:
            raise
