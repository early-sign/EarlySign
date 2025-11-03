"""
Single-ledger JSON DSL for group-sequential A/B testing (Ibis v10+, DuckDB)
===========================================================================

Intent
------
This module provides a **lazy, transaction-local DSL** for an event-sourced ledger
with a single `payload` JSON column. Within one transaction, the user can freely:

    write -> read -> write -> read -> ...

as if each write were immediately visible. Internally, all writes are *staged*
and a single batched INSERT occurs only at flush (on context exit), minimizing I/O
and ensuring consistency.

Conceptual DSL example (not executed)
-------------------------------------
    con = ibis.connect("duckdb://")
    with ledger_session(con, "ledger", "exp1") as s:
        # 1) Stage a write (no actual INSERT yet)
        s.write("observation", {"nA": 100, "mA": 10, "nB": 100, "mB": 12})

        # 2) Read freely from logical view (real UNION staged)
        agg = s.read(
            lambda L: L.filter(L.kind == "observation").aggregate(
                total_n=(
                    L.payload.json_extract('$.nA').cast('int64')
                    + L.payload.json_extract('$.nB').cast('int64')
                )
            )
        )

        # 3) Stage more derived rows based on intermediate results
        s.write("snapshot", {"total_n": agg.total_n})

        # 4) On context exit, all staged rows are inserted in a single flush.

Public API
----------
- set_design(max_n, looks): commits design rows immediately.
- update(payload): performs one batched logical update inside a single transaction.

Ledger schema
-------------
    CREATE TABLE ledger (
      exp_id   TEXT,
      ts       TIMESTAMP,
      kind     TEXT,
      payload  JSON
    );

End-to-end doctest
------------------
>>> import ibis
>>> con = ibis.connect("duckdb://")
>>> _ = con.raw_sql('''
... CREATE TABLE ledger (
...   exp_id   TEXT,
...   ts       TIMESTAMP,
...   kind     TEXT,
...   payload  JSON
... );
... ''')
>>> test = BinomialABTest(con, "exp1")
>>> test.set_design(max_n=1000, looks=[0.5, 1.0])
>>> test.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> con.execute(con.table("ledger")
...              .filter(lambda r: (r.exp_id == "exp1") & (r.kind == "decision"))
...              .count())
0
>>> test.update({"nA": 150, "mA": 15, "nB": 150, "mB": 18})
>>> q = (
...     _real_flat_view(con, "ledger", "exp1")
...       .filter(lambda r: r.kind == "decision")
...       .order_by("ts")
...       .limit(1)
... )
>>> df = con.execute(q.select(
...     look=q.look,
...     planned_t=q.planned_t,
...     info_time=q.info_time,
...     action=q.action,
... ))
>>> row = df.to_dict("records")[0]
>>> (row["look"], round(row["planned_t"], 3), round(row["info_time"], 3), row["action"])
(1, 0.5, 0.5, 'continue')
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union
import ibis


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now_ts() -> ibis.Expr:
    """Return TIMESTAMP(6) expression for schema consistency."""
    return ibis.now().cast("timestamp(6)")


def _json_concat_from_kv(items: Dict[str, ibis.Expr], *, string_keys: set[str]) -> ibis.Expr:
    """
    Build a JSON object via string concatenation and CAST JSON.
    Supports Ibis Expr values and avoids Python json.dumps on expressions.
    """
    if not items:
        return ibis.literal("{}").cast("json")

    parts: List[ibis.Expr] = []
    for k, v in items.items():
        key = ibis.literal(f'"{k}":')
        val = (
            ibis.literal('"') + v.cast("string") + ibis.literal('"')
            if k in string_keys else v.cast("string")
        )
        parts.append(key + val)

    combined = parts[0]
    for p in parts[1:]:
        combined = combined + ibis.literal(",") + p

    return (ibis.literal("{") + combined + ibis.literal("}")).cast("json")


def _boundary_OF_like(I: ibis.Expr) -> ibis.Expr:
    """Simple O'Brien–Fleming-like boundary for demo purposes."""
    return ibis.cases(
        (I <= 0.5, ibis.literal(2.963)),
        (I >= 1.0, ibis.literal(1.96)),
        else_=ibis.literal(2.963)
              + (ibis.literal(1.96) - ibis.literal(2.963)) * (I - 0.5) / 0.5,
    )


def _real_flat_view(con: ibis.Client, ledger: str, exp_id: str) -> ibis.Expr:
    """Project JSON payload into typed columns using DuckDB's json_extract."""
    sql = f"""
        SELECT
          exp_id,
          ts::TIMESTAMP(6) AS ts,
          kind,
          payload,
          TRY_CAST(json_extract(payload, '$.nA')            AS BIGINT) AS nA,
          TRY_CAST(json_extract(payload, '$.mA')            AS BIGINT) AS mA,
          TRY_CAST(json_extract(payload, '$.nB')            AS BIGINT) AS nB,
          TRY_CAST(json_extract(payload, '$.mB')            AS BIGINT) AS mB,
          TRY_CAST(json_extract(payload, '$.planned_max_n') AS BIGINT) AS planned_max_n,
          TRY_CAST(json_extract(payload, '$.planned_t')     AS DOUBLE) AS planned_t,
          TRY_CAST(json_extract(payload, '$.look')          AS BIGINT) AS look,
          TRY_CAST(json_extract(payload, '$.info_time')     AS DOUBLE) AS info_time,
          TRY_CAST(json_extract(payload, '$.z')             AS DOUBLE) AS z,
          TRY_CAST(json_extract(payload, '$.boundary')      AS DOUBLE) AS boundary,
          json_extract_string(payload, '$.action') AS action
        FROM {ledger}
        WHERE exp_id = '{exp_id}'
    """
    return con.sql(sql)


# ---------------------------------------------------------------------
# Staging system for the lazy DSL
# ---------------------------------------------------------------------

@dataclass
class _Staged:
    json_rows: List[ibis.Expr] = field(default_factory=list)
    flat_rows: List[ibis.Expr] = field(default_factory=list)

    def add(self, json_row: ibis.Expr, flat_row: ibis.Expr) -> None:
        self.json_rows.append(json_row)
        self.flat_rows.append(flat_row)

    def json_union(self) -> Optional[ibis.Expr]:
        if not self.json_rows:
            return None
        out = self.json_rows[0]
        for t in self.json_rows[1:]:
            out = out.union(t)
        return out

    def flat_union(self) -> Optional[ibis.Expr]:
        if not self.flat_rows:
            return None
        out = self.flat_rows[0]
        for t in self.flat_rows[1:]:
            out = out.union(t)
        return out


@dataclass
class LedgerSession:
    """Transaction-local lazy staging session."""
    con: ibis.Client
    ledger: str
    exp_id: str
    staged: _Staged = field(default_factory=_Staged)

    def __enter__(self):
        self.con.raw_sql("BEGIN;")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.flush()
                self.con.raw_sql("COMMIT;")
            else:
                self.con.raw_sql("ROLLBACK;")
        finally:
            self.staged = _Staged()

    def write(self, kind: str, payload: Dict[str, Union[int, float, str]],
              *, ts: Optional[ibis.Expr] = None) -> None:
        """Stage one JSON row with literal payload values."""
        ts = _now_ts() if ts is None else ts
        ts6 = ts.cast("timestamp(6)")
        expr_payload = {k: ibis.literal(v) for k, v in payload.items()}
        json_expr = _json_concat_from_kv(expr_payload, string_keys={"action"})

        anchor = ibis.memtable([{"one": 1}])
        json_row = anchor.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts6,
            kind=ibis.literal(kind),
            payload=json_expr,
        )

        def _lit(n, typ): return ibis.literal(payload[n]).cast(typ) if n in payload else ibis.null().cast(typ)
        flat_row = anchor.select(
            exp_id=ibis.literal(self.exp_id), ts=ts6, kind=ibis.literal(kind),
            payload=json_expr,
            nA=_lit("nA", "int64"), mA=_lit("mA", "int64"),
            nB=_lit("nB", "int64"), mB=_lit("mB", "int64"),
            planned_max_n=_lit("planned_max_n", "int64"),
            planned_t=_lit("planned_t", "float64"),
            look=_lit("look", "int64"), info_time=_lit("info_time", "float64"),
            z=_lit("z", "float64"), boundary=_lit("boundary", "float64"),
            action=_lit("action", "string"),
        )
        self.staged.add(json_row, flat_row)

    def write_from(self, source: ibis.Expr, kind: str,
                   payload: Dict[str, Union[int, float, str, ibis.Expr]],
                   *, ts: ibis.Expr) -> None:
        """Stage a row from an Ibis source with Expr payloads."""
        ts6 = ts.cast("timestamp(6)")
        expr_payload = {k: (v if isinstance(v, ibis.Expr) else ibis.literal(v)) for k, v in payload.items()}
        json_expr = _json_concat_from_kv(expr_payload, string_keys={"action"})

        json_row = source.select(
            exp_id=ibis.literal(self.exp_id), ts=ts6,
            kind=ibis.literal(kind), payload=json_expr
        )

        def _expr_or_null(n, typ):
            v = payload.get(n)
            if v is None:
                return ibis.null().cast(typ)
            return (v if isinstance(v, ibis.Expr) else ibis.literal(v)).cast(typ)

        flat_row = source.select(
            exp_id=ibis.literal(self.exp_id), ts=ts6, kind=ibis.literal(kind),
            payload=json_expr,
            nA=_expr_or_null("nA", "int64"), mA=_expr_or_null("mA", "int64"),
            nB=_expr_or_null("nB", "int64"), mB=_expr_or_null("mB", "int64"),
            planned_max_n=_expr_or_null("planned_max_n", "int64"),
            planned_t=_expr_or_null("planned_t", "float64"),
            look=_expr_or_null("look", "int64"),
            info_time=_expr_or_null("info_time", "float64"),
            z=_expr_or_null("z", "float64"),
            boundary=_expr_or_null("boundary", "float64"),
            action=_expr_or_null("action", "string"),
        )
        self.staged.add(json_row, flat_row)

    def view(self) -> ibis.Expr:
        """Return real ∪ staged view."""
        real = _real_flat_view(self.con, self.ledger, self.exp_id)
        staged = self.staged.flat_union()
        return real if staged is None else real.union(staged)

    def flush(self) -> None:
        """Insert all staged JSON rows at once."""
        json_union = self.staged.json_union()
        if json_union is not None:
            self.con.insert(self.ledger, json_union)

    def read(self, fn: Callable[[ibis.Expr], ibis.Expr]) -> ibis.Expr:
        """Utility for DSL-like interactive use."""
        return fn(self.view())


# ---------------------------------------------------------------------
# Public BinomialABTest API
# ---------------------------------------------------------------------

class BinomialABTest:
    """Group-sequential A/B test interface using the lazy ledger DSL."""

    def __init__(self, con: ibis.Client, exp_id: str, ledger: str = "ledger"):
        self.con = con
        self.exp_id = exp_id
        self.ledger = ledger

    def set_design(self, *, max_n: int, looks: Sequence[float]) -> None:
        """Commit design rows immediately (stable anchor)."""
        ts0 = _now_ts()
        anchor = ibis.memtable([{"one": 1}])

        maxn_payload = (
            ibis.literal('{"planned_max_n":')
            + ibis.literal(int(max_n)).cast("string")
            + ibis.literal("}")
        ).cast("json")

        maxn_row = anchor.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts0,
            kind=ibis.literal("design_max_n"),
            payload=maxn_payload,
        )

        looks_tbl = ibis.memtable([{"look": i + 1, "planned_t": float(t)} for i, t in enumerate(looks)])
        look_payload = (
            ibis.literal('{"look":')
            + looks_tbl["look"].cast("string")
            + ibis.literal(',"planned_t":')
            + looks_tbl["planned_t"].cast("string")
            + ibis.literal("}")
        ).cast("json")

        look_rows = looks_tbl.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts0,
            kind=ibis.literal("design_look"),
            payload=look_payload,
        )

        self.con.raw_sql("BEGIN;")
        try:
            self.con.insert(self.ledger, maxn_row)
            self.con.insert(self.ledger, look_rows)
            self.con.raw_sql("COMMIT;")
        except Exception:
            self.con.raw_sql("ROLLBACK;")
            raise

    def update(self, payload: Dict[str, int]) -> None:
        """Perform one update (observation + derived rows) inside a transaction."""
        with LedgerSession(self.con, self.ledger, self.exp_id) as s:
            ts_now = _now_ts()

            # (1) Stage observation (Python scalars => write)
            s.write("observation", payload, ts=ts_now)

            # (2) Build base from the logical view (real UNION staged)
            L = s.view().filter(lambda r: r.exp_id == self.exp_id)

            design_kinds = ("design_max_n", "design_look")
            Ld = L.filter(lambda r: r.kind.isin(design_kinds))
            design_ts = Ld.aggregate(ts_max=Ld.ts.max())

            dmax = (
                L.filter(lambda r: r.kind == "design_max_n")
                .join(design_ts, predicates=[L.ts == design_ts.ts_max])
                .select(Nmax=L.planned_max_n.cast("float64"))
            )

            obs = L.filter(lambda r: r.kind == "observation")
            agg = obs.aggregate(
                nA=obs.nA.sum().fill_null(0),
                mA=obs.mA.sum().fill_null(0),
                nB=obs.nB.sum().fill_null(0),
                mB=obs.mB.sum().fill_null(0),
            )

            next_look = L.filter(lambda r: r.kind == "decision").aggregate(
                look=(L.kind.count() + 1)
            )

            planned = (
                L.filter(lambda r: r.kind == "design_look")
                .join(design_ts, predicates=[L.ts == design_ts.ts_max])
                .join(next_look, predicates=[L.look.cast("int64") == next_look.look])
                .select(
                    look=L.look.cast("int64"),
                    planned_t=L.planned_t.cast("float64"),
                )
            )

            # Base cumulative after current staging (same root)
            base = (
                agg.cross_join(dmax)
                .cross_join(planned)
                .select(
                    nA1=agg.nA.cast("float64"),
                    mA1=agg.mA.cast("float64"),
                    nB1=agg.nB.cast("float64"),
                    mB1=agg.mB.cast("float64"),
                    Nmax=dmax.Nmax,
                    look=planned.look,
                    planned_t=planned.planned_t,
                )
            )

            # Information time BEFORE current staging, from real ledger only
            real_before = _real_flat_view(self.con, self.ledger, self.exp_id).filter(
                lambda r: r.ts < ts_now
            )
            obs_prev = real_before.filter(lambda r: r.kind == "observation")
            agg_prev = obs_prev.aggregate(
                nA=obs_prev.nA.sum().fill_null(0),
                nB=obs_prev.nB.sum().fill_null(0),
            )

            # Bring I0 as a column into the same root via cross join to avoid alias leakage
            I0_tbl = (
                agg_prev.cross_join(dmax)
                .select(
                    ((agg_prev.nA.cast("float64") + agg_prev.nB.cast("float64")) /
                    dmax.Nmax.nullif(0)).name("I0")
                )
            )
            baseX = (
                base.cross_join(I0_tbl)
                .select(
                    nA1=base.nA1, mA1=base.mA1, nB1=base.nB1, mB1=base.mB1,
                    Nmax=base.Nmax, look=base.look, planned_t=base.planned_t,
                    I0=I0_tbl.I0,
                )
            )

            # Reusable expressions on baseX (same root)
            I_expr = (baseX.nA1 + baseX.nB1) / baseX.Nmax.nullif(0)
            p_expr = (baseX.mA1 + baseX.mB1) / (baseX.nA1 + baseX.nB1)
            se_expr = (p_expr * (1 - p_expr) * (1 / baseX.nA1 + 1 / baseX.nB1)).sqrt().nullif(0)
            z_expr = ((baseX.mB1 / baseX.nB1) - (baseX.mA1 / baseX.nA1)) / se_expr
            bnd_expr = _boundary_OF_like(I_expr)

            is_due = (baseX.I0 < baseX.planned_t) & (baseX.planned_t <= I_expr)

            # (3) Stage derived rows from same root to avoid parent-mismatch
            s.write_from(
                baseX,
                "snapshot",
                {
                    "nA": baseX.nA1.cast("int64"),
                    "mA": baseX.mA1.cast("int64"),
                    "nB": baseX.nB1.cast("int64"),
                    "mB": baseX.mB1.cast("int64"),
                },
                ts=ts_now,
            )

            s.write_from(baseX, "stat", {"z": z_expr}, ts=ts_now)
            s.write_from(baseX, "info", {"info_time": I_expr}, ts=ts_now)

            # Decision rows: re-compute expressions bound to `due` (so payload uses only `due` columns)
            due = baseX.filter(is_due)
            I_due = (due.nA1 + due.nB1) / due.Nmax.nullif(0)
            p_due = (due.mA1 + due.mB1) / (due.nA1 + due.nB1)
            se_due = (p_due * (1 - p_due) * (1 / due.nA1 + 1 / due.nB1)).sqrt().nullif(0)
            z_due = ((due.mB1 / due.nB1) - (due.mA1 / due.nA1)) / se_due
            bnd_due = _boundary_OF_like(I_due)
            action_due = (z_due.abs() >= bnd_due).ifelse("stop_efficacy", "continue")

            s.write_from(
                due,
                "decision",
                {
                    "look": due.look,
                    "planned_t": due.planned_t,
                    "info_time": I_due,
                    "z": z_due,
                    "boundary": bnd_due,
                    "action": action_due,
                },
                ts=ts_now,
            )
