"""
Single-ledger, single-transaction demo (Ibis v10+) for group-sequential A/B testing.

We keep *one* ledger table with a STRUCT payload and do:
  observation -> snapshot/stat/info -> decision (if due)
all within **one database transaction**, with all derived values computed
on a single Ibis base relation to avoid parent-mismatch errors.

Doctest (end-to-end)
--------------------
>>> import ibis
>>> con = ibis.connect("duckdb://")

Create a single ledger table with a STRUCT payload (nullable fields per event type):

>>> sql = '''
... CREATE TABLE ledger (
...   exp_id         TEXT,
...   ts             TIMESTAMP,
...   kind           TEXT,   -- 'design_*' | 'observation' | 'snapshot' | 'stat' | 'info' | 'decision'
...   payload        STRUCT(
...     nA            INT,
...     mA            INT,
...     nB            INT,
...     mB            INT,
...     planned_max_n INT,
...     planned_t     DOUBLE,
...     look          INT,
...     info_time     DOUBLE,
...     z             DOUBLE,
...     boundary      DOUBLE,
...     action        TEXT,
...     design_json   JSON
...   )
... );
... '''
>>> _ = con.raw_sql(sql)

Bootstrap the test and set a design (max_n + looks) as ledger events:

>>> test = BinomialABTest(con, "exp1")
>>> test.set_design({"planned_max_n": 1000, "planned_info_times": [0.5, 1.0]})

First batch (I < 0.5) → no decision yet (still all inside one TX):

>>> test.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> con.execute(con.table("ledger")
...              .filter(lambda r: (r.exp_id == "exp1") & (r.kind == "decision"))
...              .count())
0

Second batch reaches I = 0.5 → first look triggers and a decision row is written
(again, one transaction; design/looks are read from the same ledger in that TX).

>>> test.update({"nA": 150, "mA": 15, "nB": 150, "mB": 18})
>>> q = (
...     con.table("ledger")
...       .filter(lambda r: (r.exp_id == "exp1") & (r.kind == "decision"))
...       .order_by("ts")
...       .limit(1)
... )
>>> df = con.execute(q.select(
...     look      = q.payload["look"],
...     planned_t = q.payload["planned_t"],
...     info_time = q.payload["info_time"],
...     action    = q.payload["action"],
... ))
>>> row = df.to_dict("records")[0]
>>> (row["look"], round(row["planned_t"], 3), round(row["info_time"], 3), row["action"])
(1, 0.5, 0.5, 'continue')
"""

from __future__ import annotations

from typing import Dict

import ibis

# ---------- Payload schema helpers ------------------------------------------------

PAY_TYPES: Dict[str, str] = {
    "nA": "int64",
    "mA": "int64",
    "nB": "int64",
    "mB": "int64",
    "planned_max_n": "int64",
    "planned_t": "float64",
    "look": "int64",
    "info_time": "float64",
    "z": "float64",
    "boundary": "float64",
    "action": "string",
    "design_json": "json",
}


def _now_like_ledger() -> ibis.Expr:
    # Unify to TIMESTAMP
    return ibis.now().cast("timestamp")


def _payload_nulls() -> Dict[str, ibis.Expr]:
    return {k: ibis.null().cast(t) for k, t in PAY_TYPES.items()}


def _payload_struct(**vals: ibis.Expr | int | float | str | None) -> ibis.Expr:
    # Fill missing with properly-typed NULLs; cast provided literals/exprs
    filled = _payload_nulls()
    for k, v in vals.items():
        typ = PAY_TYPES[k]
        if isinstance(v, ibis.Expr):
            filled[k] = v.cast(typ)
        elif v is None:
            filled[k] = ibis.null().cast(typ)
        else:
            filled[k] = ibis.literal(v).cast(typ)
    return ibis.struct(filled)


def _boundary_of_like(info_time: ibis.Expr) -> ibis.Expr:
    # OF-like boundary for demo: 0.5 -> 2.963, 1.0 -> 1.96 (linear between)
    return ibis.cases(
        (info_time <= 0.5, ibis.literal(2.963)),
        (info_time >= 1.0, ibis.literal(1.96)),
        else_=ibis.literal(2.963)
        + (ibis.literal(1.96) - ibis.literal(2.963)) * (info_time - 0.5) / 0.5,
    )


# ---------- Main class -----------------------------------------------------------


class BinomialABTest:
    def __init__(self, con: ibis.Client, exp_id: str):
        self.con = con
        self.exp_id = exp_id

    # -- Design writing (Ibis-only, single anchor ts) --

    def set_design(self, design: Dict[str, object]) -> None:
        """
        Write design rows: design_max_n + one design_look row per planned look.
        All rows share the same `ts` for consistent "latest design" retrieval.
        """
        planned_max_n = int(design["planned_max_n"])
        looks = [float(x) for x in design["planned_info_times"]]

        # One anchor that produces a single-row table with `ts0`
        ts_anchor = ibis.memtable([{"one": 1}]).select(ts0=_now_like_ledger())

        # design_max_n row
        maxn_row = ts_anchor.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts_anchor.ts0,
            kind=ibis.literal("design_max_n"),
            payload=_payload_struct(planned_max_n=planned_max_n),
        )

        # design_look rows
        looks_tbl = ibis.memtable(
            [{"look": i + 1, "planned_t": t} for i, t in enumerate(looks)]
        )
        looks_anchor = looks_tbl.cross_join(ts_anchor)
        look_rows = looks_anchor.select(
            exp_id=ibis.literal(self.exp_id),
            ts=looks_anchor.ts0,
            kind=ibis.literal("design_look"),
            payload=_payload_struct(
                look=looks_anchor["look"].cast("int64"),
                planned_t=looks_anchor["planned_t"].cast("float64"),
            ),
        )

        # One transaction: insert max_n + looks
        self.con.raw_sql("BEGIN;")
        try:
            self.con.insert("ledger", maxn_row)
            self.con.insert("ledger", look_rows)
            self.con.raw_sql("COMMIT;")
        except Exception:
            self.con.raw_sql("ROLLBACK;")
            raise

    # -- Update (Ibis-only, one transaction, one batched insert for derived rows) --

    def update(self, payload: Dict[str, int]) -> None:
        """
        Record a delta, compute before/after on existing observations (no double-count),
        then insert observation + derived rows in one UNION within the same TX.
        """
        nA, mA = int(payload["nA"]), int(payload["mA"])
        nB, mB = int(payload["nB"]), int(payload["mB"])

        L = self.con.table("ledger")
        self.con.raw_sql("BEGIN;")
        try:
            # --- Build base from *existing* rows only (no new insert yet) ---
            design_kinds = ("design_max_n", "design_look")

            design_ts = L.filter(
                (L.exp_id == self.exp_id) & L.kind.isin(design_kinds)
            ).aggregate(
                ts_max=L.ts.max()
            )  # 1-row table

            dmax = (
                L.filter((L.exp_id == self.exp_id) & (L.kind == "design_max_n"))
                .join(design_ts, predicates=[L.ts == design_ts.ts_max])
                .select(planned_max_n=L.payload["planned_max_n"].cast("int64"))
            )  # 1-row table

            obs = L.filter((L.exp_id == self.exp_id) & (L.kind == "observation"))
            agg_before = obs.aggregate(
                nA=obs.payload["nA"].cast("int64").sum().fill_null(0),
                mA=obs.payload["mA"].cast("int64").sum().fill_null(0),
                nB=obs.payload["nB"].cast("int64").sum().fill_null(0),
                mB=obs.payload["mB"].cast("int64").sum().fill_null(0),
            )  # 1-row table

            next_look = L.filter(
                (L.exp_id == self.exp_id) & (L.kind == "decision")
            ).aggregate(
                look=(L.kind.count() + 1)
            )  # 1-row table

            planned = (
                L.filter((L.exp_id == self.exp_id) & (L.kind == "design_look"))
                .join(design_ts, predicates=[L.ts == design_ts.ts_max])
                .join(
                    next_look,
                    predicates=[L.payload["look"].cast("int64") == next_look.look],
                )
                .select(
                    look=L.payload["look"].cast("int64"),
                    planned_t=L.payload["planned_t"].cast("float64"),
                )
            )  # 1-row table

            base = (
                agg_before.cross_join(dmax)
                .cross_join(planned)
                .select(
                    nA0=agg_before.nA.cast("float64"),
                    mA0=agg_before.mA.cast("float64"),
                    nB0=agg_before.nB.cast("float64"),
                    mB0=agg_before.mB.cast("float64"),
                    Nmax=dmax.planned_max_n.cast("float64"),
                    look=planned.look,
                    planned_t=planned.planned_t,
                )
            )

            # --- After-adding (computed on base; current delta not yet inserted) ---
            nA1 = (base.nA0 + ibis.literal(nA)).cast("float64")
            mA1 = (base.mA0 + ibis.literal(mA)).cast("float64")
            nB1 = (base.nB0 + ibis.literal(nB)).cast("float64")
            mB1 = (base.mB0 + ibis.literal(mB)).cast("float64")

            info_before = (base.nA0 + base.nB0) / base.Nmax.nullif(0)
            info_after = (nA1 + nB1) / base.Nmax.nullif(0)

            p_pool = (mA1 + mB1) / (nA1 + nB1)
            se = (p_pool * (1 - p_pool) * (1 / nA1 + 1 / nB1)).sqrt().nullif(0)
            z_after = ((mB1 / nB1) - (mA1 / nA1)) / se

            boundary = _boundary_of_like(info_after)
            is_due = (info_before < base.planned_t) & (base.planned_t <= info_after)
            action = (z_after.abs() >= boundary).ifelse("stop_efficacy", "continue")

            ts_now = _now_like_ledger()
            anchor = ibis.memtable([{"one": 1}])

            # --- Build all rows from a single plan and insert in one shot ---
            obs_row = anchor.select(
                exp_id=ibis.literal(self.exp_id),
                ts=ts_now,
                kind=ibis.literal("observation"),
                payload=_payload_struct(nA=nA, mA=mA, nB=nB, mB=mB),
            )

            snapshot_row = base.select(
                exp_id=ibis.literal(self.exp_id),
                ts=ts_now,
                kind=ibis.literal("snapshot"),
                payload=_payload_struct(
                    nA=nA1.cast("int64"),
                    mA=mA1.cast("int64"),
                    nB=nB1.cast("int64"),
                    mB=mB1.cast("int64"),
                ),
            )

            stat_row = base.select(
                exp_id=ibis.literal(self.exp_id),
                ts=ts_now,
                kind=ibis.literal("stat"),
                payload=_payload_struct(z=z_after),
            )

            info_row = base.select(
                exp_id=ibis.literal(self.exp_id),
                ts=ts_now,
                kind=ibis.literal("info"),
                payload=_payload_struct(info_time=info_after),
            )

            decision_row = base.filter(is_due).select(
                exp_id=ibis.literal(self.exp_id),
                ts=ts_now,
                kind=ibis.literal("decision"),
                payload=_payload_struct(
                    look=base.look,
                    planned_t=base.planned_t,
                    info_time=info_after,
                    z=z_after,
                    boundary=boundary,
                    action=action,
                ),
            )

            out = (
                obs_row.union(snapshot_row)
                .union(stat_row)
                .union(info_row)
                .union(decision_row)
            )
            self.con.insert("ledger", out)

            self.con.raw_sql("COMMIT;")
        except Exception:
            self.con.raw_sql("ROLLBACK;")
            raise
