"""
Single-ledger JSON DSL + Sequential Procedures (expr-only, backend-stable)
=========================================================================

Deliverables
------------
(A) Two-proportions group-sequential test (design precommitted; Z/boundary as Ibis expressions)
(B) Normal-mean E-process anomaly detector (mixture over a θ-grid; expressions only)

Key idea
--------
- NEVER call JSON functions from Ibis. Instead, use one backend SQL view
  `_real_flat_view(...)` that projects JSON keys into typed columns with DuckDB
  `json_extract/json_extract_string`. All "committed" reads use this view.
- During `update(...)`, compute "previous totals" from the real view (committed rows)
  and ADD the current payload values as Ibis literals. This avoids any Ibis-level
  JSON dependency and keeps the DSL payload-agnostic.

Ledger schema
-------------
    CREATE TABLE ledger (
      exp_id   TEXT,
      ts       TIMESTAMP,
      kind     TEXT,
      payload  JSON
    );

--------------------
Doctest (DuckDB/Ibis)
--------------------
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

### (A) Two-proportions group-sequential test
>>> ab = BinomialABTest(con, "exp_ab")
>>> ab.set_design(max_n=1000, looks=[0.5, 1.0])
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> # No decision yet
>>> con.execute(con.table("ledger")
...   .filter(lambda r: (r.exp_id == "exp_ab") & (r.kind == "decision"))
...   .count())
0
>>> # Cross first planned look
>>> ab.update({"nA": 150, "mA": 15, "nB": 150, "mB": 18})
>>> qA = (
...   _real_flat_view(con, "ledger", "exp_ab")
...     .filter(lambda r: r.kind == "decision")
...     .order_by("ts")
...     .limit(1)
... )
>>> dfA = con.execute(qA.select(
...   qA.look.name("look"),
...   qA.planned_t.name("planned_t"),
...   qA.info_time.name("info_time"),
...   qA.action.name("action"),
... ))
>>> rowA = dfA.to_dict("records")[0]
>>> (rowA["look"], round(rowA["planned_t"], 3), round(rowA["info_time"], 3), rowA["action"])  # doctest: +ELLIPSIS
(1, 0.5, 0.5, 'continue')

### (B) Normal E-process (θ-grid mixture) on the same ledger
>>> ep = ENormalMixture(con, "exp_e")
>>> ep.set_design(alpha=0.05, thetas=[0.25, 0.5, 0.75])  # uniform weights 1/K
>>> for _ in range(12):
...     ep.update(x=0.8)
>>> qE = (
...   _real_flat_view(con, "ledger", "exp_e")
...     .filter(lambda r: r.kind == "e_state")
...     .order_by(lambda r: r.ts.desc())
...     .limit(1)
... )
>>> dfE = con.execute(qE.select(
...   qE.e_value.name("e_value"),
...   qE.alarm.name("alarm"),
... ))
>>> rowE = dfE.to_dict("records")[0]
>>> (round(rowE["e_value"], 2), bool(rowE["alarm"]))
(26.84, True)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union
import ibis


# ============================================================================
# Utilities
# ============================================================================

def _now_ts() -> ibis.Expr:
    """Consistent timestamp precision."""
    return ibis.now().cast("timestamp(6)")


def _json_from_kv(items: Dict[str, ibis.Expr]) -> ibis.Expr:
    """Construct a JSON object purely from Ibis expressions (no Python JSON)."""
    if not items:
        return ibis.literal("{}").cast("json")
    parts: List[ibis.Expr] = []
    for k, v in items.items():
        key = ibis.literal(f'"{k}":')
        val = (
            ibis.literal('"') + v.cast("string") + ibis.literal('"')
            if v.type().is_string() else v.cast("string")
        )
        parts.append(key + val)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + ibis.literal(",") + p
    return (ibis.literal("{") + combined + ibis.literal("}")).cast("json")


def _real_flat_view(con: ibis.Client, ledger: str, exp_id: str) -> ibis.Expr:
    """
    Project JSON payload into typed columns using DuckDB SQL json_extract*.

    We include the columns used by both procedures:
      - A/B fields: nA, mA, nB, mB, planned_max_n, planned_t, look, info_time, z, boundary, action
      - E-process fields: x, alpha, theta, e_value, alarm
    Missing fields for a given row come out as NULLs.
    """
    sql = f"""
        SELECT
          exp_id,
          ts::TIMESTAMP(6) AS ts,
          kind,
          payload,

          -- A/B test fields
          TRY_CAST(json_extract(payload, '$.nA')            AS DOUBLE)   AS nA,
          TRY_CAST(json_extract(payload, '$.mA')            AS DOUBLE)   AS mA,
          TRY_CAST(json_extract(payload, '$.nB')            AS DOUBLE)   AS nB,
          TRY_CAST(json_extract(payload, '$.mB')            AS DOUBLE)   AS mB,
          TRY_CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE)   AS planned_max_n,
          TRY_CAST(json_extract(payload, '$.planned_t')     AS DOUBLE)   AS planned_t,
          TRY_CAST(json_extract(payload, '$.look')          AS BIGINT)   AS look,
          TRY_CAST(json_extract(payload, '$.info_time')     AS DOUBLE)   AS info_time,
          TRY_CAST(json_extract(payload, '$.z')             AS DOUBLE)   AS z,
          TRY_CAST(json_extract(payload, '$.boundary')      AS DOUBLE)   AS boundary,
          json_extract_string(payload, '$.action')                      AS action,

          -- E-process fields
          TRY_CAST(json_extract(payload, '$.x')             AS DOUBLE)   AS x,
          TRY_CAST(json_extract(payload, '$.alpha')         AS DOUBLE)   AS alpha,
          TRY_CAST(json_extract(payload, '$.theta')         AS DOUBLE)   AS theta,
          TRY_CAST(json_extract(payload, '$.e_value')       AS DOUBLE)   AS e_value,
          TRY_CAST(json_extract(payload, '$.alarm')         AS BOOLEAN)  AS alarm

        FROM {ledger}
        WHERE exp_id = '{exp_id}'
    """
    return con.sql(sql)


# ============================================================================
# Staging (payload-agnostic)
# ============================================================================

@dataclass
class _Staged:
    rows: List[ibis.Expr] = field(default_factory=list)

    def add(self, row: ibis.Expr) -> None:
        self.rows.append(row)

    def union(self) -> Optional[ibis.Expr]:
        if not self.rows:
            return None
        out = self.rows[0]
        for t in self.rows[1:]:
            out = out.union(t)
        return out


@dataclass
class LedgerSession:
    """Transaction-local staging for (exp_id, ts, kind, payload(JSON)) only."""
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
        """Stage one JSON row with literal payload values (schema-free)."""
        ts = _now_ts() if ts is None else ts
        anchor = ibis.memtable([{"one": 1}])
        row = anchor.select(
            exp_id = ibis.literal(self.exp_id),
            ts     = ts.cast("timestamp(6)"),
            kind   = ibis.literal(kind),
            payload= _json_from_kv({k: ibis.literal(v) for k, v in payload.items()}),
        )
        self.staged.add(row)

    def write_from(self, src: ibis.Expr, kind: str,
                   payload: Dict[str, Union[int, float, str, ibis.Expr]],
                   *, ts: ibis.Expr) -> None:
        """Stage a row produced from Ibis expressions."""
        row = src.select(
            exp_id = ibis.literal(self.exp_id),
            ts     = ts.cast("timestamp(6)"),
            kind   = ibis.literal(kind),
            payload= _json_from_kv({k: (v if isinstance(v, ibis.Expr) else ibis.literal(v))
                                     for k, v in payload.items()}),
        )
        self.staged.add(row)

    def view(self) -> ibis.Expr:
        """Return real ∪ staged JSON view (no typed projection here)."""
        real = self.con.table(self.ledger).filter(lambda r: r.exp_id == self.exp_id) \
                                         .select("exp_id", "ts", "kind", "payload")
        staged = self.staged.union()
        return real if staged is None else real.union(staged)

    def flush(self) -> None:
        union = self.staged.union()
        if union is not None:
            self.con.insert(self.ledger, union)

    def read(self, fn: Callable[[ibis.Expr], ibis.Expr]) -> ibis.Expr:
        return fn(self.view())


# ============================================================================
# (A) Two-proportions Group-Sequential Test
# ============================================================================

class BinomialABTest:
    """Two-proportions group-sequential test (OF-like demo boundary).

    Contract:
      - Observations are written as payload JSON with keys "nA","mA","nB","mB".
      - The ledger remains schema-free; this class only *reads* via _real_flat_view.
    """

    def __init__(self, con: ibis.Client, exp_id: str, ledger: str = "ledger"):
        self.con, self.exp_id, self.ledger = con, exp_id, ledger

    def set_design(self, *, max_n: int, looks: Sequence[float]) -> None:
        """Commit design rows immediately."""
        ts0 = _now_ts()
        anchor = ibis.memtable([{"one": 1}])

        maxn = anchor.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts0,
            kind=ibis.literal("design_max_n"),
            payload=_json_from_kv({"planned_max_n": ibis.literal(int(max_n))}),
        )

        tbl = ibis.memtable([{"look": i+1, "planned_t": float(t)} for i, t in enumerate(looks)])
        looks_rows = tbl.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts0,
            kind=ibis.literal("design_look"),
            payload=_json_from_kv({"look": tbl["look"], "planned_t": tbl["planned_t"]}),
        )

        self.con.raw_sql("BEGIN;")
        try:
            self.con.insert(self.ledger, maxn)
            self.con.insert(self.ledger, looks_rows)
            self.con.raw_sql("COMMIT;")
        except Exception:
            self.con.raw_sql("ROLLBACK;")
            raise

    def update(self, payload: Dict[str, int]) -> None:
        """One update step: write observation and derive info/stat/decision rows."""
        nA_add = float(payload.get("nA", 0))
        mA_add = float(payload.get("mA", 0))
        nB_add = float(payload.get("nB", 0))
        mB_add = float(payload.get("mB", 0))

        with LedgerSession(self.con, self.ledger, self.exp_id) as s:
            ts_now = _now_ts()

            # (1) Stage the raw observation JSON
            s.write("observation", payload, ts=ts_now)

            # (2) Read committed-only info from the real flat view
            Vreal = _real_flat_view(self.con, self.ledger, self.exp_id)

            # Latest design (Nmax and next planned look)
            design_ts = Vreal.filter(lambda r: r.kind.isin(("design_max_n","design_look"))) \
                             .aggregate(ts_max=Vreal.ts.max())

            Dmax = Vreal.filter(lambda r: r.kind == "design_max_n") \
                        .join(design_ts, predicates=[Vreal.ts == design_ts.ts_max]) \
                        .select(Nmax=Vreal.planned_max_n)

            Looks = Vreal.filter(lambda r: r.kind == "design_look") \
                         .join(design_ts, predicates=[Vreal.ts == design_ts.ts_max]) \
                         .select(look=Vreal.look.cast("int64"), planned_t=Vreal.planned_t)

            Dec = Vreal.filter(lambda r: r.kind == "decision")
            next_look_tbl = Dec.aggregate(next_look=(Dec.kind.count() + 1))

            planned = Looks.join(next_look_tbl, predicates=[Looks.look == next_look_tbl.next_look]) \
                           .select(look=Looks.look, planned_t=Looks.planned_t)

            # Previous cumulative observations (before this update)
            Obs_prev = Vreal.filter(lambda r: r.kind == "observation")
            agg_prev = Obs_prev.aggregate(
                nA=Obs_prev.nA.sum().fill_null(0.0),
                mA=Obs_prev.mA.sum().fill_null(0.0),
                nB=Obs_prev.nB.sum().fill_null(0.0),
                mB=Obs_prev.mB.sum().fill_null(0.0),
            )

            # Build a one-row table of literals for current additions
            add_tbl = ibis.memtable([{
                "nA_add": nA_add, "mA_add": mA_add, "nB_add": nB_add, "mB_add": mB_add
            }])

            # Base after current update = previous totals + additions
            base = agg_prev.cross_join(add_tbl).cross_join(Dmax).cross_join(planned).select(
                nA = (agg_prev.nA + add_tbl.nA_add),
                mA = (agg_prev.mA + add_tbl.mA_add),
                nB = (agg_prev.nB + add_tbl.nB_add),
                mB = (agg_prev.mB + add_tbl.mB_add),
                Nmax = Dmax.Nmax,
                look = planned.look,
                planned_t = planned.planned_t,
            )

            # Information time before and after this update
            I0_tbl = agg_prev.cross_join(Dmax).select(
                ((agg_prev.nA + agg_prev.nB) / Dmax.Nmax.nullif(0)).name("I0")
            )
            X = base.cross_join(I0_tbl).select(
                nA=base.nA, mA=base.mA, nB=base.nB, mB=base.mB,
                Nmax=base.Nmax, look=base.look, planned_t=base.planned_t, I0=I0_tbl.I0
            )
            I_expr = (X.nA + X.nB) / X.Nmax.nullif(0)

            # Z-statistic (B - A, pooled SE) and OF-like boundary
            p_expr = (X.mA + X.mB) / (X.nA + X.nB)
            se_expr = (p_expr * (1 - p_expr) * (1 / X.nA + 1 / X.nB)).sqrt().nullif(0)
            z_expr = ((X.mB / X.nB) - (X.mA / X.nA)) / se_expr

            bnd_expr = ibis.cases(
                (I_expr <= 0.5, ibis.literal(2.963)),
                (I_expr >= 1.0, ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                      + (ibis.literal(1.96) - ibis.literal(2.963)) * (I_expr - 0.5) / 0.5,
            )

            # Due if we cross the planned information time now
            is_due = (X.I0 < X.planned_t) & (X.planned_t <= I_expr)

            # (3) Emit derived rows
            s.write_from(X, "snapshot", {
                "nA": X.nA, "mA": X.mA, "nB": X.nB, "mB": X.mB
            }, ts=ts_now)

            s.write_from(X, "stat", {"z": z_expr}, ts=ts_now)
            s.write_from(X, "info", {"info_time": I_expr}, ts=ts_now)

            due = X.filter(is_due)
            I_due = (due.nA + due.nB) / due.Nmax.nullif(0)
            p_due = (due.mA + due.mB) / (due.nA + due.nB)
            se_due = (p_due * (1 - p_due) * (1 / due.nA + 1 / due.nB)).sqrt().nullif(0)
            z_due = ((due.mB / due.nB) - (due.mA / due.nA)) / se_due
            bnd_due = ibis.cases(
                (I_due <= 0.5, ibis.literal(2.963)),
                (I_due >= 1.0, ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                      + (ibis.literal(1.96) - ibis.literal(2.963)) * (I_due - 0.5) / 0.5,
            )
            action = (z_due.abs() >= bnd_due).ifelse("stop_efficacy", "continue")

            s.write_from(due, "decision", {
                "look": due.look,
                "planned_t": due.planned_t,
                "info_time": I_due,
                "z": z_due,
                "boundary": bnd_due,
                "action": action,
            }, ts=ts_now)


# ============================================================================
# (B) Normal-mixture E-process (expr-only; committed-read strategy)
# ============================================================================

class ENormalMixture:
    """
    E-process for detecting mean shifts against i.i.d. N(0,1) null using a θ-grid mixture.

    Design rows (committed once):
      - kind='e_design' : {"alpha": ...}
      - kind='e_theta'  : {"theta": ...}  (multiple rows; uniform weights 1/K)

    Update step:
      - Stage kind='e_obs' with {"x": ...}
      - Read previous S,t, alpha, theta from the committed real flat view
      - Compute the new mixture E-value with the literal x added
      - Emit kind='e_state' with {"e_value": ..., "alarm": bool}
    """

    def __init__(self, con: ibis.Client, exp_id: str, ledger: str = "ledger"):
        self.con, self.exp_id, self.ledger = con, exp_id, ledger

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        ts0 = _now_ts()
        anchor = ibis.memtable([{"one": 1}])

        d = anchor.select(
            exp_id=ibis.literal(self.exp_id),
            ts=ts0,
            kind=ibis.literal("e_design"),
            payload=_json_from_kv({"alpha": ibis.literal(float(alpha))}),
        )

        self.con.raw_sql("BEGIN;")
        try:
            self.con.insert(self.ledger, d)
            for th in thetas:
                row = anchor.select(
                    exp_id=ibis.literal(self.exp_id),
                    ts=ts0,
                    kind=ibis.literal("e_theta"),
                    payload=_json_from_kv({"theta": ibis.literal(float(th))}),
                )
                self.con.insert(self.ledger, row)
            self.con.raw_sql("COMMIT;")
        except Exception:
            self.con.raw_sql("ROLLBACK;")
            raise

    def update(self, *, x: float) -> None:
        x_add = float(x)

        with LedgerSession(self.con, self.ledger, self.exp_id) as s:
            ts_now = _now_ts()

            # (1) Stage the new observation
            s.write("e_obs", {"x": x_add}, ts=ts_now)

            # (2) Read committed design and past observations from the real flat view
            Vreal = _real_flat_view(self.con, self.ledger, self.exp_id)

            # Latest alpha
            D = Vreal.filter(lambda r: r.kind == "e_design")
            Dts = D.aggregate(ts_max=D.ts.max())
            Dcur = D.join(Dts, predicates=[D.ts == Dts.ts_max]).select(alpha=D.alpha)

            # Theta grid and K
            Theta = Vreal.filter(lambda r: r.kind == "e_theta").select(theta=Vreal.theta)
            K = Theta.aggregate(k=Theta.theta.count())

            # Previous S and t (committed only)
            Eobs_prev = Vreal.filter(lambda r: r.kind == "e_obs").select(x_prev=Vreal.x)
            Agg_prev = Eobs_prev.aggregate(S_prev=Eobs_prev.x_prev.sum().fill_null(0.0),
                                           t_prev=Eobs_prev.x_prev.count())

            # Add current x (literal) to get S, t
            add_tbl = ibis.memtable([{"x_add": x_add, "one": 1}])
            Agg = Agg_prev.cross_join(add_tbl).select(
                S = Agg_prev.S_prev + add_tbl.x_add,
                t = Agg_prev.t_prev + add_tbl.one,
            )

            # Mixture E_t = (1/K) * Σ exp(theta*S - 0.5*theta^2*t)
            parts = Theta.cross_join(Agg).cross_join(K).select(
                term = ( (ibis.literal(1.0) / K.k.nullif(0))
                         * ( (Theta.theta * Agg.S) - (0.5 * Theta.theta * Theta.theta * Agg.t) ).exp() )
            )
            E = parts.aggregate(e_value = parts.term.sum())

            # Threshold 1/alpha and alarm flag
            Thr = Dcur.select(thr=(ibis.literal(1.0) / Dcur.alpha.nullif(0)))
            S_all = E.cross_join(Thr).select(
                e_value=E.e_value,
                alarm=(E.e_value >= Thr.thr)
            )

            # (3) Emit current state
            s.write_from(S_all, "e_state", {
                "e_value": S_all.e_value,
                "alarm": S_all.alarm.cast("boolean"),
            }, ts=ts_now)
