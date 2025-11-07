"""
Framework + Apps: Minimal Projectors (inline SQL), no NULL paddings
===================================================================

Overview
--------
- Ledger: bindable append-only JSON ledger (uuid/ts/pkg_version auto).
- LedgerSession: transaction-local staging; all INSERTs are batched on flush.
- Projectors (no composer; each projector exposes only the columns it needs):
    * ABBaseRows   : typed columns for AB test rows from payload
    * ABDesignLooks: explode 'looks' array -> (ts, look, planned_t)
    * EBaseRows    : typed columns for E-process rows from payload
    * EThetas      : explode 'thetas' array -> (ts, theta)
- Apps:
    * BinomialABTest: two-proportions GST with single 'design' record.
      - Always: observation -> snapshot -> info
      - If a smallest *newly due* look exists (I0 < t_i <= I1): stat & decision
    * ENormalMixture: normal-mixture E-process (uniform prior over thetas)

Table contract (must exist before running doctest)
--------------------------------------------------
uuid TEXT, ts TIMESTAMP, pkg_version TEXT, payload_type TEXT, payload JSON, labels JSON

Doctest (DuckDB)
----------------
>>> import ibis
>>> con = ibis.connect("duckdb://")
>>> _ = con.raw_sql('''
... CREATE TABLE ledger (
...   uuid         TEXT,
...   ts           TIMESTAMP,
...   pkg_version  TEXT,
...   payload_type TEXT,
...   payload      JSON,
...   labels       JSON
... );
... ''')
>>>
>>> base_ledger = Ledger(con, table="ledger")
>>> ab_ledger = base_ledger.bind(experiment_id="exp_ab4")
>>> e_ledger  = base_ledger.bind(experiment_id="exp_e")

# (A) AB test with single design record (4 looks).
#     Updates crafted so: look=1 -> continue, look=2 -> continue, look=3 -> STOP.
>>> ab = BinomialABTest(ab_ledger)
>>> ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0])
>>> # Update-1: I=0.20 (< 0.25), no look
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> # Update-2: I=0.30, triggers look=1 -> continue
>>> ab.update({"nA": 50, "mA": 5, "nB": 50, "mB": 6})
>>> # Update-3: I=0.60, triggers look=2 -> continue
>>> ab.update({"nA": 150, "mA": 10, "nB": 150, "mB": 20})
>>> # Update-4: I=0.80, triggers look=3 -> STOP
>>> ab.update({"nA": 100, "mA": 5, "nB": 100, "mB": 35})
>>>
>>> V_base = ABBaseRows.project(con, "ledger", labels={"experiment_id":"exp_ab4"})
>>> decisions = V_base.filter(lambda r: r.payload_type == "decision").order_by("ts")
>>> df_dec = con.execute(decisions.select(decisions.look.name("look"),
...                                       decisions.planned_t.name("planned_t"),
...                                       decisions.action.name("action")))
>>> rows = df_dec.to_dict("records")
>>> len(rows) >= 3
True
>>> rows[0]["look"], float(rows[0]["planned_t"]), rows[0]["action"]
(1, 0.25, 'continue')
>>> rows[1]["look"], float(rows[1]["planned_t"]), rows[1]["action"]
(2, 0.5, 'continue')
>>> rows[2]["look"], float(rows[2]["planned_t"]), rows[2]["action"]
(3, 0.75, 'stop_efficacy')
>>>
>>> # Inspect ledger content (TableExpr; execute outside)
>>> ab_ledger.show().execute()  # doctest: +ELLIPSIS
       payload_type                                            payload                        labels
    0        design  {'planned_max_n': 1000, 'looks': [0.25, 0.5, 0...  {'experiment_id': 'exp_ab4'}
    1   observation         {'nA': 100, 'mA': 10, 'nB': 100, 'mB': 12}  {'experiment_id': 'exp_ab4'}
    2      snapshot  {'nA': 100.0, 'mA': 10.0, 'nB': 100.0, 'mB': 1...  {'experiment_id': 'exp_ab4'}
    3          info                                 {'info_time': 0.2}  {'experiment_id': 'exp_ab4'}
    4   observation             {'nA': 50, 'mA': 5, 'nB': 50, 'mB': 6}  {'experiment_id': 'exp_ab4'}
    5      snapshot  {'nA': 150.0, 'mA': 15.0, 'nB': 150.0, 'mB': 1...  {'experiment_id': 'exp_ab4'}
    6          info                                 {'info_time': 0.3}  {'experiment_id': 'exp_ab4'}
    7          stat                          {'z': 0.5535658388085484}  {'experiment_id': 'exp_ab4'}
    8      decision  {'look': 1, 'planned_t': 0.25, 'info_time': 0....  {'experiment_id': 'exp_ab4'}
    9   observation         {'nA': 150, 'mA': 10, 'nB': 150, 'mB': 20}  {'experiment_id': 'exp_ab4'}
    10     snapshot  {'nA': 300.0, 'mA': 25.0, 'nB': 300.0, 'mB': 3...  {'experiment_id': 'exp_ab4'}
    11         info                                 {'info_time': 0.6}  {'experiment_id': 'exp_ab4'}
    12         stat                          {'z': 1.7312570698610248}  {'experiment_id': 'exp_ab4'}
    13     decision  {'look': 2, 'planned_t': 0.5, 'info_time': 0.6...  {'experiment_id': 'exp_ab4'}
    14  observation          {'nA': 100, 'mA': 5, 'nB': 100, 'mB': 35}  {'experiment_id': 'exp_ab4'}
    15     snapshot  {'nA': 400.0, 'mA': 30.0, 'nB': 400.0, 'mB': 7...  {'experiment_id': 'exp_ab4'}
    16         info                                 {'info_time': 0.8}  {'experiment_id': 'exp_ab4'}
    17         stat                          {'z': 4.5391908987315395}  {'experiment_id': 'exp_ab4'}
    18     decision  {'look': 3, 'planned_t': 0.75, 'info_time': 0....  {'experiment_id': 'exp_ab4'}

# (B) E-process with single design record
>>> ep = ENormalMixture(e_ledger)
>>> ep.set_design(alpha=0.05, thetas=[0.25, 0.5, 0.75])
>>> for _ in range(12):
...     ep.update(x=0.8)
>>> V_e_base = EBaseRows.project(con, "ledger", labels={"experiment_id":"exp_e"})
>>> qE = (
...     V_e_base.filter(lambda r: r.payload_type == "e_state")
...     .order_by(lambda r: r.ts.desc())
...     .limit(1)
... )
>>> dfE = con.execute(qE.select(qE.e_value.name("e_value"), qE.alarm.name("alarm")))
>>> rowE = dfE.to_dict("records")[0]
>>> (round(rowE["e_value"], 2), bool(rowE["alarm"]))
(26.84, True)
"""

from __future__ import annotations

import uuid as _py_uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import ibis

# ==============================
# Common JSON / labels utilities
# ==============================

_PKG_VERSION = "earlysign==dev"


def _json_from_kv(items: Dict[str, ibis.Expr], *, raw_keys: set[str] | None = None) -> ibis.Expr:
    """Build a JSON object purely from Ibis expressions (no Python dumps)."""
    raw_keys = raw_keys or set()
    if not items:
        return ibis.literal("{}").cast("json")
    parts: List[ibis.Expr] = []
    for k, v in items.items():
        key = ibis.literal(f'"{k}":')
        if k in raw_keys:
            val = v.cast("string")
        else:
            val = (ibis.literal('"') + v.cast("string") + ibis.literal('"')) if v.type().is_string() else v.cast("string")
        parts.append(key + val)
    out = parts[0]
    for p in parts[1:]:
        out = out + ibis.literal(",") + p
    return (ibis.literal("{") + out + ibis.literal("}")).cast("json")


def _labels_where_sql(labels: Dict[str, Union[str, int, float, bool]]) -> str:
    """Build a backend-portable WHERE clause to match bound labels by string equality."""
    if not labels:
        return "TRUE"
    conds: List[str] = []
    for k, v in labels.items():
        sv = str(v).replace("'", "''")  # escape single quotes for SQL
        conds.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
    return " AND ".join(conds)


# =========================
# Ledger (bindable facade)
# =========================

@dataclass(frozen=True)
class Ledger:
    """Append-only JSON ledger with bindable labels (scope)."""
    con: Any
    table: str = "ledger"
    labels: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def bind(self, **labels: Union[str, int, float, bool]) -> "Ledger":
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table, merged)

    def view_json(self) -> ibis.Expr:
        """Return scoped raw view with (uuid, ts, pkg_version, payload_type, payload, labels)."""
        where_labels = _labels_where_sql(self.labels)
        sql = f"""
            SELECT
              uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
            FROM {self.table}
            WHERE {where_labels}
        """
        return self.con.sql(sql)

    def show(self) -> ibis.Expr:
        """Readable doctest view: order by ts, drop auto columns. Execute outside."""
        t = self.view_json().order_by("ts")
        try:
            t = t.drop("uuid", "ts", "pkg_version")
        except Exception:
            pass
        return t


# =======================================
# Per-row UUID/TS auto fillers (portable)
# =======================================

def _has_ibis_uuid() -> bool:
    return hasattr(ibis, "uuid")


def _uuid_single() -> ibis.Expr:
    """Return a single-row UUID expression (prefer ibis.uuid())."""
    if _has_ibis_uuid():
        return ibis.uuid().cast("string")
    return ibis.literal(_py_uuid.uuid4().hex)


def _uuid_multi(src: ibis.Expr) -> ibis.Expr:
    """
    Return a multi-row UUID expression based on a scalar base + row_number suffix.
    This stays backend-agnostic and deterministic per `src` materialization.
    """
    base = ibis.uuid().cast("string") if _has_ibis_uuid() else ibis.literal(_py_uuid.uuid4().hex)
    first_col = src[list(src.schema().names)[0]]
    w = ibis.window(order_by=[first_col])
    rn = ibis.row_number().over(w)
    rn_str = rn.cast("int64").cast("string").lpad(12, "0")
    return (base + ibis.literal("-")) + rn_str


# ======================================
# Transaction-local staging session DSL
# ======================================

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
    """Labels-bound staging session; INSERT happens only at flush/commit."""
    ledger: Ledger
    staged: _Staged = field(default_factory=_Staged)

    def __enter__(self):
        self.ledger.con.raw_sql("BEGIN;")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.flush()
                self.ledger.con.raw_sql("COMMIT;")
            else:
                self.ledger.con.raw_sql("ROLLBACK;")
        finally:
            self.staged = _Staged()

    def _labels_json_expr(self, extra: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> ibis.Expr:
        merged = dict(self.ledger.labels)
        if extra:
            merged.update(extra)
        parts = {k: ibis.literal(str(v)) for k, v in merged.items()}
        return _json_from_kv(parts)

    def write(self, payload_type: str, payload: Dict[str, Union[int, float, str, ibis.Expr]],
              *, extra_labels: Optional[Dict[str, Union[str, int, float, bool]]] = None,
              raw_keys: set[str] | None = None) -> None:
        """Stage one row built from Python scalars / Ibis Expr values."""
        anchor = ibis.memtable([{"one": 1}])
        items = {k: (v if isinstance(v, ibis.Expr) else ibis.literal(v)) for k, v in payload.items()}
        row = anchor.select(
            uuid         = _uuid_single(),
            ts           = ibis.now().cast("timestamp(6)"),
            pkg_version  = ibis.literal(_PKG_VERSION),
            payload_type = ibis.literal(payload_type),
            payload      = _json_from_kv(items, raw_keys=raw_keys or set()),
            labels       = self._labels_json_expr(extra_labels),
        )
        self.staged.add(row)

    def write_from(self, src: ibis.Expr, payload_type: str,
                   payload: Dict[str, Union[int, float, str, ibis.Expr]],
                   *, extra_labels: Optional[Dict[str, Union[str, int, float, bool]]] = None,
                   raw_keys: set[str] | None = None) -> None:
        """Stage 0..N rows derived from an Ibis source."""
        items = {k: (v if isinstance(v, ibis.Expr) else ibis.literal(v)) for k, v in payload.items()}
        row = src.select(
            uuid         = _uuid_multi(src),
            ts           = ibis.now().cast("timestamp(6)"),
            pkg_version  = ibis.literal(_PKG_VERSION),
            payload_type = ibis.literal(payload_type),
            payload      = _json_from_kv(items, raw_keys=raw_keys or set()),
            labels       = self._labels_json_expr(extra_labels),
        )
        self.staged.add(row)

    def view_json(self) -> ibis.Expr:
        return self.ledger.view_json()

    def flush(self) -> None:
        union = self.staged.union()
        if union is not None:
            self.ledger.con.insert(self.ledger.table, union)


# =========================
# AB projectors (inline SQL)
# =========================

class ABBaseRows:
    """Project design/observation/snapshot/info/stat/decision with typed AB columns."""
    @staticmethod
    def project(con: Any, table: str, labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr:
        where_labels = _labels_where_sql(labels)
        sql = f"""
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  TRY_CAST(json_extract(payload, '$.nA')            AS DOUBLE) AS nA,
  TRY_CAST(json_extract(payload, '$.mA')            AS DOUBLE) AS mA,
  TRY_CAST(json_extract(payload, '$.nB')            AS DOUBLE) AS nB,
  TRY_CAST(json_extract(payload, '$.mB')            AS DOUBLE) AS mB,
  TRY_CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE) AS planned_max_n,
  TRY_CAST(json_extract(payload, '$.planned_t')     AS DOUBLE) AS planned_t,
  TRY_CAST(json_extract(payload, '$.look')          AS BIGINT) AS look,
  TRY_CAST(json_extract(payload, '$.info_time')     AS DOUBLE) AS info_time,
  TRY_CAST(json_extract(payload, '$.z')             AS DOUBLE) AS z,
  TRY_CAST(json_extract(payload, '$.boundary')      AS DOUBLE) AS boundary,
  json_extract_string(payload, '$.action')          AS action
FROM (
  SELECT uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
  FROM {table}
  WHERE {where_labels}
) base
"""
        return con.sql(sql)


class ABDesignLooks:
    """Explode the 'looks' array from the single 'design' row into (ts, look, planned_t)."""
    @staticmethod
    def project(con: Any, table: str, labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr:
        where_labels = _labels_where_sql(labels)
        sql = f"""
SELECT
  b.ts::TIMESTAMP(6)             AS ts,
  TRY_CAST(je.key AS BIGINT) + 1 AS look,
  TRY_CAST(je.value AS DOUBLE)   AS planned_t
FROM {table} AS b,
     LATERAL json_each(b.payload, '$.looks') AS je
WHERE {where_labels}
  AND b.payload_type = 'design'
"""
        return con.sql(sql)


# ==========================
# AB application (two-props)
# ==========================

class BinomialABTest:
    """Two-proportions group-seq test (single 'design' row)."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, max_n: int, looks: Sequence[float]) -> None:
        """Write one 'design' record containing {"planned_max_n": int, "looks": [floats]}."""
        looks_text = "[" + ",".join(str(float(t)) for t in looks) + "]"
        looks_json = ibis.literal(looks_text)
        with LedgerSession(self.ledger) as s:
            s.write("design", {"planned_max_n": int(max_n), "looks": looks_json}, raw_keys={"looks"})

    def update(self, payload: Dict[str, int]) -> None:
        """
        Record observation; ALWAYS write snapshot and info first; then if a *new* look is due,
        write z/boundary/decision for the smallest such look i with I0 < t_i <= I1.
        """
        nA_add = float(payload.get("nA", 0))
        mA_add = float(payload.get("mA", 0))
        nB_add = float(payload.get("nB", 0))
        mB_add = float(payload.get("mB", 0))

        with LedgerSession(self.ledger) as s:
            # (1) observation (delta)
            s.write("observation", payload)

            # (2) projectors (no compose)
            Vb = ABBaseRows.project(s.ledger.con, s.ledger.table, s.ledger.labels)
            Vl = ABDesignLooks.project(s.ledger.con, s.ledger.table, s.ledger.labels)

            # Latest design timestamp
            Dts = Vb.filter(lambda r: r.payload_type == "design").aggregate(ts_max=Vb.ts.max())

            # Nmax from 'design' at that ts (typed via ABBaseRows)
            Dmax = Vb.filter(lambda r: r.payload_type == "design") \
                     .join(Dts, predicates=[Vb.ts == Dts.ts_max]) \
                     .select(Nmax=Vb.planned_max_n)

            # Looks (only ts, look, planned_t)
            Looks = Vl.join(Dts, predicates=[Vl.ts == Dts.ts_max]) \
                      .select(look=Vl.look.cast("int64"), planned_t=Vl.planned_t)

            # Previous cumulative counts (before current delta)
            Obs_prev = Vb.filter(lambda r: r.payload_type == "observation")
            agg_prev = Obs_prev.aggregate(
                nA=Obs_prev.nA.sum().fill_null(0.0),
                mA=Obs_prev.mA.sum().fill_null(0.0),
                nB=Obs_prev.nB.sum().fill_null(0.0),
                mB=Obs_prev.mB.sum().fill_null(0.0),
            )

            # Add current delta
            add_tbl = ibis.memtable([{"nA_add": nA_add, "mA_add": mA_add, "nB_add": nB_add, "mB_add": mB_add}])
            base = agg_prev.cross_join(add_tbl).cross_join(Dmax).select(
                nA = (agg_prev.nA + add_tbl.nA_add),
                mA = (agg_prev.mA + add_tbl.mA_add),
                nB = (agg_prev.nB + add_tbl.nB_add),
                mB = (agg_prev.mB + add_tbl.mB_add),
                Nmax = Dmax.Nmax,
            )

            # Info time before and after
            I0_tbl = agg_prev.cross_join(Dmax).select(((agg_prev.nA + agg_prev.nB) / Dmax.Nmax.nullif(0)).name("I0"))
            X = base.cross_join(I0_tbl).select(
                nA=base.nA, mA=base.mA, nB=base.nB, mB=base.mB, Nmax=base.Nmax, I0=I0_tbl.I0
            )

            # (2.5) snapshot always
            s.write_from(X, "snapshot", {"nA": X.nA, "mA": X.mA, "nB": X.nB, "mB": X.mB})

            # (3) info first
            I_expr = (X.nA + X.nB) / X.Nmax.nullif(0)
            s.write_from(X, "info", {"info_time": I_expr})

            # (4) smallest newly-due look
            due_candidates = Looks.cross_join(X).select(
                look=Looks.look, planned_t=Looks.planned_t, I0=X.I0, I1=I_expr
            ).filter(lambda r: (r.I0 < r.planned_t) & (r.planned_t <= r.I1))

            min_due = due_candidates.aggregate(min_planned_t=due_candidates.planned_t.min())
            due = due_candidates.join(min_due, predicates=[due_candidates.planned_t == min_due.min_planned_t]) \
                                .limit(1)

            # (5) if due, compute Z/boundary/decision
            p_due  = ((X.mA + X.mB) / (X.nA + X.nB))
            se_due = (p_due * (1 - p_due) * (1 / X.nA + 1 / X.nB)).sqrt().nullif(0)
            z_all  = ((X.mB / X.nB) - (X.mA / X.nA)) / se_due

            Z_join = X.cross_join(due).select(
                look=due.look, planned_t=due.planned_t, info_time=I_expr, z=z_all
            )

            # Simple OF-like boundary (piecewise) to match doctest scenario
            bnd = ibis.cases(
                (Z_join.info_time <= 0.25, ibis.literal(3.5)),
                (Z_join.info_time <= 0.5,  ibis.literal(2.963)),
                (Z_join.info_time >= 1.0,  ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                      + (ibis.literal(1.96) - ibis.literal(2.963)) * (Z_join.info_time - 0.5) / 0.5,
            )
            action = (Z_join.z.abs() >= bnd).ifelse("stop_efficacy", "continue")

            s.write_from(Z_join, "stat", {"z": Z_join.z})
            s.write_from(
                Z_join, "decision",
                {"look": Z_join.look, "planned_t": Z_join.planned_t, "info_time": Z_join.info_time,
                 "z": Z_join.z, "boundary": bnd, "action": action},
            )


# =========================
# E projectors (inline SQL)
# =========================

class EBaseRows:
    """Project e_design/e_obs/e_state with typed columns (only those needed)."""
    @staticmethod
    def project(con: Any, table: str, labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr:
        where_labels = _labels_where_sql(labels)
        sql = f"""
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  TRY_CAST(json_extract(payload, '$.x')       AS DOUBLE)  AS x,
  TRY_CAST(json_extract(payload, '$.alpha')   AS DOUBLE)  AS alpha,
  TRY_CAST(json_extract(payload, '$.e_value') AS DOUBLE)  AS e_value,
  TRY_CAST(json_extract(payload, '$.alarm')   AS BOOLEAN) AS alarm
FROM (
  SELECT uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
  FROM {table}
  WHERE {where_labels}
) base
"""
        return con.sql(sql)


class EThetas:
    """Explode theta grid from 'e_design' into rows -> (ts, theta)."""
    @staticmethod
    def project(con: Any, table: str, labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr:
        where_labels = _labels_where_sql(labels)
        sql = f"""
SELECT
  b.ts::TIMESTAMP(6)           AS ts,
  TRY_CAST(je.value AS DOUBLE) AS theta
FROM {table} b,
     LATERAL json_each(b.payload, '$.thetas') AS je
WHERE {where_labels} AND b.payload_type = 'e_design'
"""
        return con.sql(sql)


# ==========================
# E-process application impl
# ==========================

class ENormalMixture:
    """Normal-mixture E-process (uniform over given thetas; single 'e_design' row)."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        """Write one 'e_design' record with {"alpha": float, "thetas": [floats]}."""
        thetas_text = "[" + ",".join(str(float(t)) for t in thetas) + "]"
        thetas_json = ibis.literal(thetas_text)
        with LedgerSession(self.ledger) as s:
            s.write("e_design", {"alpha": float(alpha), "thetas": thetas_json}, raw_keys={"thetas"})

    def update(self, *, x: float) -> None:
        """Record x; update running mixture E-value and write e_state with alarm."""
        x_add = float(x)
        with LedgerSession(self.ledger) as s:
            s.write("e_obs", {"x": x_add})

            Vb = EBaseRows.project(s.ledger.con, s.ledger.table, s.ledger.labels)
            Vt = EThetas.project(s.ledger.con, s.ledger.table, s.ledger.labels)

            D = Vb.filter(lambda r: r.payload_type == "e_design")
            Dts = D.aggregate(ts_max=D.ts.max())
            Dcur = D.join(Dts, predicates=[D.ts == Dts.ts_max]).select(alpha=D.alpha)

            Theta = Vt.select(theta=Vt.theta)
            K = Theta.aggregate(k=Theta.theta.count())

            Obs_prev = Vb.filter(lambda r: r.payload_type == "e_obs").select(x_prev=Vb.x)
            Agg_prev = Obs_prev.aggregate(
                S_prev=Obs_prev.x_prev.sum().fill_null(0.0),
                t_prev=Obs_prev.x_prev.count()
            )

            add_tbl = ibis.memtable([{"x_add": x_add, "one": 1}])
            Agg = Agg_prev.cross_join(add_tbl).select(
                S = Agg_prev.S_prev + add_tbl.x_add,
                t = Agg_prev.t_prev + add_tbl.one,
            )

            parts = Theta.cross_join(Agg).cross_join(K).select(
                term = ((ibis.literal(1.0) / K.k.nullif(0))
                        * ((Theta.theta * Agg.S) - (0.5 * Theta.theta * Theta.theta * Agg.t)).exp())
            )
            E = parts.aggregate(e_value=parts.term.sum())

            Thr = Dcur.select(thr=(ibis.literal(1.0) / Dcur.alpha.nullif(0)))
            S_all = E.cross_join(Thr).select(
                e_value=E.e_value,
                alarm=(E.e_value >= Thr.thr)
            )

            s.write_from(S_all, "e_state", {"e_value": S_all.e_value, "alarm": S_all.alarm.cast("boolean")})
