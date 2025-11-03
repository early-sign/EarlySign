"""
Framework + Apps: Labels-bound Ledger DSL with Pluggable Flat Projectors
========================================================================

This single file demonstrates a clean separation between a **framework layer**
and **applications** that implement their own JSON→typed projections.

Framework (backend-agnostic)
----------------------------
- Table contract (must already exist):
    uuid TEXT, ts TIMESTAMP, pkg_version TEXT, payload_type TEXT, payload JSON, labels JSON
- Ledger: bindable scope over (table, labels), with convenient show() returning a TableExpr
- LedgerSession: transaction-local staging with auto uuid/ts/pkg_version
- FlatProjector (abstract): (con, table, labels) -> typed ibis table

Applications
------------
- BinomialABTest: two-proportions group-sequential (design **single record**)
  - Supplies ABFlatProjector (AB fields only)
- ENormalMixture: normal-mixture E-process (design **single record**)
  - Supplies EFlatProjector (E fields only)

Design single-record contract
-----------------------------
- AB:    payload_type="design",    payload={"planned_max_n": <int>, "looks": [t1, t2, ...]}
- Eproc: payload_type="e_design",  payload={"alpha": <float>, "thetas": [θ1, θ2, ...]}

Projectors synthesize helper rows
---------------------------------
- ABFlatProjector  : base rows UNION ALL
    * synthetic 'design_max_n' row (extract planned_max_n from 'design')
    * synthetic 'design_look'   rows (json_each looks array)
- EFlatProjector   : base rows UNION ALL
    * synthetic 'e_theta' rows (json_each thetas array)

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
>>> # Prepare a base ledger and bind separate label scopes for AB and E
>>> base_ledger = Ledger(con, table="ledger")
>>> ab_ledger = base_ledger.bind(experiment_id="exp_ab4")
>>> e_ledger  = base_ledger.bind(experiment_id="exp_e")

# (A) AB test with single design record (4 looks).
#     We craft updates so that: look=1 (t=0.25) -> continue,
#                               look=2 (t=0.5)  -> continue,
#                               look=3 (t=0.75) -> STOP (first stop).
>>> ab = BinomialABTest(ab_ledger)
>>> ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0])
>>> # Update-1: I=0.20 (< 0.25), no look
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> # Update-2: I=0.30, triggers look=1 (t=0.25) -> continue
>>> ab.update({"nA": 50, "mA": 5, "nB": 50, "mB": 6})
>>> # Update-3: I=0.60, triggers look=2 (t=0.5) -> continue
>>> ab.update({"nA": 150, "mA": 10, "nB": 150, "mB": 20})
>>> # Update-4: I=0.80, triggers look=3 (t=0.75) -> STOP
>>> ab.update({"nA": 100, "mA": 5, "nB": 100, "mB": 35})
>>>
>>> V4 = ABFlatProjector.project(con, "ledger", labels={"experiment_id":"exp_ab4"})
>>> decisions = V4.filter(lambda r: r.payload_type == "decision").order_by("ts")
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
>>> # Inspect ledger content
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
>>> ep.set_design(alpha=0.05, thetas=[0.25, 0.5, 0.75])  # uniform weights 1/K
>>> for _ in range(12):
...     ep.update(x=0.8)
>>> qE = (
...   EFlatProjector.project(con, "ledger", labels={"experiment_id":"exp_e"})
...     .filter(lambda r: r.payload_type == "e_state")
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
from typing import Dict, List, Optional, Protocol, Sequence, Union

import ibis


# ===================================
# Framework: common JSON/labels utils
# ===================================

_PKG_VERSION = "earlysign==dev"


def _json_from_kv(items: Dict[str, ibis.Expr], *, raw_keys: set[str] | None = None) -> ibis.Expr:
    """
    Build a JSON object from Ibis expressions only (no Python json.dumps).

    - If key is in `raw_keys`, the value is injected as-is (no extra quotes).
      This is necessary to embed JSON arrays/objects as values.
    - Otherwise:
        - if value is string-typed -> quoted
        - else -> cast to string and injected (no extra quotes)
    """
    raw_keys = raw_keys or set()
    if not items:
        return ibis.literal("{}").cast("json")
    parts: List[ibis.Expr] = []
    for k, v in items.items():
        key = ibis.literal(f'"{k}":')
        if k in raw_keys:
            val = v.cast("string")  # raw JSON text (e.g., "[0.5,1.0]") injected
        else:
            val = (
                ibis.literal('"') + v.cast("string") + ibis.literal('"')
                if v.type().is_string() else v.cast("string")
            )
        parts.append(key + val)
    out = parts[0]
    for p in parts[1:]:
        out = out + ibis.literal(",") + p
    return (ibis.literal("{") + out + ibis.literal("}")).cast("json")


def _labels_where_sql(labels: Dict[str, Union[str, int, float, bool]]) -> str:
    """Return a DuckDB SQL predicate that matches exact stringified labels."""
    if not labels:
        return "TRUE"
    conds = []
    for k, v in labels.items():
        sv = str(v).replace("'", "''")
        conds.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
    return " AND ".join(conds)


# =========================
# Framework: Ledger & bind
# =========================

@dataclass(frozen=True)
class Ledger:
    """A bindable, append-only JSON ledger facade over an Ibis table."""
    con: ibis.Client
    table: str = "ledger"
    labels: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def bind(self, **labels: Union[str, int, float, bool]) -> "Ledger":
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table, merged)

    def view_json(self) -> ibis.Expr:
        """(uuid, ts, pkg_version, payload_type, payload, labels) scoped by bound labels."""
        where_labels = _labels_where_sql(self.labels)
        sql = f"""
            SELECT
              uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
            FROM {self.table}
            WHERE {where_labels}
        """
        return self.con.sql(sql)

    def show(self) -> ibis.Expr:
        """Return a TableExpr (not executed) ordered by ts; auto cols dropped for readability."""
        t = self.view_json().order_by("ts")
        try:
            t = t.drop("uuid", "ts", "pkg_version")
        except Exception:
            pass
        return t


# =======================================
# Framework: per-row uuid/ts auto-fillers
# =======================================

def _has_ibis_uuid() -> bool:
    return hasattr(ibis, "uuid")


def _uuid_single() -> ibis.Expr:
    """Single-row UUID expr: prefer ibis.uuid(), fallback to Python uuid4()."""
    if _has_ibis_uuid():
        return ibis.uuid().cast("string")
    return ibis.literal(_py_uuid.uuid4().hex)


def _uuid_multi(src: ibis.Expr) -> ibis.Expr:
    """
    Multi-row UUID expr:
      - base = ibis.uuid() (or Python uuid) as scalar
      - suffix = row_number() anchored to `src`
    """
    base = ibis.uuid().cast("string") if _has_ibis_uuid() else ibis.literal(_py_uuid.uuid4().hex)
    first_col = src[list(src.schema().names)[0]]
    w = ibis.window(order_by=[first_col])
    rn = ibis.row_number().over(w)
    rn_str = rn.cast("int64").cast("string").lpad(12, "0")
    return (base + ibis.literal("-")) + rn_str


# ========================================
# Framework: Transaction-local staging DSL
# ========================================

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
    """Labels-bound staging session (append-only)."""
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
        """Stage a single JSON row (uuid/ts/pkg_version auto)."""
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
        """Stage rows produced from an Ibis source (0..N rows)."""
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


# ======================================
# Framework: FlatProjector (application)
# ======================================

class FlatProjector(Protocol):
    """Application implements JSON->typed projection for its own payloads."""
    @staticmethod
    def project(con: ibis.Client, table: str,
                labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr: ...


# ==========================================
# App: AB test (its own flat view projector)
# ==========================================

class ABFlatProjector:
    """Typed projection for AB fields.
       Synthesizes `design_max_n` + `design_look` rows from single `design` record."""

    @staticmethod
    def project(con, table: str, labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr:
        where_labels = _labels_where_sql(labels)
        sql = f"""
WITH base AS (
  SELECT
    uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
  FROM {table}
  WHERE {where_labels}
),
dmax AS (
  SELECT
    uuid,
    ts::TIMESTAMP(6) AS ts,
    pkg_version,
    'design_max_n'   AS payload_type,
    payload,
    labels,
    TRY_CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE) AS planned_max_n
  FROM base
  WHERE payload_type = 'design'
),
dlooks AS (
  SELECT
    b.uuid,
    b.ts::TIMESTAMP(6) AS ts,
    b.pkg_version,
    'design_look'      AS payload_type,
    b.payload,
    b.labels,
    TRY_CAST(je.key AS BIGINT) + 1              AS look,
    TRY_CAST(je.value AS DOUBLE)                AS planned_t
  FROM base b,
       LATERAL json_each(b.payload, '$.looks') AS je
  WHERE b.payload_type = 'design'
)
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  -- AB fields (base rows, including decision fields)
  TRY_CAST(json_extract(payload, '$.nA') AS DOUBLE) AS nA,
  TRY_CAST(json_extract(payload, '$.mA') AS DOUBLE) AS mA,
  TRY_CAST(json_extract(payload, '$.nB') AS DOUBLE) AS nB,
  TRY_CAST(json_extract(payload, '$.mB') AS DOUBLE) AS mB,
  TRY_CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE) AS planned_max_n,
  TRY_CAST(json_extract(payload, '$.planned_t')     AS DOUBLE) AS planned_t,
  TRY_CAST(json_extract(payload, '$.look')          AS BIGINT) AS look,
  TRY_CAST(json_extract(payload, '$.info_time')     AS DOUBLE) AS info_time,
  TRY_CAST(json_extract(payload, '$.z')             AS DOUBLE) AS z,
  TRY_CAST(json_extract(payload, '$.boundary')      AS DOUBLE) AS boundary,
  json_extract_string(payload, '$.action')                  AS action
FROM base
UNION ALL
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  NULL, NULL, NULL, NULL,
  planned_max_n, NULL, NULL, NULL, NULL, NULL, NULL
FROM dmax
UNION ALL
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  NULL, NULL, NULL, NULL,
  NULL, planned_t, look, NULL, NULL, NULL, NULL
FROM dlooks
"""
        return con.sql(sql)


class BinomialABTest:
    """Two-proportions group-sequential test using ABFlatProjector (single design row)."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, max_n: int, looks: Sequence[float]) -> None:
        """Write a single design record: {"planned_max_n": int, "looks": [floats]}."""
        looks_text = "[" + ",".join(str(float(t)) for t in looks) + "]"
        looks_json = ibis.literal(looks_text)
        with LedgerSession(self.ledger) as s:
            s.write(
                "design",
                {"planned_max_n": int(max_n), "looks": looks_json},
                raw_keys={"looks"},
            )

    def update(self, payload: Dict[str, int]) -> None:
        """Record observation; always write snapshot and info; Z/decision only when a look becomes newly due."""
        nA_add = float(payload.get("nA", 0))
        mA_add = float(payload.get("mA", 0))
        nB_add = float(payload.get("nB", 0))
        mB_add = float(payload.get("mB", 0))

        with LedgerSession(self.ledger) as s:
            # (1) record observation (delta)
            s.write("observation", payload)

            # (2) AB-specific typed view (includes synthetic design_* rows)
            V = ABFlatProjector.project(s.ledger.con, s.ledger.table, s.ledger.labels)

            # Latest design timestamp
            design_ts = V.filter(lambda r: r.payload_type.isin(("design_max_n", "design_look"))) \
                         .aggregate(ts_max=V.ts.max())

            Dmax = V.filter(lambda r: r.payload_type == "design_max_n") \
                    .join(design_ts, predicates=[V.ts == design_ts.ts_max]) \
                    .select(Nmax=V.planned_max_n)

            Looks = V.filter(lambda r: r.payload_type == "design_look") \
                     .join(design_ts, predicates=[V.ts == design_ts.ts_max]) \
                     .select(look=V.look.cast("int64"), planned_t=V.planned_t)

            # Previous cumulative counts (before this delta)
            Obs_prev = V.filter(lambda r: r.payload_type == "observation")
            agg_prev = Obs_prev.aggregate(
                nA=Obs_prev.nA.sum().fill_null(0.0),
                mA=Obs_prev.mA.sum().fill_null(0.0),
                nB=Obs_prev.nB.sum().fill_null(0.0),
                mB=Obs_prev.mB.sum().fill_null(0.0),
            )

            # Add this delta
            add_tbl = ibis.memtable([{"nA_add": nA_add, "mA_add": mA_add, "nB_add": nB_add, "mB_add": mB_add}])
            base = agg_prev.cross_join(add_tbl).cross_join(Dmax).select(
                nA = (agg_prev.nA + add_tbl.nA_add),
                mA = (agg_prev.mA + add_tbl.mA_add),
                nB = (agg_prev.nB + add_tbl.nB_add),
                mB = (agg_prev.mB + add_tbl.mB_add),
                Nmax = Dmax.Nmax,
            )

            # Info time BEFORE and AFTER
            I0_tbl = agg_prev.cross_join(Dmax).select(
                ((agg_prev.nA + agg_prev.nB) / Dmax.Nmax.nullif(0)).name("I0")
            )
            X = base.cross_join(I0_tbl).select(
                nA=base.nA, mA=base.mA, nB=base.nB, mB=base.mB,
                Nmax=base.Nmax, I0=I0_tbl.I0
            )

            # (2.5) ALWAYS write snapshot (cumulative counts)
            s.write_from(X, "snapshot", {"nA": X.nA, "mA": X.mA, "nB": X.nB, "mB": X.mB})

            # (3) ALWAYS write info first
            I_expr = (X.nA + X.nB) / X.Nmax.nullif(0)
            s.write_from(X, "info", {"info_time": I_expr})

            # (4) Decide if ANY planned look became newly due: pick the smallest such look
            due_candidates = Looks.cross_join(X).select(
                look=Looks.look,
                planned_t=Looks.planned_t,
                I0=X.I0,
                I1=I_expr
            ).filter(lambda r: (r.I0 < r.planned_t) & (r.planned_t <= r.I1))

            # Choose the smallest planned_t if exists
            min_due = due_candidates.aggregate(min_planned_t=due_candidates.planned_t.min())
            due = due_candidates.join(min_due, predicates=[due_candidates.planned_t == min_due.min_planned_t]) \
                                .limit(1)

            # (5) If due exists, compute Z / boundary / decision, and write them
            p_due  = ( (X.mA + X.mB) / (X.nA + X.nB) )
            se_due = ( p_due * (1 - p_due) * (1 / X.nA + 1 / X.nB) ).sqrt().nullif(0)
            z_all  = ((X.mB / X.nB) - (X.mA / X.nA)) / se_due

            Z_join = X.cross_join(due).select(
                look=due.look,
                planned_t=due.planned_t,
                info_time=I_expr,
                z=z_all
            )

            # OF-like boundary (slightly higher at <=0.25 to be conservative early)
            bnd = ibis.cases(
                (Z_join.info_time <= 0.25, ibis.literal(3.5)),
                (Z_join.info_time <= 0.5,  ibis.literal(2.963)),
                (Z_join.info_time >= 1.0,  ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                      + (ibis.literal(1.96) - ibis.literal(2.963)) * (Z_join.info_time - 0.5) / 0.5,
            )
            action = (Z_join.z.abs() >= bnd).ifelse("stop_efficacy", "continue")

            # Guarded: 0 rows if no due (so no write)
            s.write_from(Z_join, "stat", {"z": Z_join.z})
            s.write_from(
                Z_join, "decision",
                {"look": Z_join.look, "planned_t": Z_join.planned_t, "info_time": Z_join.info_time,
                 "z": Z_join.z, "boundary": bnd, "action": action},
            )


# ===============================================
# App: E-process (its own flat view projector)
# ===============================================

class EFlatProjector:
    """Typed projection for E fields.
       Synthesizes `e_theta` rows from single `e_design` record."""

    @staticmethod
    def project(con, table: str, labels: Dict[str, Union[str, int, float, bool]]) -> ibis.Expr:
        where_labels = _labels_where_sql(labels)
        sql = f"""
WITH base AS (
  SELECT
    uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
  FROM {table}
  WHERE {where_labels}
),
ethetas AS (
  SELECT
    b.uuid,
    b.ts::TIMESTAMP(6) AS ts,
    b.pkg_version,
    'e_theta'          AS payload_type,
    b.payload,
    b.labels,
    TRY_CAST(je.value AS DOUBLE)                AS theta
  FROM base b,
       LATERAL json_each(b.payload, '$.thetas') AS je
  WHERE b.payload_type = 'e_design'
)
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  -- E fields for base rows
  TRY_CAST(json_extract(payload, '$.x')       AS DOUBLE)   AS x,
  TRY_CAST(json_extract(payload, '$.alpha')   AS DOUBLE)   AS alpha,
  NULL::DOUBLE AS theta,
  TRY_CAST(json_extract(payload, '$.e_value') AS DOUBLE)   AS e_value,
  TRY_CAST(json_extract(payload, '$.alarm')   AS BOOLEAN)  AS alarm
FROM base
UNION ALL
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  NULL, NULL, theta, NULL, NULL
FROM ethetas
"""
        return con.sql(sql)


class ENormalMixture:
    """Normal-mixture E-process with uniform weights over provided theta grid (single design row)."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        """Write a single design record: {"alpha": float, "thetas": [floats]}."""
        thetas_text = "[" + ",".join(str(float(t)) for t in thetas) + "]"
        thetas_json = ibis.literal(thetas_text)
        with LedgerSession(self.ledger) as s:
            s.write(
                "e_design",
                {"alpha": float(alpha), "thetas": thetas_json},
                raw_keys={"thetas"},
            )

    def update(self, *, x: float) -> None:
        """Record x; compute running mixture E-value and alarm (α)."""
        x_add = float(x)
        with LedgerSession(self.ledger) as s:
            s.write("e_obs", {"x": x_add})

            V = EFlatProjector.project(s.ledger.con, s.ledger.table, s.ledger.labels)

            D = V.filter(lambda r: r.payload_type == "e_design")
            Dts = D.aggregate(ts_max=D.ts.max())
            Dcur = D.join(Dts, predicates=[D.ts == Dts.ts_max]).select(alpha=D.alpha)

            Theta = EFlatProjector.project(s.ledger.con, s.ledger.table, s.ledger.labels) \
                    .filter(lambda r: r.payload_type == "e_theta") \
                    .select(theta=V.theta)

            K = Theta.aggregate(k=Theta.theta.count())

            Obs_prev = V.filter(lambda r: r.payload_type == "e_obs").select(x_prev=V.x)
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
                term = ( (ibis.literal(1.0) / K.k.nullif(0))
                         * ( (Theta.theta * Agg.S) - (0.5 * Theta.theta * Theta.theta * Agg.t) ).exp() )
            )
            E = parts.aggregate(e_value=parts.term.sum())

            Thr = Dcur.select(thr=(ibis.literal(1.0) / Dcur.alpha.nullif(0)))
            S_all = E.cross_join(Thr).select(
                e_value=E.e_value,
                alarm=(E.e_value >= Thr.thr)
            )

            s.write_from(S_all, "e_state", {"e_value": S_all.e_value, "alarm": S_all.alarm.cast("boolean")})
