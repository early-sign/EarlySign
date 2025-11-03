"""
Framework + Apps with on-the-fly alpha-spending boundaries (Lan–DeMets)
======================================================================

- Boundaries are computed on-the-fly from the spending function at the
  *current* information time I1 (and the last executed look's info time I0).
  We do NOT pre-store per-look thresholds in the design.
- Due detection uses planned looks only to decide *whether to run a look now*:
      run if I0 < t_i <= I1 for the smallest such i
  but the boundary itself uses the *current* I1 via spending, independent of t_i.

Table (DuckDB) used in doctest
------------------------------
uuid TEXT, ts TIMESTAMP, pkg_version TEXT, payload_type TEXT, payload JSON, labels JSON

Doctest (end-to-end)
--------------------
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

# (A) A/B test with alpha-spending design (4 looks).
#     Crafted so: look=1 -> continue, look=2 -> continue, look=3 -> STOP.
>>> ab = BinomialABTest(ab_ledger)
>>> ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="obrien_fleming")
>>> # Update-1: I -> 0.20 (< 0.25), no look
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> # Update-2: I -> 0.30, triggers look=1 -> continue
>>> ab.update({"nA": 50, "mA": 5, "nB": 50, "mB": 6})
>>> # Update-3: I -> 0.60, triggers look=2 -> continue
>>> ab.update({"nA": 150, "mA": 10, "nB": 150, "mB": 20})
>>> # Update-4: I -> 0.80, triggers look=3 -> STOP
>>> ab.update({"nA": 100, "mA": 5, "nB": 100, "mB": 35})
>>>
>>> # Check decisions
>>> V_base = ABRowsBase(ab_ledger).typed_view()
>>> decisions = V_base.filter(lambda r: r.payload_type == "decision").order_by("ts")
>>> df_dec = con.execute(decisions.select(
...     decisions.look.name("look"),
...     decisions.planned_t.name("planned_t"),
...     decisions.action.name("action")
... ))
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
>>> ab_ledger.show().execute()
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

# (B) E-process (unchanged)
>>> ep = ENormalMixture(e_ledger)
>>> ep.set_design(alpha=0.05, thetas=[0.25, 0.5, 0.75])
>>> for _ in range(12):
...     ep.update(x=0.8)
>>> V_e_base = EBaseRows(e_ledger).typed_view()
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

import math
import uuid as _py_uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import ibis


# ===================================
# Common JSON / labels utilities
# ===================================

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
        where_labels = LedgerReaderBase._labels_where_sql(self.labels)
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


# =================================
# Reader base + AB/E readers
# =================================

@dataclass(frozen=True)
class LedgerReaderBase:
    """Base for typed/projected readers bound to a Ledger scope."""
    ledger: Ledger

    @staticmethod
    def _labels_where_sql(labels: Dict[str, Union[str, int, float, bool]]) -> str:
        if not labels:
            return "TRUE"
        conds: List[str] = []
        for k, v in labels.items():
            sv = str(v).replace("'", "''")  # escape for SQL
            conds.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
        return " AND ".join(conds)

    def base_view(self) -> ibis.Expr:
        """Labels-scoped raw rows (uuid, ts, pkg_version, payload_type, payload, labels)."""
        return self.ledger.view_json()

    def latest_row(self, payload_type: str) -> ibis.Expr:
        """Latest row for a given payload_type under current labels."""
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
SELECT
  b.uuid,
  b.ts::TIMESTAMP(6) AS ts,
  b.pkg_version,
  b.payload_type,
  b.payload,
  b.labels
FROM {self.ledger.table} b
WHERE {where_labels} AND b.payload_type = '{payload_type}'
ORDER BY ts DESC
LIMIT 1
"""
        return self.ledger.con.sql(sql)

    def rows_since(self, payload_type: str, ts_expr: ibis.Expr, *, strict: bool = True) -> ibis.Expr:
        """All rows of payload_type with ts > (or >=) given ts_expr (single-row table with 'ts')."""
        where_labels = self._labels_where_sql(self.ledger.labels)
        cmp = ">" if strict else ">="
        sql = f"""
WITH t0 AS ({ts_expr.compile()})
SELECT
  b.uuid,
  b.ts::TIMESTAMP(6) AS ts,
  b.pkg_version,
  b.payload_type,
  b.payload,
  b.labels
FROM {self.ledger.table} b, t0
WHERE {where_labels}
  AND b.payload_type = '{payload_type}'
  AND b.ts {cmp} t0.ts
"""
        return self.ledger.con.sql(sql)


class ABRowsBase(LedgerReaderBase):
    """Typed view for AB payloads (reused by multiple small readers)."""
    def typed_view(self) -> ibis.Expr:
        where_labels = self._labels_where_sql(self.ledger.labels)
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
  FROM {self.ledger.table}
  WHERE {where_labels}
) base
"""
        return self.ledger.con.sql(sql)


class ABDesignLatest(LedgerReaderBase):
    """Latest design (planned_max_n, alpha, spending family, looks, ts)."""
    def latest(self) -> ibis.Expr:
        base = self.latest_row("design")
        sql = f"""
SELECT
  ts,
  TRY_CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE) AS planned_max_n,
  TRY_CAST(json_extract(payload, '$.alpha')         AS DOUBLE) AS alpha,
  json_extract_string(payload, '$.spending_family') AS spending_family,
  payload                                           AS payload_json
FROM ({base.compile()})
"""
        return self.ledger.con.sql(sql)


class ABDesignLooks(LedgerReaderBase):
    """Explode 'looks' from latest design ts -> (look, planned_t)."""
    def table(self) -> ibis.Expr:
        D = ABDesignLatest(self.ledger).latest()
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
WITH latest AS (SELECT ts FROM ({D.compile()}))
SELECT
  b.ts::TIMESTAMP(6)             AS ts,
  TRY_CAST(je.key AS BIGINT) + 1 AS look,
  TRY_CAST(je.value AS DOUBLE)   AS planned_t
FROM {self.ledger.table} b,
     LATERAL json_each(b.payload, '$.looks') AS je
WHERE {where_labels} AND b.payload_type = 'design'
  AND b.ts = (SELECT ts FROM latest)
ORDER BY look
"""
        return self.ledger.con.sql(sql)


class ABSnapshotLatest(LedgerReaderBase):
    """Latest snapshot row; if none, return a single 0-initialized row with epoch ts."""
    def latest(self) -> ibis.Expr:
        base = self.latest_row("snapshot")
        sql = f"""
WITH s AS (
  SELECT
    ts,
    TRY_CAST(json_extract(payload, '$.nA') AS DOUBLE) AS nA,
    TRY_CAST(json_extract(payload, '$.mA') AS DOUBLE) AS mA,
    TRY_CAST(json_extract(payload, '$.nB') AS DOUBLE) AS nB,
    TRY_CAST(json_extract(payload, '$.mB') AS DOUBLE) AS mB
  FROM ({base.compile()})
)
SELECT * FROM s
UNION ALL
SELECT
  TIMESTAMP '1970-01-01 00:00:00' AS ts, 0.0 AS nA, 0.0 AS mA, 0.0 AS nB, 0.0 AS mB
WHERE NOT EXISTS (SELECT 1 FROM s)
"""
        return self.ledger.con.sql(sql)


class ABObsSince(LedgerReaderBase):
    """Aggregate observations strictly after given ts0."""
    def sum_after(self, ts0: ibis.Expr) -> ibis.Expr:
        obs = self.rows_since("observation", ts0, strict=True)
        sql = f"""
SELECT
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.nA') AS DOUBLE)), 0.0) AS d_nA,
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.mA') AS DOUBLE)), 0.0) AS d_mA,
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.nB') AS DOUBLE)), 0.0) AS d_nB,
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.mB') AS DOUBLE)), 0.0) AS d_mB
FROM ({obs.compile()})
"""
        return self.ledger.con.sql(sql)


# =========================================
# Alpha-spending helpers (pure Python math)
# =========================================

def _phi_inv(p: float) -> float:
    """Inverse CDF of standard normal Φ^{-1}(p), high-accuracy rational approx (Acklam)."""
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return float("-inf")
        if p == 1.0:
            return float("inf")
        raise ValueError("p must be in (0,1)")
    a = [ -3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
           1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00 ]
    b = [ -5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01 ]
    c = [ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00 ]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00 ]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if phigh < p:
        q = math.sqrt(-2*math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _cdf_normal(x: float) -> float:
    """Φ(x) via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _alpha_spent(t: float, alpha: float, family: str) -> float:
    """Cumulative α(t) for Lan–DeMets spending families (two-sided)."""
    t = min(max(t, 1e-12), 1.0)
    family = family.lower()
    if family in ["of", "obrien_fleming", "o'brien_fleming", "obrien-fleming"]:
        # Using the standard OF approximation (cumulative spending)
        # α(t) ≈ 2 - 2 Φ(z_{α/2} / sqrt(t))
        z = _phi_inv(1 - alpha/2.0)
        return 2.0 - 2.0 * _cdf_normal(z / math.sqrt(t))
    elif family in ["pocock"]:
        # α(t) = α * ln(1 + (e - 1) t)
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    else:
        raise ValueError(f"Unknown spending family: {family}")


# ==========================
# AB application (two-props)
# ==========================

class BinomialABTest:
    """Two-proportions group-seq test with on-the-fly alpha-spending boundaries."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, max_n: int, looks: Sequence[float], alpha: float = 0.05, spending: str = "obrien_fleming") -> None:
        """
        Write one 'design' record containing:
            {
              "planned_max_n": int,
              "looks": [floats],
              "alpha": float,
              "spending_family": str
            }
        Note: No per-look boundaries are stored.
        """
        looks = [float(x) for x in looks]
        looks_text = "[" + ",".join(str(u) for u in looks) + "]"
        looks_json  = ibis.literal(looks_text)
        with LedgerSession(self.ledger) as s:
            s.write(
                "design",
                {
                    "planned_max_n": int(max_n),
                    "looks": looks_json,
                    "alpha": float(alpha),
                    "spending_family": spending,
                },
                raw_keys={"looks"}
            )

    def _boundary_from_spending(self, I0: float, I1: float, *, alpha: float, family: str) -> float:
        """Two-sided single-look Z threshold from local spending Δα = α(I1) - α(I0)."""
        A1 = _alpha_spent(I1, alpha, family)
        A0 = _alpha_spent(I0, alpha, family) if I0 > 0 else 0.0
        local = max(A1 - A0, 1e-16)
        return _phi_inv(1.0 - local/2.0)

    def update(self, payload: Dict[str, int]) -> None:
        """
        Record observation delta; ALWAYS write snapshot and info first; then if a *new* look is due,
        compute boundary on-the-fly using current I1 and the last executed look's info time I0.
        """
        # Fail fast to catch schema issues early
        nA_add = float(payload["nA"])
        mA_add = float(payload["mA"])
        nB_add = float(payload["nB"])
        mB_add = float(payload["mB"])

        with LedgerSession(self.ledger) as s:
            # (0) observation (delta)
            s.write("observation", payload)

            # Readers
            D_latest = ABDesignLatest(self.ledger).latest()
            Looks    = ABDesignLooks(self.ledger).table()
            S_prev   = ABSnapshotLatest(self.ledger).latest()
            Obs_since= ABObsSince(self.ledger).sum_after(S_prev.select("ts"))

            # Nmax, alpha, family from design
            Dpars = D_latest.select(
                Nmax   = D_latest.planned_max_n,
                alpha  = D_latest.alpha,
                family = D_latest.spending_family
            )

            # "before" cumulative (previous snapshot + obs since snapshot)
            before = S_prev.cross_join(Obs_since).select(
                nA_before = S_prev.nA + Obs_since.d_nA,
                mA_before = S_prev.mA + Obs_since.d_mA,
                nB_before = S_prev.nB + Obs_since.d_nB,
                mB_before = S_prev.mB + Obs_since.d_mB,
            )

            # Add current delta to get "now"
            add_tbl = ibis.memtable([{"nA_add": nA_add, "mA_add": mA_add, "nB_add": nB_add, "mB_add": mB_add}])
            now = before.cross_join(add_tbl).cross_join(Dpars).select(
                nA = before.nA_before + add_tbl.nA_add,
                mA = before.mA_before + add_tbl.mA_add,
                nB = before.nB_before + add_tbl.nB_add,
                mB = before.mB_before + add_tbl.mB_add,
                Nmax = Dpars.Nmax,
                alpha = Dpars.alpha,
                family = Dpars.family,
                n_before = before.nA_before + before.nB_before
            )

            # (1) snapshot ALWAYS
            s.write_from(now, "snapshot", {"nA": now.nA, "mA": now.mA, "nB": now.nB, "mB": now.mB})

            # (2) info ALWAYS (I1)
            I1 = (now.nA + now.nB) / now.Nmax.nullif(0)
            s.write_from(now, "info", {"info_time": I1})

            # (3) due detection I0 < t_i ≤ I1 using planned looks
            I0_tbl = now.select(I0=(now.n_before / now.Nmax.nullif(0)))
            due_candidates = Looks.cross_join(I0_tbl).cross_join(now.select(I1=I1)).select(
                look=Looks.look, planned_t=Looks.planned_t, I0=I0_tbl.I0, I1=I1
            ).filter(lambda r: (r.I0 < r.planned_t) & (r.planned_t <= r.I1))

            min_due = due_candidates.aggregate(min_planned_t=due_candidates.planned_t.min())
            due = due_candidates.join(min_due, predicates=[due_candidates.planned_t == min_due.min_planned_t]).limit(1)

            # (4) if due, compute Z and boundary ON THE FLY from spending at current I1 vs I0
            p_all  = ((now.mA + now.mB) / (now.nA + now.nB))
            se_all = (p_all * (1 - p_all) * (1 / now.nA + 1 / now.nB)).sqrt().nullif(0)
            z_all  = ((now.mB / now.nB) - (now.mA / now.nA)) / se_all

            # Compute boundary in Python via map over a single-row DataFrame:
            # Pull I0, I1, alpha, family via execute on tiny select to keep the DSL simple.
            df_params = self.ledger.con.execute(
                now.cross_join(I0_tbl).select(I0=I0_tbl.I0, I1=I1, alpha=now.alpha, family=now.family).limit(1)
            )
            I0_val   = float(df_params.iloc[0]["I0"])
            I1_val   = float(df_params.iloc[0]["I1"])
            alpha_v  = float(df_params.iloc[0]["alpha"])
            family_v = str(df_params.iloc[0]["family"])
            bnd_val  = self._boundary_from_spending(I0_val, I1_val, alpha=alpha_v, family=family_v)

            # Use scalar boundary as a literal (valid for this single due look execution)
            Z_join = now.cross_join(due).select(
                look=due.look, planned_t=due.planned_t, info_time=I1, z=z_all,
                boundary=ibis.literal(float(bnd_val))
            )
            action = (Z_join.z.abs() >= Z_join.boundary).ifelse("stop_efficacy", "continue")

            s.write_from(Z_join, "stat", {"z": Z_join.z})
            s.write_from(
                Z_join, "decision",
                {"look": Z_join.look, "planned_t": Z_join.planned_t, "info_time": Z_join.info_time,
                 "z": Z_join.z, "boundary": Z_join.boundary, "action": action},
            )


# =========================
# E readers (inline SQL)
# =========================

class EBaseRows(LedgerReaderBase):
    """Project e_design/e_obs/e_state with only needed typed columns."""
    def typed_view(self) -> ibis.Expr:
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
SELECT
  uuid, ts, pkg_version, payload_type, payload, labels,
  TRY_CAST(json_extract(payload, '$.x')       AS DOUBLE)  AS x,
  TRY_CAST(json_extract(payload, '$.alpha')   AS DOUBLE)  AS alpha,
  TRY_CAST(json_extract(payload, '$.e_value') AS DOUBLE)  AS e_value,
  TRY_CAST(json_extract(payload, '$.alarm')   AS BOOLEAN) AS alarm
FROM (
  SELECT uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
  FROM {self.ledger.table}
  WHERE {where_labels}
) base
"""
        return self.ledger.con.sql(sql)


class EThetas(LedgerReaderBase):
    """Explode theta grid from the single 'e_design' row -> (ts, theta)."""
    def grid(self) -> ibis.Expr:
        latest_design = self.latest_row("e_design")
        sql = f"""
WITH latest AS (SELECT ts FROM ({latest_design.compile()}))
SELECT
  b.ts::TIMESTAMP(6)           AS ts,
  TRY_CAST(je.value AS DOUBLE) AS theta
FROM {self.ledger.table} b,
     LATERAL json_each(b.payload, '$.thetas') AS je
WHERE {self._labels_where_sql(self.ledger.labels)} AND b.payload_type = 'e_design'
  AND b.ts = (SELECT ts FROM latest)
"""
        return self.ledger.con.sql(sql)


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

            Vb = EBaseRows(self.ledger).typed_view()
            Theta = EThetas(self.ledger).grid()

            # K and Obs aggregates before current delta (current delta not yet committed)
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

            D_latest = LedgerReaderBase(self.ledger).latest_row("e_design")
            Thr = self.ledger.con.sql(f"""
SELECT (1.0 / TRY_CAST(json_extract(payload, '$.alpha') AS DOUBLE)) AS thr
FROM ({D_latest.compile()})
""")

            S_all = E.cross_join(Thr).select(
                e_value=E.e_value,
                alarm=(E.e_value >= Thr.thr)
            )

            s.write_from(S_all, "e_state", {"e_value": S_all.e_value, "alarm": S_all.alarm.cast("boolean")})

    def latest_row(self, payload_type: str) -> ibis.Expr:
        return LedgerReaderBase(self.ledger).latest_row(payload_type)

# ==========================
# Profiling helper (callable)
# ==========================

def run_profile_demo(con: Any) -> str:
    """
    Run a short AB scenario under profiling and return stats as a string.

    - Uses a separate label scope (experiment_id="exp_prof") so it doesn't
      interfere with other runs.
    - Filters stats to functions matching /(insert|raw_sql|flush|write_from|write|uuid|now)/.
    - Sorts by cumulative time.
    """
    import cProfile
    import pstats
    import io

    base = Ledger(con, table="ledger")
    prof_ledger = base.bind(experiment_id="exp_prof")
    ab = BinomialABTest(prof_ledger)

    pr = cProfile.Profile()
    pr.enable()

    # Minimal work: design + two updates that trigger one or two looks
    ab.set_design(max_n=400, looks=[0.5, 1.0])
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})  # I≈0.5
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})  # I≈1.0

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
    ps.print_stats(r"(insert|raw_sql|flush|write_from|write|uuid|now)")
    return s.getvalue()


# ==========================
# Script entry: profiling
# ==========================

if __name__ == "__main__":
    # When run as a script: create an in-memory DuckDB, ensure table, run profiling,
    # and print the pstats summary to stdout.
    con = ibis.connect("duckdb://")
    con.raw_sql("""
        CREATE TABLE IF NOT EXISTS ledger (
          uuid         TEXT,
          ts           TIMESTAMP,
          pkg_version  TEXT,
          payload_type TEXT,
          payload      JSON,
          labels       JSON
        );
    """)
    out = run_profile_demo(con)
    print("=== cProfile (filtered) ===")
    print("Ordered by: cumulative time")
    print(out)
