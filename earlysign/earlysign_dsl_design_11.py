r"""
Framework + Apps with on-the-fly alpha-spending boundaries (Lan–DeMets)
======================================================================

- Each INSERT timestamp is evaluated at the database side via `now()` (no Python ts handoff).
- Every read inside `update()` uses a *frozen past* view cut at the latest `observation` timestamp
  for the same label scope, i.e., `ts < latest_observation_ts`. This avoids self-reference/order issues.
- Boundaries and stats are expressed purely in Ibis algebra (no Python literal injection).
- In addition to z, we also store a Gaussian-approx e-value: e = exp(z^2/2).

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
>>> # Check decisions via typed view
>>> V_base = ABRowsBase(ab_ledger).typed_view()
>>> decisions = V_base.filter(lambda r: r.payload_type == "decision").order_by(lambda r: r.ts)
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
>>> # Inspect ledger content (illustrative)
>>> _ = ab_ledger.show().execute()

# (B) E-process (mixture over fixed thetas, Ville alarm at 1/alpha)
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
>>> (round(float(rowE["e_value"]), 2), bool(rowE["alarm"]))
(26.84, True)
"""

from __future__ import annotations

import uuid as _py_uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import ibis

# =========================================================
# Backend helpers & tiny plumbing (DuckDB / BigQuery aware)
# =========================================================

_PKG_VERSION = "earlysign==dev"
_E_CONST = 2.718281828459045  # literal e (avoid ibis.e() which may not exist)


def _backend(con: Any) -> str:
    name = getattr(con, "name", None) or type(con).__name__.lower()
    if "duck" in name:
        return "duckdb"
    if "bigquery" in name:
        return "bigquery"
    return "duckdb"


def _uuid_expr() -> ibis.Expr:
    if hasattr(ibis, "uuid"):
        return ibis.uuid().cast("string")
    return ibis.literal(_py_uuid.uuid4().hex)


def _now_ts() -> ibis.Expr:
    # Evaluate on DB side at execution time
    return ibis.now().cast("timestamp")


def _to_json_wrapper_sql(con: Any, col_sql: str) -> str:
    be = _backend(con)
    if be == "duckdb":
        return f"to_json({col_sql})"
    else:
        return f"TO_JSON({col_sql})"


def _labels_where_sql(con: Any, labels: Dict[str, object]) -> str:
    if not labels:
        return "TRUE"
    be = _backend(con)
    parts = []
    for k, v in labels.items():
        sv = str(v).replace("'", "''")
        if be == "duckdb":
            parts.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
        else:
            parts.append(f"JSON_VALUE(labels, '$.{k}') = '{sv}'")
    return " AND ".join(parts)


# Safe arithmetic helpers (avoid ibis.nullif dependency)
def _safe_inv(x: ibis.Expr) -> ibis.Expr:
    return (x == 0).ifelse(ibis.null(), 1.0 / x)


def _safe_div(num: ibis.Expr, den: ibis.Expr) -> ibis.Expr:
    return num / ((den == 0).ifelse(ibis.null(), den))


# ==========
# Ledger API
# ==========

@dataclass(frozen=True)
class Ledger:
    con: Any
    table: str = "ledger"
    labels: Dict[str, object] = field(default_factory=dict)

    def bind(self, **labels: object) -> "Ledger":
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table, merged)

    def view_json(self) -> ibis.Expr:
        where = _labels_where_sql(self.con, self.labels)
        sql = f"""
SELECT
  uuid,
  CAST(ts AS TIMESTAMP) AS ts,
  pkg_version,
  payload_type,
  payload,
  labels
FROM {self.table}
WHERE {where}
"""
        return self.con.sql(sql)

    def show(self) -> ibis.Expr:
        where = _labels_where_sql(self.con, self.labels)
        sql = f"""
SELECT
  CAST(payload_type AS VARCHAR) AS payload_type,
  payload,
  labels
FROM {self.table}
WHERE {where}
ORDER BY ts
"""
        return self.con.sql(sql)


# ==========================================
# Ibis-only math: Φ, Φ^{-1}, alpha spending
# ==========================================

def _tanh_expr(x: ibis.Expr) -> ibis.Expr:
    e2x = (x * 2.0).exp()
    return (e2x - 1.0) / (e2x + 1.0)


def ibis_norm_cdf(x: ibis.Expr) -> ibis.Expr:
    a = 0.044715
    c = (2.0 / 3.141592653589793) ** 0.5  # √(2/π)
    return 0.5 * (1.0 + _tanh_expr(c * (x + a * x * x * x)))


def ibis_norm_ppf(p: ibis.Expr) -> ibis.Expr:
    eps = 1e-12
    p_clip = ibis.greatest(ibis.least(p, 1.0 - eps), eps)

    plow = 0.02425
    phigh = 1.0 - plow

    a0 = -3.969683028665376e+01
    a1 =  2.209460984245205e+02
    a2 = -2.759285104469687e+02
    a3 =  1.383577518672690e+02
    a4 = -3.066479806614716e+01
    a5 =  2.506628277459239e+00

    b0 = -5.447609879822406e+01
    b1 =  1.615858368580409e+02
    b2 = -1.556989798598866e+02
    b3 =  6.680131188771972e+01
    b4 = -1.328068155288572e+01

    c0 = -7.784894002430293e-03
    c1 = -3.223964580411365e-01
    c2 = -2.400758277161838e+00
    c3 = -2.549732539343734e+00
    c4 =  4.374664141464968e+00
    c5 =  2.938163982698783e+00

    d0 =  7.784695709041462e-03
    d1 =  3.224671290700398e-01
    d2 =  2.445134137142996e+00
    d3 =  3.754408661907416e+00

    is_low  = p_clip < plow
    is_high = p_clip > phigh

    q = p_clip - 0.5
    r = q * q
    num_c = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
    den_c = (((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0)
    x_c = num_c / den_c

    ql = (-2.0 * (p_clip).log()).sqrt()
    num_l = (((((c0 * ql + c1) * ql + c2) * ql + c3) * ql + c4) * ql + c5)
    den_l = ((((d0 * ql + d1) * ql + d2) * ql + d3) * ql + 1.0)
    x_l = num_l / den_l

    qh = (-2.0 * (1.0 - p_clip).log()).sqrt()
    num_h = -(((((c0 * qh + c1) * qh + c2) * qh + c3) * qh + c4) * qh + c5)
    den_h = ((((d0 * qh + d1) * qh + d2) * qh + d3) * qh + 1.0)
    x_h = num_h / den_h

    return ibis.cases((is_low, x_l), (is_high, x_h), else_=x_c)


def ibis_alpha_spent(t: ibis.Expr, alpha: ibis.Expr, family: ibis.Expr) -> ibis.Expr:
    t_clip = ibis.greatest(ibis.least(t, 1.0), 1e-12)
    fam = family.lower()

    of = ibis.literal("obrien_fleming")
    pc = ibis.literal("pocock")

    z = ibis_norm_ppf(1.0 - alpha / 2.0)
    of_spend = 2.0 - 2.0 * ibis_norm_cdf(z / t_clip.sqrt())
    pc_spend = alpha * (1.0 + ibis.literal(_E_CONST - 1.0) * t_clip).log()

    return ibis.cases((fam == of, of_spend), (fam == pc, pc_spend), else_=pc_spend)


# ==================================
# Minimal SQL batcher (semicolon txn)
# ==================================

@dataclass
class LedgerSessionSQL:
    ledger: Ledger
    _stmts: list[str] = field(default_factory=list)

    def __enter__(self) -> "LedgerSessionSQL":
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None or not self._stmts:
            return
        joined = "BEGIN;\n" + ";\n".join(self._stmts) + ";\nCOMMIT;"
        self.ledger.con.raw_sql(joined)
        self._stmts.clear()

    def _insert_select_sql(
        self,
        src: ibis.Expr,
        payload_type: str,
        payload_items: Dict[str, ibis.Expr],
        *,
        extra_labels: Optional[Dict[str, object]] = None,
    ) -> str:
        labels_map = dict(self.ledger.labels)
        if extra_labels:
            labels_map.update(extra_labels)

        row = src.select(
            uuid=_uuid_expr().name("uuid"),
            ts=_now_ts().name("ts"),
            pkg_version=ibis.literal(_PKG_VERSION).name("pkg_version"),
            payload_type=ibis.literal(payload_type).name("payload_type"),
            payload=ibis.struct(payload_items).name("payload"),
            labels=ibis.struct({k: ibis.literal(str(v)).cast("string") for k, v in labels_map.items()}).name("labels"),
        )
        row_sql = row.compile()
        pj = _to_json_wrapper_sql(self.ledger.con, "payload")
        lj = _to_json_wrapper_sql(self.ledger.con, "labels")
        return f"""
INSERT INTO {self.ledger.table} (uuid, ts, pkg_version, payload_type, payload, labels)
SELECT uuid, ts, pkg_version, payload_type, {pj}, {lj}
FROM ({row_sql}) AS t
""".strip()

    def insert_from_select(
        self,
        src: ibis.Expr,
        payload_type: str,
        exprs: Dict[str, ibis.Expr],
        *,
        extra_labels: Optional[Dict[str, object]] = None,
    ) -> None:
        self._stmts.append(self._insert_select_sql(src, payload_type, exprs, extra_labels=extra_labels))

    def insert_row(
        self,
        payload_type: str,
        exprs: Dict[str, ibis.Expr],
        *,
        extra_labels: Optional[Dict[str, object]] = None,
    ) -> None:
        anchor = self.ledger.view_json().limit(0)
        self._stmts.append(self._insert_select_sql(anchor, payload_type, exprs, extra_labels=extra_labels))


# ==========================
# Reader-ish frozen helpers
# ==========================

class _R:
    @staticmethod
    def _latest_observation_ts_rel(ledger: Ledger) -> ibis.Expr:
        where = _labels_where_sql(ledger.con, ledger.labels)
        sql = f"""
SELECT MAX(ts) AS ts
FROM {ledger.table}
WHERE {where} AND payload_type = 'observation'
"""
        return ledger.con.sql(sql)

    @staticmethod
    def latest_by_type_before_tsrel(ledger: Ledger, payload_type: str, ts_rel: ibis.Expr) -> ibis.Expr:
        where = _labels_where_sql(ledger.con, ledger.labels)
        ts_rel_sql = ts_rel.select(ts_rel.ts.name("ts0")).compile()
        sql = f"""
WITH cut AS ({ts_rel_sql})
SELECT
  uuid, CAST(ts AS TIMESTAMP) AS ts, pkg_version, payload_type, payload, labels
FROM {ledger.table}
WHERE {where} AND payload_type = '{payload_type}' AND ts < (SELECT ts0 FROM cut)
ORDER BY ts DESC
LIMIT 1
"""
        return ledger.con.sql(sql)

    @staticmethod
    def looks_from_design_before_tsrel(ledger: Ledger, ts_rel: ibis.Expr) -> ibis.Expr:
        be = _backend(ledger.con)
        latest = _R.latest_by_type_before_tsrel(ledger, "design", ts_rel)
        latest_ts_sql = f"({latest.select(latest.ts.name('ts')).compile()})"
        where = _labels_where_sql(ledger.con, ledger.labels)
        if be == "duckdb":
            sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  b.ts::TIMESTAMP AS ts,
  CAST(je.key AS BIGINT) + 1 AS look,
  CAST(je.value AS DOUBLE)   AS planned_t
FROM {ledger.table} b,
     LATERAL json_each(b.payload, '$.looks') AS je
WHERE {where}
  AND b.payload_type = 'design'
  AND b.ts = (SELECT ts FROM latest)
ORDER BY look
"""
        else:
            sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  b.ts AS ts,
  SAFE_CAST(OFFSET + 1 AS INT64) AS look,
  CAST(JSON_VALUE(elem) AS FLOAT64) AS planned_t
FROM {ledger.table} b,
UNNEST(JSON_QUERY_ARRAY(b.payload, '$.looks')) AS elem WITH OFFSET
WHERE {where}
  AND b.payload_type = 'design'
  AND b.ts = (SELECT ts FROM latest)
ORDER BY look
"""
        return ledger.con.sql(sql)

    @staticmethod
    def design_params_before_tsrel(ledger: Ledger, ts_rel: ibis.Expr) -> ibis.Expr:
        be = _backend(ledger.con)
        latest = _R.latest_by_type_before_tsrel(ledger, "design", ts_rel)
        latest_ts_sql = f"({latest.select(latest.ts.name('ts')).compile()})"
        where = _labels_where_sql(ledger.con, ledger.labels)
        if be == "duckdb":
            sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE) AS Nmax,
  CAST(json_extract(payload, '$.alpha') AS DOUBLE)         AS alpha,
  json_extract_string(payload, '$.spending_family')        AS family
FROM {ledger.table}
WHERE {where} AND payload_type='design'
  AND ts = (SELECT ts FROM latest)
"""
        else:
            sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  CAST(JSON_VALUE(payload, '$.planned_max_n') AS FLOAT64) AS Nmax,
  CAST(JSON_VALUE(payload, '$.alpha') AS FLOAT64)         AS alpha,
  JSON_VALUE(payload, '$.spending_family')                AS family
FROM {ledger.table}
WHERE {where} AND payload_type='design'
  AND ts = (SELECT ts FROM latest)
"""
        return ledger.con.sql(sql)

    @staticmethod
    def latest_snapshot_before_tsrel(ledger: Ledger, ts_rel: ibis.Expr) -> ibis.Expr:
        be = _backend(ledger.con)
        latest = _R.latest_by_type_before_tsrel(ledger, "snapshot", ts_rel)
        latest_ts_sql = f"({latest.select(latest.ts.name('ts')).compile()})"
        where = _labels_where_sql(ledger.con, ledger.labels)
        if be == "duckdb":
            s_sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  ts::TIMESTAMP AS ts,
  CAST(json_extract(payload, '$.nA') AS DOUBLE) AS nA,
  CAST(json_extract(payload, '$.mA') AS DOUBLE) AS mA,
  CAST(json_extract(payload, '$.nB') AS DOUBLE) AS nB,
  CAST(json_extract(payload, '$.mB') AS DOUBLE) AS mB
FROM {ledger.table}
WHERE {where} AND payload_type='snapshot'
  AND ts = (SELECT ts FROM latest)
"""
            zero_sql = "SELECT TIMESTAMP '1970-01-01 00:00:00' AS ts, 0.0 AS nA, 0.0 AS mA, 0.0 AS nB, 0.0 AS mB"
        else:
            s_sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  ts AS ts,
  CAST(JSON_VALUE(payload, '$.nA') AS FLOAT64) AS nA,
  CAST(JSON_VALUE(payload, '$.mA') AS FLOAT64) AS mA,
  CAST(JSON_VALUE(payload, '$.nB') AS FLOAT64) AS nB,
  CAST(JSON_VALUE(payload, '$.mB') AS FLOAT64) AS mB
FROM {ledger.table}
WHERE {where} AND payload_type='snapshot'
  AND ts = (SELECT ts FROM latest)
"""
            zero_sql = "SELECT TIMESTAMP('1970-01-01 00:00:00') AS ts, 0.0 AS nA, 0.0 AS mA, 0.0 AS nB, 0.0 AS mB"
        sql = f"""
WITH s AS ({s_sql}),
z AS ({zero_sql})
SELECT * FROM s
UNION ALL
SELECT * FROM z WHERE NOT EXISTS (SELECT 1 FROM s)
"""
        return ledger.con.sql(sql)

    @staticmethod
    def obs_sum_after_until_tsrel(ledger: Ledger, ts_rel_from: ibis.Expr, ts_rel_to: ibis.Expr) -> ibis.Expr:
        be = _backend(ledger.con)
        where = _labels_where_sql(ledger.con, ledger.labels)
        t_from_sql = ts_rel_from.select(ts_rel_from.ts.name("ts_from")).compile()
        t_to_sql = ts_rel_to.select(ts_rel_to.ts.name("ts_to")).compile()
        if be == "duckdb":
            sql = f"""
WITH t0 AS ({t_from_sql}), cut AS ({t_to_sql})
SELECT
  COALESCE(SUM(CAST(json_extract(payload, '$.nA') AS DOUBLE)), 0.0) AS d_nA,
  COALESCE(SUM(CAST(json_extract(payload, '$.mA') AS DOUBLE)), 0.0) AS d_mA,
  COALESCE(SUM(CAST(json_extract(payload, '$.nB') AS DOUBLE)), 0.0) AS d_nB,
  COALESCE(SUM(CAST(json_extract(payload, '$.mB') AS DOUBLE)), 0.0) AS d_mB
FROM {ledger.table}
WHERE {where}
  AND payload_type = 'observation'
  AND ts > (SELECT ts_from FROM t0)
  AND ts <= (SELECT ts_to   FROM cut)
"""
        else:
            sql = f"""
WITH t0 AS ({t_from_sql}), cut AS ({t_to_sql})
SELECT
  COALESCE(SUM(CAST(JSON_VALUE(payload, '$.nA') AS FLOAT64)), 0.0) AS d_nA,
  COALESCE(SUM(CAST(JSON_VALUE(payload, '$.mA') AS FLOAT64)), 0.0) AS d_mA,
  COALESCE(SUM(CAST(JSON_VALUE(payload, '$.nB') AS FLOAT64)), 0.0) AS d_nB,
  COALESCE(SUM(CAST(JSON_VALUE(payload, '$.mB') AS FLOAT64)), 0.0) AS d_mB
FROM {ledger.table}
WHERE {where}
  AND payload_type = 'observation'
  AND ts > (SELECT ts_from FROM t0)
  AND ts <= (SELECT ts_to   FROM cut)
"""
        return ledger.con.sql(sql)

    # ---- E-process readers ----

    @staticmethod
    def e_latest_design_before_tsrel(ledger: Ledger, ts_rel: ibis.Expr) -> ibis.Expr:
        be = _backend(ledger.con)
        latest = _R.latest_by_type_before_tsrel(ledger, "e_design", ts_rel)
        latest_ts_sql = f"({latest.select(latest.ts.name('ts')).compile()})"
        where = _labels_where_sql(ledger.con, ledger.labels)
        if be == "duckdb":
            sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  CAST(json_extract(payload, '$.alpha') AS DOUBLE) AS alpha,
  json_extract(payload, '$.thetas') AS thetas_json
FROM {ledger.table}
WHERE {where} AND payload_type='e_design'
  AND ts = (SELECT ts FROM latest)
"""
        else:
            sql = f"""
WITH latest AS {latest_ts_sql}
SELECT
  CAST(JSON_VALUE(payload, '$.alpha') AS FLOAT64) AS alpha,
  JSON_QUERY(payload, '$.thetas') AS thetas_json
FROM {ledger.table}
WHERE {where} AND payload_type='e_design'
  AND ts = (SELECT ts FROM latest)
"""
        return ledger.con.sql(sql)

    @staticmethod
    def e_sum_and_count_upto_latest(ledger: Ledger, ts_rel_to: ibis.Expr) -> ibis.Expr:
        be = _backend(ledger.con)
        where = _labels_where_sql(ledger.con, ledger.labels)
        t_to_sql = ts_rel_to.select(ts_rel_to.ts.name("ts_to")).compile()
        if be == "duckdb":
            sql = f"""
WITH cut AS ({t_to_sql})
SELECT
  COALESCE(SUM(CAST(json_extract(payload, '$.x') AS DOUBLE)), 0.0) AS S,
  CAST(COUNT(*) AS BIGINT) AS n
FROM {ledger.table}
WHERE {where}
  AND payload_type = 'e_obs'
  AND ts <= (SELECT ts_to FROM cut)
"""
        else:
            sql = f"""
WITH cut AS ({t_to_sql})
SELECT
  COALESCE(SUM(CAST(JSON_VALUE(payload, '$.x') AS FLOAT64)), 0.0) AS S,
  CAST(COUNT(*) AS INT64) AS n
FROM {ledger.table}
WHERE {where}
  AND payload_type = 'e_obs'
  AND ts <= (SELECT ts_to FROM cut)
"""
        return ledger.con.sql(sql)

    @staticmethod
    def e_latest_state_before_tsrel(ledger: Ledger, ts_rel: ibis.Expr) -> ibis.Expr:
        return _R.latest_by_type_before_tsrel(ledger, "e_state", ts_rel)


# ==========================
# Binomial A/B application
# ==========================

class BinomialABTest:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, max_n: int, looks: list[float], alpha: float = 0.05, spending: str = "obrien_fleming") -> None:
        looks = [float(x) for x in looks]
        with LedgerSessionSQL(self.ledger) as s:
            s.insert_row(
                "design",
                {
                    "planned_max_n": ibis.literal(float(max_n)),
                    "looks": ibis.array([ibis.literal(float(x)) for x in looks]),
                    "alpha": ibis.literal(float(alpha)),
                    "spending_family": ibis.literal(str(spending)),
                }
            )

    def update(self, payload: Dict[str, int]) -> None:
      """Insert one observation and write derived rows using a frozen-past cut.

      Semantics:
      - Insert the observation immediately with ts evaluated on DB (now()).
      - All subsequent reads in this call use a frozen cut at the latest
        observation timestamp (same label scope): ts <= latest_observation_ts.
      - Stats/boundaries are expressed purely in Ibis algebra.
      - E-value (Gaussian approx) is also stored: e = exp(z^2 / 2).
      """
      nA_add = float(payload["nA"]); mA_add = float(payload["mA"])
      nB_add = float(payload["nB"]); mB_add = float(payload["mB"])

      # (0) Insert the observation immediately (ts evaluated on DB)
      anchor = self.ledger.view_json().limit(0)
      obs_row = anchor.select(
          _uuid_expr().name("uuid"),
          _now_ts().name("ts"),
          ibis.literal(_PKG_VERSION).name("pkg_version"),
          ibis.literal("observation").name("payload_type"),
          ibis.struct({
              "nA": ibis.literal(nA_add).cast("float64"),
              "mA": ibis.literal(mA_add).cast("float64"),
              "nB": ibis.literal(nB_add).cast("float64"),
              "mB": ibis.literal(mB_add).cast("float64"),
          }).name("payload"),
          ibis.struct({k: ibis.literal(str(v)).cast("string") for k, v in self.ledger.labels.items()}).name("labels"),
      )
      row_sql = obs_row.compile()
      self.ledger.con.raw_sql(f"""
  INSERT INTO {self.ledger.table} (uuid, ts, pkg_version, payload_type, payload, labels)
  SELECT uuid, ts, pkg_version, payload_type, {_to_json_wrapper_sql(self.ledger.con, "payload")}, {_to_json_wrapper_sql(self.ledger.con, "labels")}
  FROM ({row_sql}) t
  """)

      # (1) Remaining writes batched in a single transaction (BEGIN...COMMIT)
      with LedgerSessionSQL(self.ledger) as s:
          # Frozen cut at latest observation ts (same labels)
          T_latest = _R._latest_observation_ts_rel(self.ledger)

          # Read previous snapshot (or zeros), design params, and looks as of the cut
          S_prev = _R.latest_snapshot_before_tsrel(self.ledger, T_latest)
          Dpars  = _R.design_params_before_tsrel(self.ledger, T_latest)
          Looks  = _R.looks_from_design_before_tsrel(self.ledger, T_latest)

          # Aggregate new observations between previous snapshot ts and the cut
          T_from  = S_prev.select(S_prev.ts.name("ts"))
          Obs_inc = _R.obs_sum_after_until_tsrel(self.ledger, T_from, T_latest)

          # Accumulate counts to "now" (at the cut)
          before = S_prev.cross_join(Obs_inc).select(
              (S_prev.nA + Obs_inc.d_nA).name("nA_before"),
              (S_prev.mA + Obs_inc.d_mA).name("mA_before"),
              (S_prev.nB + Obs_inc.d_nB).name("nB_before"),
              (S_prev.mB + Obs_inc.d_mB).name("mB_before"),
          )
          now = before.cross_join(Dpars).select(
              before.nA_before.name("nA"),
              before.mA_before.name("mA"),
              before.nB_before.name("nB"),
              before.mB_before.name("mB"),
              (before.nA_before + before.nB_before).name("n_before"),
              Dpars.Nmax.name("Nmax"),
              Dpars.alpha.name("alpha"),
              Dpars.family.name("family"),
          )

          # Snapshot at the cut
          s.insert_from_select(now, "snapshot", {"nA": now.nA, "mA": now.mA, "nB": now.nB, "mB": now.mB})

          # Information times I0 (previous total) and I1 (current total)
          I0_tbl = now.select((_safe_div(now.n_before, now.Nmax)).name("I0"))
          I1_tbl = now.select((_safe_div(now.nA + now.nB, now.Nmax)).name("I1"))
          s.insert_from_select(I1_tbl, "info", {"info_time": I1_tbl.I1})

          # Due detection: smallest planned_t with I0 < t <= I1
          due_candidates = Looks.cross_join(I0_tbl).cross_join(I1_tbl).select(
              Looks.look.name("look"),
              Looks.planned_t.name("planned_t"),
              I0_tbl.I0.name("I0"),
              I1_tbl.I1.name("I1"),
          ).filter(lambda r: (r.I0 < r.planned_t) & (r.planned_t <= r.I1))
          min_due = due_candidates.aggregate(min_planned_t=due_candidates.planned_t.min())
          due = due_candidates.join(min_due, predicates=[due_candidates.planned_t == min_due.min_planned_t]).limit(1)

          # --- All stats and boundaries computed on a single base relation to avoid parent-mismatch ---
          Zbase = now.cross_join(I0_tbl).cross_join(I1_tbl)

          # Wald z for two-sample binomial with pooled variance
          pA = _safe_div(Zbase.mA, Zbase.nA)
          pB = _safe_div(Zbase.mB, Zbase.nB)
          p_all = _safe_div(Zbase.mA + Zbase.mB, Zbase.nA + Zbase.nB)
          se = (p_all * (1.0 - p_all) * (_safe_inv(Zbase.nA) + _safe_inv(Zbase.nB))).sqrt()
          z = ((pB - pA) / (se == 0).ifelse(ibis.null(), se)).name("z")

          # Lan–DeMets local alpha and boundary
          A1 = ibis_alpha_spent(Zbase.I1, Zbase.alpha, Zbase.family)
          A0 = ibis_alpha_spent(Zbase.I0, Zbase.alpha, Zbase.family)
          local_alpha = ibis.greatest(A1 - A0, 1e-16).name("local_alpha")
          boundary = ibis_norm_ppf(1.0 - local_alpha / 2.0).name("boundary")

          # Gaussian-approx e-value
          e_gauss = ((z * z) / 2.0).exp().name("e_value")

          # Persist stat row
          Zrow = Zbase.select(
              z.name("z"),
              local_alpha.name("local_alpha"),
              boundary.name("boundary"),
              e_gauss.name("e_value"),
              Zbase.I1.name("info_time"),
          )
          s.insert_from_select(Zrow, "stat", {"z": Zrow.z, "e_value": Zrow.e_value})

          # Decision if a due look exists
          Decision = Zrow.cross_join(due).select(
              look=due.look,
              planned_t=due.planned_t,
              info_time=Zrow.info_time,
              z=Zrow.z,
              boundary=Zrow.boundary,
              e_value=Zrow.e_value,
              action=(Zrow.z.abs() >= Zrow.boundary).ifelse("stop_efficacy", "continue").name("action"),
          )
          s.insert_from_select(
              Decision, "decision",
              {
                  "look": Decision.look,
                  "planned_t": Decision.planned_t,
                  "info_time": Decision.info_time,
                  "z": Decision.z,
                  "boundary": Decision.boundary,
                  "e_value": Decision.e_value,
                  "action": Decision.action,
              },
          )

# ==========================
# E-process: Normal mixture
# ==========================

class ENormalMixture:
    """Mixture-e martingale for N(0,1) vs {N(theta,1)} with fixed theta grid.

    e_n = average_j exp(theta_j * S_n - n * theta_j^2 / 2), where
    S_n is the cumulative sum of observed x_t and n is the count.
    Alarm triggers when e_n >= 1/alpha (Ville).
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        thetas = [float(t) for t in thetas]
        with LedgerSessionSQL(self.ledger) as s:
            s.insert_row(
                "e_design",
                {
                    "alpha": ibis.literal(float(alpha)),
                    "thetas": ibis.array([ibis.literal(float(t)) for t in thetas]),
                }
            )

    def update(self, *, x: float) -> None:
        # Insert observation
        anchor = self.ledger.view_json().limit(0)
        obs_row = anchor.select(
            _uuid_expr().name("uuid"),
            _now_ts().name("ts"),
            ibis.literal(_PKG_VERSION).name("pkg_version"),
            ibis.literal("e_obs").name("payload_type"),
            ibis.struct({"x": ibis.literal(float(x)).cast("float64")}).name("payload"),
            ibis.struct({k: ibis.literal(str(v)).cast("string") for k, v in self.ledger.labels.items()}).name("labels"),
        )
        row_sql = obs_row.compile()
        self.ledger.con.raw_sql(f"""
INSERT INTO {self.ledger.table} (uuid, ts, pkg_version, payload_type, payload, labels)
SELECT uuid, ts, pkg_version, payload_type, {_to_json_wrapper_sql(self.ledger.con, "payload")}, {_to_json_wrapper_sql(self.ledger.con, "labels")}
FROM ({row_sql}) t
""")

        # Compute e-state at latest cut and insert
        with LedgerSessionSQL(self.ledger) as s:
            T_latest = _R._latest_observation_ts_rel(self.ledger)  # latest e_obs ts
            D = _R.e_latest_design_before_tsrel(self.ledger, T_latest)
            Sn = _R.e_sum_and_count_upto_latest(self.ledger, T_latest)

            # Compute mixture e-value from S and n and stored theta grid
            be = _backend(self.ledger.con)
            if be == "duckdb":
                # UNNEST JSON array: json_extract(thetas_json, '$') returns JSON list; use json_each to iterate
                sql_emix = f"""
WITH pars AS ({D.compile()}),
     sagg AS ({Sn.compile()}),
     theta AS (
       SELECT CAST(je.value AS DOUBLE) AS theta
       FROM pars, LATERAL json_each(pars.thetas_json) AS je
     )
SELECT
  sagg.S AS S,
  sagg.n AS n,
  (SELECT AVG(EXP(theta * sagg.S - (sagg.n * theta * theta) / 2.0)) FROM theta) AS e_value,
  (SELECT CAST(pars.alpha AS DOUBLE) FROM pars) AS alpha
FROM sagg
"""
            else:
                # BigQuery JSON array unnest
                sql_emix = f"""
WITH pars AS ({D.compile()}),
     sagg AS ({Sn.compile()}),
     theta AS (
       SELECT CAST(JSON_VALUE(elem) AS FLOAT64) AS theta
       FROM pars, UNNEST(JSON_QUERY_ARRAY(pars.thetas_json)) AS elem
     )
SELECT
  sagg.S AS S,
  sagg.n AS n,
  (SELECT AVG(EXP(theta * sagg.S - (sagg.n * theta * theta) / 2.0)) FROM theta) AS e_value,
  (SELECT CAST(pars.alpha AS FLOAT64) FROM pars) AS alpha
FROM sagg
"""
            EM = self.ledger.con.sql(sql_emix)

            ins = EM.select(
                _uuid_expr().name("uuid"),
                _now_ts().name("ts"),
                ibis.literal(_PKG_VERSION).name("pkg_version"),
                ibis.literal("e_state").name("payload_type"),
                ibis.struct({
                    "e_value": EM.e_value,
                    "alarm": (EM.e_value >= (1.0 / EM.alpha)).ifelse(True, False),
                }).name("payload"),
                ibis.struct({k: ibis.literal(str(v)).cast("string") for k, v in self.ledger.labels.items()}).name("labels"),
            )
            ins_sql = ins.compile()
            self.ledger.con.raw_sql(f"""
INSERT INTO {self.ledger.table} (uuid, ts, pkg_version, payload_type, payload, labels)
SELECT uuid, ts, pkg_version, payload_type, {_to_json_wrapper_sql(self.ledger.con, "payload")}, {_to_json_wrapper_sql(self.ledger.con, "labels")}
FROM ({ins_sql}) t
""")


# ==========================
# Typed views for doctests
# ==========================

@dataclass
class ABRowsBase:
    ledger: Ledger

    def typed_view(self) -> ibis.Expr:
        be = _backend(self.ledger.con)
        where = _labels_where_sql(self.ledger.con, self.ledger.labels)
        if be == "duckdb":
            sql = f"""
SELECT
  CAST(ts AS TIMESTAMP) AS ts,
  payload_type,
  -- decision fields
  CASE WHEN payload_type='decision' THEN CAST(json_extract(payload,'$.look') AS BIGINT) END AS look,
  CASE WHEN payload_type='decision' THEN CAST(json_extract(payload,'$.planned_t') AS DOUBLE) END AS planned_t,
  CASE WHEN payload_type IN ('decision','info') THEN CAST(json_extract(payload,'$.info_time') AS DOUBLE) END AS info_time,
  CASE WHEN payload_type IN ('decision','stat') THEN CAST(json_extract(payload,'$.z') AS DOUBLE) END AS z,
  CASE WHEN payload_type='decision' THEN CAST(json_extract(payload,'$.boundary') AS DOUBLE) END AS boundary,
  CASE WHEN payload_type IN ('decision','stat') THEN CAST(json_extract(payload,'$.e_value') AS DOUBLE) END AS e_value,
  CASE WHEN payload_type='decision' THEN json_extract_string(payload,'$.action') END AS action,
  -- snapshot fields
  CASE WHEN payload_type='snapshot' THEN CAST(json_extract(payload,'$.nA') AS DOUBLE) END AS nA,
  CASE WHEN payload_type='snapshot' THEN CAST(json_extract(payload,'$.mA') AS DOUBLE) END AS mA,
  CASE WHEN payload_type='snapshot' THEN CAST(json_extract(payload,'$.nB') AS DOUBLE) END AS nB,
  CASE WHEN payload_type='snapshot' THEN CAST(json_extract(payload,'$.mB') AS DOUBLE) END AS mB
FROM {self.ledger.table}
WHERE {where}
"""
        else:
            sql = f"""
SELECT
  ts AS ts,
  payload_type,
  CASE WHEN payload_type='decision' THEN CAST(JSON_VALUE(payload,'$.look') AS INT64) END AS look,
  CASE WHEN payload_type='decision' THEN CAST(JSON_VALUE(payload,'$.planned_t') AS FLOAT64) END AS planned_t,
  CASE WHEN payload_type IN ('decision','info') THEN CAST(JSON_VALUE(payload,'$.info_time') AS FLOAT64) END AS info_time,
  CASE WHEN payload_type IN ('decision','stat') THEN CAST(JSON_VALUE(payload,'$.z') AS FLOAT64) END AS z,
  CASE WHEN payload_type='decision' THEN CAST(JSON_VALUE(payload,'$.boundary') AS FLOAT64) END AS boundary,
  CASE WHEN payload_type IN ('decision','stat') THEN CAST(JSON_VALUE(payload,'$.e_value') AS FLOAT64) END AS e_value,
  CASE WHEN payload_type='decision' THEN JSON_VALUE(payload,'$.action') END AS action,
  CASE WHEN payload_type='snapshot' THEN CAST(JSON_VALUE(payload,'$.nA') AS FLOAT64) END AS nA,
  CASE WHEN payload_type='snapshot' THEN CAST(JSON_VALUE(payload,'$.mA') AS FLOAT64) END AS mA,
  CASE WHEN payload_type='snapshot' THEN CAST(JSON_VALUE(payload,'$.nB') AS FLOAT64) END AS nB,
  CASE WHEN payload_type='snapshot' THEN CAST(JSON_VALUE(payload,'$.mB') AS FLOAT64) END AS mB
FROM {self.ledger.table}
WHERE {where}
"""
        return self.ledger.con.sql(sql)


@dataclass
class EBaseRows:
    ledger: Ledger

    def typed_view(self) -> ibis.Expr:
        be = _backend(self.ledger.con)
        where = _labels_where_sql(self.ledger.con, self.ledger.labels)
        if be == "duckdb":
            sql = f"""
SELECT
  CAST(ts AS TIMESTAMP) AS ts,
  payload_type,
  CASE WHEN payload_type='e_state' THEN CAST(json_extract(payload,'$.e_value') AS DOUBLE) END AS e_value,
  CASE WHEN payload_type='e_state' THEN CAST(json_extract(payload,'$.alarm') AS BOOLEAN) END AS alarm
FROM {self.ledger.table}
WHERE {where}
"""
        else:
            sql = f"""
SELECT
  ts AS ts,
  payload_type,
  CASE WHEN payload_type='e_state' THEN CAST(JSON_VALUE(payload,'$.e_value') AS FLOAT64) END AS e_value,
  CASE WHEN payload_type='e_state' THEN CAST(JSON_VALUE(payload,'$.alarm') AS BOOL) END AS alarm
FROM {self.ledger.table}
WHERE {where}
"""
        return self.ledger.con.sql(sql)


# ==========================
# (Optional) profiling hook
# ==========================

def _ibis_depth(expr: ibis.Expr) -> int:
    """Best-effort estimate of an Ibis expression tree depth."""
    try:
        node = expr.op()
    except Exception:
        return 1
    if not hasattr(node, "args") or node.args is None:
        return 1
    depths = []
    for a in node.args:
        try:
            if isinstance(a, ibis.Expr):
                depths.append(_ibis_depth(a))
            elif hasattr(a, "op"):
                depths.append(_ibis_depth(a))
            elif isinstance(a, (list, tuple)):
                depths.append(max((_ibis_depth(x) for x in a if hasattr(x, "op")), default=0))
            else:
                depths.append(0)
        except Exception:
            depths.append(0)
    return 1 + (max(depths) if depths else 0)


def run_profile_demo(con: Any) -> str:
    """Small smoke run with cProfile; returns text summary incl. Ibis depth."""
    import cProfile, pstats, io

    base = Ledger(con, table="ledger")
    prof = base.bind(experiment_id="exp_prof")
    ab = BinomialABTest(prof)

    pr = cProfile.Profile()
    pr.enable()

    ab.set_design(max_n=400, looks=[0.5, 1.0], alpha=0.05, spending="obrien_fleming")
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})

    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime").print_stats(r"(raw_sql|insert|uuid|now)")

    V = base.view_json()
    stat = V.filter(lambda r: r.payload_type == "stat").order_by(lambda r: r.ts.desc()).limit(1)
    decision = V.filter(lambda r: r.payload_type == "decision").order_by(lambda r: r.ts.desc()).limit(1)

    z_expr = stat.payload["z"].cast("float64")
    e_expr = stat.payload["e_value"].cast("float64")
    b_expr = decision.payload["boundary"].cast("float64")

    s.write("\n[IBIS DEPTH]\n")
    s.write(f"depth_z(stat): {_ibis_depth(z_expr)}\n")
    s.write(f"depth_e_value(stat): {_ibis_depth(e_expr)}\n")
    s.write(f"depth_boundary(decision): {_ibis_depth(b_expr)}\n")
    return s.getvalue()


if __name__ == "__main__":
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
    print(run_profile_demo(con))
