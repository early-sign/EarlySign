"""
Framework + Apps with on-the-fly alpha-spending boundaries (Lan–DeMets) — SQL-only version
==========================================================================================

- Boundaries/decisions are computed entirely in Ibis/SQL (no Python-side inverse CDF at update).
- Uses an Abramowitz–Stegun 7.1.26 approximation Φ(x) built from Ibis primitives.
- Profiling helper prints cProfile (filtered) and an Ibis expression-depth report.
"""

from __future__ import annotations

import math
import uuid as _py_uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import ibis

_PKG_VERSION = "earlysign==dev"


def _json_from_kv(items: Dict[str, ibis.Expr], *, raw_keys: set[str] | None = None) -> ibis.Expr:
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


def ibis_depth(expr: ibis.Expr) -> tuple[int, int]:
    from collections import deque

    def to_node(x):
        if hasattr(x, "op") and callable(getattr(x, "op")):
            try:
                return x.op()
            except Exception:
                return None
        if hasattr(x, "args"):
            return x
        return None

    root = to_node(expr)
    if root is None:
        return (0, 0)

    seen = set()
    q = deque([(root, 1)])
    max_d = 0
    cnt = 0
    while q:
        node, d = q.popleft()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        cnt += 1
        max_d = max(max_d, d)
        args = getattr(node, "args", [])
        for arg in (args if isinstance(args, (list, tuple)) else [args]):
            it = arg if isinstance(arg, (list, tuple)) else (arg,)
            for a in it:
                child = to_node(a)
                if child is not None:
                    q.append((child, d + 1))
    return max_d, cnt


@dataclass(frozen=True)
class Ledger:
    con: Any
    table: str = "ledger"
    labels: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def bind(self, **labels: Union[str, int, float, bool]) -> "Ledger":
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table, merged)

    def view_json(self) -> ibis.Expr:
        where_labels = LedgerReaderBase._labels_where_sql(self.labels)
        sql = f"""
            SELECT
              uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
            FROM {self.table}
            WHERE {where_labels}
        """
        return self.con.sql(sql)

    def show(self) -> ibis.Expr:
        t = self.view_json().order_by("ts")
        try:
            t = t.drop("uuid", "ts", "pkg_version")
        except Exception:
            pass
        return t


def _has_ibis_uuid() -> bool:
    return hasattr(ibis, "uuid")


def _uuid_single() -> ibis.Expr:
    if _has_ibis_uuid():
        return ibis.uuid().cast("string")
    return ibis.literal(_py_uuid.uuid4().hex)


def _uuid_multi(src: ibis.Expr) -> ibis.Expr:
    base = ibis.uuid().cast("string") if _has_ibis_uuid() else ibis.literal(_py_uuid.uuid4().hex)
    first_col = src[list(src.schema().names)[0]]
    w = ibis.window(order_by=[first_col])
    rn = ibis.row_number().over(w)
    rn_str = rn.cast("int64").cast("string").lpad(12, "0")
    return (base + ibis.literal("-")) + rn_str


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
        anchor = ibis.memtable([{"one": 1}])
        items = {k: (v if isinstance(v, ibis.Expr) else ibis.literal(v)) for k, v in payload.items()}
        row = anchor.select(
            uuid=_uuid_single(),
            ts=ibis.now().cast("timestamp(6)"),
            pkg_version=ibis.literal(_PKG_VERSION),
            payload_type=ibis.literal(payload_type),
            payload=_json_from_kv(items, raw_keys=raw_keys or set()),
            labels=self._labels_json_expr(extra_labels),
        )
        self.staged.add(row)

    def write_from(self, src: ibis.Expr, payload_type: str,
                   payload: Dict[str, Union[int, float, str, ibis.Expr]],
                   *, extra_labels: Optional[Dict[str, Union[str, int, float, bool]]] = None,
                   raw_keys: set[str] | None = None) -> None:
        items = {k: (v if isinstance(v, ibis.Expr) else ibis.literal(v)) for k, v in payload.items()}
        row = src.select(
            uuid=_uuid_multi(src),
            ts=ibis.now().cast("timestamp(6)"),
            pkg_version=ibis.literal(_PKG_VERSION),
            payload_type=ibis.literal(payload_type),
            payload=_json_from_kv(items, raw_keys=raw_keys or set()),
            labels=self._labels_json_expr(extra_labels),
        )
        self.staged.add(row)

    def view_json(self) -> ibis.Expr:
        return self.ledger.view_json()

    def flush(self) -> None:
        union = self.staged.union()
        if union is not None:
            self.ledger.con.insert(self.ledger.table, union)


@dataclass(frozen=True)
class LedgerReaderBase:
    ledger: Ledger

    @staticmethod
    def _labels_where_sql(labels: Dict[str, Union[str, int, float, bool]]) -> str:
        if not labels:
            return "TRUE"
        conds: List[str] = []
        for k, v in labels.items():
            sv = str(v).replace("'", "''")
            conds.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
        return " AND ".join(conds)

    def base_view(self) -> ibis.Expr:
        return self.ledger.view_json()

    def latest_row(self, payload_type: str) -> ibis.Expr:
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
    def latest(self) -> ibis.Expr:
        base = self.latest_row("design")
        sql = f"""
SELECT
  ts,
  TRY_CAST(json_extract(payload, '$.planned_max_n')   AS DOUBLE) AS planned_max_n,
  TRY_CAST(json_extract(payload, '$.alpha')           AS DOUBLE) AS alpha,
  json_extract_string(payload, '$.spending_family')   AS spending_family,
  TRY_CAST(json_extract(payload, '$.z_alpha_over_2')  AS DOUBLE) AS z_alpha_over_2,
  payload                                             AS payload_json
FROM ({base.compile()})
"""
        return self.ledger.con.sql(sql)


class ABDesignLooks(LedgerReaderBase):
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


def _phi_inv(p: float) -> float:
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


class BinomialABTest:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self._last_depths: Dict[str, tuple[int, int]] = {}

    def set_design(self, *, max_n: int, looks: Sequence[float], alpha: float = 0.05, spending: str = "obrien_fleming") -> None:
        looks = [float(x) for x in looks]
        looks_text = "[" + ",".join(str(u) for u in looks) + "]"
        looks_json  = ibis.literal(looks_text)
        z_alpha_over_2 = float(_phi_inv(1.0 - float(alpha)/2.0))
        with LedgerSession(self.ledger) as s:
            s.write(
                "design",
                {
                    "planned_max_n": int(max_n),
                    "looks": looks_json,
                    "alpha": float(alpha),
                    "spending_family": spending,
                    "z_alpha_over_2": z_alpha_over_2,
                },
                raw_keys={"looks"}
            )

    def update(self, payload: Dict[str, int]) -> None:
        nA_add = float(payload["nA"])
        mA_add = float(payload["mA"])
        nB_add = float(payload["nB"])
        mB_add = float(payload["mB"])

        with LedgerSession(self.ledger) as s:
            s.write("observation", payload)

            D_latest = ABDesignLatest(self.ledger).latest()
            Looks    = ABDesignLooks(self.ledger).table()
            S_prev   = ABSnapshotLatest(self.ledger).latest()
            Obs_since= ABObsSince(self.ledger).sum_after(S_prev.select("ts"))

            Dpars = D_latest.select(
                Nmax   = D_latest.planned_max_n,
                alpha  = D_latest.alpha,
                family = D_latest.spending_family,
                zfix   = D_latest.z_alpha_over_2,
            )

            before = S_prev.cross_join(Obs_since).select(
                nA_before = S_prev.nA + Obs_since.d_nA,
                mA_before = S_prev.mA + Obs_since.d_mA,
                nB_before = S_prev.nB + Obs_since.d_nB,
                mB_before = S_prev.mB + Obs_since.d_mB,
            )

            add_tbl = ibis.memtable([{"nA_add": nA_add, "mA_add": mA_add, "nB_add": nB_add, "mB_add": mB_add}])
            now = before.cross_join(add_tbl).cross_join(Dpars).select(
                nA = before.nA_before + add_tbl.nA_add,
                mA = before.mA_before + add_tbl.mA_add,
                nB = before.nB_before + add_tbl.nB_add,
                mB = before.mB_before + add_tbl.mB_add,
                Nmax = Dpars.Nmax,
                alpha = Dpars.alpha,
                family = Dpars.family,
                zfix = Dpars.zfix,
                n_before = before.nA_before + before.nB_before
            )

            s.write_from(now, "snapshot", {"nA": now.nA, "mA": now.mA, "nB": now.nB, "mB": now.mB})

            I1 = (now.nA + now.nB) / now.Nmax.nullif(0)
            s.write_from(now, "info", {"info_time": I1})

            I0_tbl = now.select(I0=(now.n_before / now.Nmax.nullif(0)))
            due_candidates = Looks.cross_join(I0_tbl).cross_join(now.select(I1=I1)).select(
                look=Looks.look, planned_t=Looks.planned_t, I0=I0_tbl.I0, I1=I1
            ).filter(lambda r: (r.I0 < r.planned_t) & (r.planned_t <= r.I1))

            min_due = due_candidates.aggregate(min_planned_t=due_candidates.planned_t.min())
            due = due_candidates.join(min_due, predicates=[due_candidates.planned_t == min_due.min_planned_t]).limit(1)

            p_all  = ((now.mA + now.mB) / (now.nA + now.nB))
            se_all = (p_all * (1 - p_all) * (1 / now.nA + 1 / now.nB)).sqrt().nullif(0)
            z_all  = ((now.mB / now.nB) - (now.mA / now.nA)) / se_all
            z_abs  = z_all.abs()

            INV_SQRT_2PI = ibis.literal(1.0 / math.sqrt(2.0 * math.pi))
            p_const = ibis.literal(0.2316419)
            b1 = ibis.literal(0.319381530)
            b2 = ibis.literal(-0.356563782)
            b3 = ibis.literal(1.781477937)
            b4 = ibis.literal(-1.821255978)
            b5 = ibis.literal(1.330274429)

            def Phi_ibis(x: ibis.Expr) -> ibis.Expr:
                ax = x.abs()
                t  = 1.0 / (1.0 + p_const * ax)
                poly = (((b5 * t + b4) * t + b3) * t + b2) * t + b1
                poly = poly * t
                phi  = INV_SQRT_2PI * (-0.5 * ax * ax).exp()
                Phi_abs = 1.0 - phi * poly
                return (x >= 0).ifelse(Phi_abs, 1.0 - Phi_abs)

            p_two = (ibis.literal(2.0) * (1.0 - Phi_ibis(z_abs)))

            I_tbl = now.cross_join(I0_tbl).select(I0=I0_tbl.I0, I1=I1, alpha=now.alpha, zfix=now.zfix, family=now.family)

            A1_OF = 2.0 - 2.0 * Phi_ibis(I_tbl.zfix / I_tbl.I1.sqrt().nullif(0))
            A0_OF = 2.0 - 2.0 * Phi_ibis(I_tbl.zfix / I_tbl.I0.sqrt().nullif(0))

            EM1   = ibis.literal(math.e - 1.0)
            A1_P  = I_tbl.alpha * ((1.0 + EM1 * I_tbl.I1).log())
            A0_P  = I_tbl.alpha * ((1.0 + EM1 * I_tbl.I0).log())

            fam_lc = I_tbl.family.lower()
            local_alpha = (
                ibis.case()
                .when((fam_lc == "pocock"), (A1_P - A0_P))
                .when((fam_lc == "of") | (fam_lc == "obrien_fleming") | (fam_lc == "o'brien_fleming") | (fam_lc == "obrien-fleming"), (A1_OF - A0_OF))
                .else_(A1_OF - A0_OF)
                .end()
            )
            local_alpha = (local_alpha + ibis.literal(1e-16))

            action_expr = (p_two <= local_alpha).ifelse("stop_efficacy", "continue")

            decision_base = now.cross_join(due).cross_join(I_tbl).select(
                look=due.look,
                planned_t=due.planned_t,
                info_time=I_tbl.I1,
                z=z_all,
                p_two=p_two,
                local_alpha=local_alpha,
                action=action_expr,
            )

            self._last_depths = {
                "I1":           ibis_depth(I1),
                "z_all":        ibis_depth(z_all),
                "p_two":        ibis_depth(p_two),
                "local_alpha":  ibis_depth(local_alpha),
                "action":       ibis_depth(action_expr),
            }

            s.write_from(decision_base, "stat", {"z": decision_base.z})
            s.write_from(
                decision_base, "decision",
                {
                    "look": decision_base.look,
                    "planned_t": decision_base.planned_t,
                    "info_time": decision_base.info_time,
                    "z": decision_base.z,
                    "boundary": ibis.literal(None).cast("double"),
                    "action": decision_base.action,
                },
            )


class EBaseRows(LedgerReaderBase):
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


class ENormalMixture:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        thetas_text = "[" + ",".join(str(float(t)) for t in thetas) + "]"
        thetas_json = ibis.literal(thetas_text)
        with LedgerSession(self.ledger) as s:
            s.write("e_design", {"alpha": float(alpha), "thetas": thetas_json}, raw_keys={"thetas"})

    def update(self, *, x: float) -> None:
        x_add = float(x)
        with LedgerSession(self.ledger) as s:
            s.write("e_obs", {"x": x_add})

            Vb = EBaseRows(self.ledger).typed_view()
            Theta = EThetas(self.ledger).grid()

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


def run_profile_demo(con: Any) -> str:
    import cProfile
    import pstats
    import io

    base = Ledger(con, table="ledger")
    prof_ledger = base.bind(experiment_id="exp_prof")
    ab = BinomialABTest(prof_ledger)

    pr = cProfile.Profile()
    pr.enable()

    ab.set_design(max_n=400, looks=[0.5, 1.0])
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
    ps.print_stats(r"(insert|raw_sql|flush|write_from|write|uuid|now)")
    out = s.getvalue()

    depth_lines = []
    depths = ab._last_depths if hasattr(ab, "_last_depths") else {}
    if depths:
        depth_lines.append("=== Ibis Expression Depth (10: SQL-only boundary) ===")
        for k in ("I1", "z_all", "p_two", "local_alpha", "action"):
            if k in depths:
                d, n = depths[k]
                depth_lines.append(f"{k:<11} depth={d:>2}, nodes={n}")
    depth_report = "\n".join(depth_lines)

    return out + ("\n" + depth_report if depth_report else "")


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
    out = run_profile_demo(con)
    print("=== cProfile (filtered) ===")
    print("Ordered by: cumulative time")
    print(out)
