"""EarlySign DSL Design v19 - staged ibis-native ledger DSL demo.

Doctest (end-to-end workflows)
------------------------------
>>> import ibis
>>> con = ibis.duckdb.connect()
>>> ledger = Ledger(con, "ledger_dsl19_doctest", overwrite=True)
>>> ab = BinomialABTestV19(ledger, labels={"experiment_id": "exp_ab4"})
>>> ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="obrien_fleming")
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> ab.update({"nA": 50, "mA": 5, "nB": 50, "mB": 6})
>>> ab.update({"nA": 150, "mA": 10, "nB": 150, "mB": 20})
>>> ab.update({"nA": 100, "mA": 5, "nB": 100, "mB": 35})
>>> tbl = ledger.table
>>> decisions = (
...     tbl.filter((tbl.kind == "decision") & (tbl.labels["experiment_id"].str == "exp_ab4"))
...        .order_by(tbl.ts)
... )
>>> result = con.execute(
...     decisions.select(
...         decisions.payload["look"].unwrap_as("int64").name("look"),
...         decisions.payload["planned_t"].unwrap_as("float64").name("planned_t"),
...         decisions.payload["action"].str.name("action"),
...     )
... )
>>> rows = result.to_dict("records")
>>> [row["look"] for row in rows]
[1, 2, 3]
>>> [round(row["planned_t"], 2) for row in rows]
[0.25, 0.5, 0.75]
>>> rows[-1]["action"]
'stop_efficacy'

>>> # E-process example (copied from v9 to ensure parity)
>>> eledger = Ledger(con, "ledger_dsl19_eproc", overwrite=True)
>>> ep = ENormalMixtureV19(eledger, labels={"run": "mixture"})
>>> ep.set_design(alpha=0.05, thetas=[0.25, 0.5, 0.75])
>>> for _ in range(12):
...     ep.update(x=0.8)
>>> ereader = eledger.bind(run="mixture").reader()
>>> latest = ereader.latest_json("e_state", {"e_value": ("e_value", "float64"), "alarm": ("alarm", "int64")})
>>> round(latest["e_value"], 2), bool(latest["alarm"])  # doctest: +NORMALIZE_WHITESPACE
(26.84, True)
>>> e_rows = ereader.table_of_kind("e_state").count().execute()
>>> e_rows > 0
True
"""

from __future__ import annotations

import cProfile
import json
import math
import pstats
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import ibis
import sqlglot as sg
from ibis.backends import BaseBackend
from ibis.expr.types import (
    FloatingValue,
    IntegerValue,
    JSONValue,
    StringValue,
)
from ibis.expr.types import (
    Table as IbisTable,
)
from sqlglot import expressions as sge

__version__ = "19.0.0"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def json_get_str(json_expr: JSONValue, key: str) -> StringValue:
    return json_expr[key].unwrap_as("string")


def json_get_i64(json_expr: JSONValue, key: str) -> IntegerValue:
    return json_expr[key].unwrap_as("int64")


def json_get_f64(json_expr: JSONValue, key: str) -> FloatingValue:
    return json_expr[key].unwrap_as("float64")


def _json_literal(data: Dict[str, Any]) -> JSONValue:
    return ibis.literal(json.dumps(data, sort_keys=True), type="string").cast("json")


# ---------------------------------------------------------------------------
# Ledger + scoped DSL
# ---------------------------------------------------------------------------


class Ledger:
    def __init__(
        self,
        con: BaseBackend,
        table_name: str,
        *,
        labels: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
    ):
        self.con = con
        self.table_name = table_name
        self.labels = dict(labels or {})
        schema = ibis.schema(
            dict(
                id="uuid",
                ts="timestamp",
                pkg_version="string",
                kind="string",
                labels="json",
                payload="json",
            )
        )
        if overwrite:
            try:
                self.con.drop_table(self.table_name)
            except Exception:
                pass
        if self.table_name not in self.con.list_tables():
            self.con.create_table(self.table_name, schema=schema)

    @property
    def table(self) -> IbisTable:
        return self.con.table(self.table_name)

    def bind(self, **labels: Any) -> "Ledger":
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table_name, labels=merged, overwrite=False)

    def session(self) -> "LedgerSession":
        return LedgerSession(self)

    def reader(self) -> "LedgerReader":
        return LedgerReader(self)


class LedgerSession:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self._jobs: List[Callable[[], None]] = []

    def __enter__(self) -> "LedgerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return
        self.flush()

    def flush(self) -> None:
        jobs = self._jobs
        self._jobs = []
        for job in jobs:
            job()

    def insert(
        self,
        *,
        kind: str,
        payload: Optional[Dict[str, Any]] = None,
        payload_expr: Optional[ibis.Expr] = None,
        source: Optional[IbisTable] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        if (payload is None) == (payload_expr is None):
            raise ValueError("Provide exactly one of payload or payload_expr")
        if payload_expr is None:
            payload_expr = ibis.struct({k: ibis.literal(v) for k, v in payload.items()})
        if source is None:
            source = ibis.memtable([{"_anchor": 1}])
        merged_labels = dict(self.ledger.labels)
        if labels:
            merged_labels.update(labels)
        row_select = source.select(
            id=ibis.uuid(),
            ts=ibis.now(),
            pkg_version=ibis.literal(__version__),
            kind=ibis.literal(kind),
            labels=_json_literal(merged_labels),
            payload=payload_expr.cast("json"),
        )

        def _exec() -> None:
            backend = self.ledger.con
            backend._run_pre_execute_hooks(row_select)
            compiler = backend.compiler
            target_schema = backend.get_schema(self.ledger.table_name)
            target_cols = list(target_schema.keys())
            source_cols = list(row_select.schema().keys())
            if set(source_cols) <= set(target_cols):
                columns = source_cols
            else:
                columns = target_cols
            insert_expr = sge.Insert(
                this=sg.table(self.ledger.table_name, quoted=compiler.quoted),
                expression=compiler.to_sqlglot(row_select),
                columns=[
                    sg.to_identifier(col, quoted=compiler.quoted) for col in columns
                ],
            )
            backend.raw_sql(insert_expr.sql(dialect=compiler.dialect))

        self._jobs.append(_exec)


class LedgerReader:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def table(self) -> IbisTable:
        tbl = self.ledger.table
        if not self.ledger.labels:
            return tbl
        lbl = tbl["labels"]
        predicates = [
            lbl[key].unwrap_as("string") == ibis.literal(str(value))
            for key, value in self.ledger.labels.items()
        ]
        return tbl.filter(predicates) if predicates else tbl

    def table_of_kind(self, kind: str) -> IbisTable:
        tbl = self.table()
        return tbl.filter(tbl["kind"] == kind)

    def latest_json(
        self,
        kind: str,
        mapping: Dict[str, Tuple[str, str]],
        default: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tbl = self.table_of_kind(kind)
        tbl = tbl.order_by(tbl.ts.desc()).limit(1)
        payload = tbl["payload"]
        selects = [
            payload[field].unwrap_as(dtype).name(alias)
            for alias, (field, dtype) in mapping.items()
        ]
        if not selects:
            return dict(default) if default else {}
        df = self.ledger.con.execute(tbl.select(*selects))
        if df.empty:
            return dict(default) if default else {}
        return df.to_dict("records")[0]

    def list_json(
        self,
        kind: str,
        mapping: Dict[str, Tuple[str, str]],
        order_by: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        tbl = self.table_of_kind(kind)
        payload = tbl["payload"]
        if order_by is not None:
            order_expr = (
                payload[mapping[order_by][0]].unwrap_as(mapping[order_by][1])
                if order_by in mapping
                else tbl[order_by]
            )
            tbl = tbl.order_by(order_expr)
        selects = [
            payload[field].unwrap_as(dtype).name(alias)
            for alias, (field, dtype) in mapping.items()
        ]
        df = self.ledger.con.execute(tbl.select(*selects))
        return df.to_dict("records")


# ---------------------------------------------------------------------------
# Math helpers (scalar and expression variants)
# ---------------------------------------------------------------------------


def _cdf_normal(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / ((((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    )


def _alpha_spent(t: float, alpha: float, family: str) -> float:
    t = min(max(t, 1e-12), 1.0)
    family_key = family.lower()
    if family_key in {"of", "obrien_fleming", "o'brien_fleming", "obrien-fleming"}:
        z = _phi_inv(1.0 - alpha / 2.0)
        return 2.0 - 2.0 * _cdf_normal(z / math.sqrt(t))
    if family_key == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"Unknown spending family: {family}")


def _boundary_from_spending_scalar(
    I0: float, I1: float, *, alpha: float, family: str
) -> float:
    A1 = _alpha_spent(I1, alpha, family)
    A0 = _alpha_spent(I0, alpha, family) if I0 > 0 else 0.0
    local = max(A1 - A0, 1e-16)
    return _phi_inv(1.0 - local / 2.0)


def _erf_expr(x: FloatingValue) -> FloatingValue:
    sign = ibis.ifelse(x >= 0, ibis.literal(1.0), ibis.literal(-1.0))
    abs_x = x.abs()
    p = ibis.literal(0.3275911)
    a1 = ibis.literal(0.254829592)
    a2 = ibis.literal(-0.284496736)
    a3 = ibis.literal(1.421413741)
    a4 = ibis.literal(-1.453152027)
    a5 = ibis.literal(1.061405429)
    t = 1 / (1 + p * abs_x)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    exp_term = (-abs_x * abs_x).exp()
    return sign * (1 - poly * exp_term)


def _cdf_normal_expr(x: FloatingValue) -> FloatingValue:
    return 0.5 * (1 + _erf_expr(x / ibis.literal(math.sqrt(2.0))))


def _phi_inv_expr(p_raw: FloatingValue) -> FloatingValue:
    eps = ibis.literal(1e-12)
    one = ibis.literal(1.0)
    p = ibis.greatest(ibis.least(p_raw, one - eps), eps)
    plow = ibis.literal(0.02425)
    phigh = one - plow

    def _poly(coeffs: List[float], var: FloatingValue) -> FloatingValue:
        expr = ibis.literal(coeffs[0])
        for coeff in coeffs[1:]:
            expr = expr * var + ibis.literal(coeff)
        return expr

    q_low = (-2 * p.log()).sqrt()
    low = _poly(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ],
        q_low,
    ) / (
        _poly(
            [
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e00,
                3.754408661907416e00,
            ],
            q_low,
        )
        * q_low
        + 1
    )

    q_high = (-2 * (1 - p).log()).sqrt()
    high = -_poly(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ],
        q_high,
    ) / (
        _poly(
            [
                7.784695709041462e-03,
                3.224671290700398e-01,
                2.445134137142996e00,
                3.754408661907416e00,
            ],
            q_high,
        )
        * q_high
        + 1
    )

    q = p - 0.5
    r = q * q
    mid = (
        _poly(
            [
                -3.969683028665376e01,
                2.209460984245205e02,
                -2.759285104469687e02,
                1.383577518672690e02,
                -3.066479806614716e01,
                2.506628277459239e00,
            ],
            r,
        )
        * q
        / (
            _poly(
                [
                    -5.447609879822406e01,
                    1.615858368580409e02,
                    -1.556989798598866e02,
                    6.680131188771972e01,
                    -1.328068155288572e01,
                ],
                r,
            )
            * r
            + 1
        )
    )

    return ibis.ifelse(p < plow, low, ibis.ifelse(p > phigh, high, mid))


def _boundary_from_spending_expr(
    I0: FloatingValue, I1: FloatingValue, *, alpha: FloatingValue, family: StringValue
) -> FloatingValue:
    eps = ibis.literal(1e-16)
    A1 = _alpha_spent_expr(I1, alpha, family)
    A0 = ibis.ifelse(I0 > 0, _alpha_spent_expr(I0, alpha, family), ibis.literal(0.0))
    local = ibis.greatest(A1 - A0, eps)
    return _phi_inv_expr(ibis.literal(1.0) - local / 2.0)


def _alpha_spent_expr(
    t: FloatingValue, alpha: FloatingValue, family: StringValue
) -> FloatingValue:
    clipped_t = ibis.greatest(ibis.least(t, ibis.literal(1.0)), ibis.literal(1e-12))
    family_lower = family.lower()
    alpha_half = alpha / 2.0
    z = _phi_inv_expr(ibis.literal(1.0) - alpha_half)
    obrien = 2 - 2 * _cdf_normal_expr(z / clipped_t.sqrt())
    pocock = alpha * (1 + ibis.literal(math.e - 1.0) * clipped_t).log()
    return ibis.ifelse(
        family_lower.isin(
            ["of", "obrien_fleming", "o'brien_fleming", "obrien-fleming"]
        ),
        obrien,
        ibis.ifelse(family_lower == "pocock", pocock, ibis.literal(0.0)),
    )


def _pooled_z_expr(
    nA: FloatingValue, mA: FloatingValue, nB: FloatingValue, mB: FloatingValue
) -> FloatingValue:
    eps = ibis.literal(1e-9)
    valid = (nA > 0) & (nB > 0)
    nA_safe = ibis.greatest(nA, eps)
    nB_safe = ibis.greatest(nB, eps)
    total_n = nA + nB
    total_safe = ibis.greatest(total_n, eps)
    pA = mA / nA_safe
    pB = mB / nB_safe
    pooled = (mA + mB) / total_safe
    denom = (pooled * (1 - pooled) * (1 / nA_safe + 1 / nB_safe) + eps).sqrt()
    return ibis.ifelse(valid & (denom > eps), (pB - pA) / denom, ibis.literal(0.0))


def _pooled_z(nA: float, mA: float, nB: float, mB: float) -> float:
    eps = 1e-9
    if nA <= 0 or nB <= 0:
        return 0.0
    pA = mA / max(nA, eps)
    pB = mB / max(nB, eps)
    total_n = nA + nB
    pooled = (mA + mB) / max(total_n, eps)
    denom = math.sqrt(
        pooled * (1.0 - pooled) * (1.0 / max(nA, eps) + 1.0 / max(nB, eps)) + eps
    )
    if denom <= eps:
        return 0.0
    return (pB - pA) / denom


# ---------------------------------------------------------------------------
# Binomial AB test (staged ibis-native workflow)
# ---------------------------------------------------------------------------


class BinomialABTestV19:
    def __init__(self, ledger: Ledger, labels: Dict[str, Any]):
        self.ledger = ledger.bind(**labels)
        self.reader = self.ledger.reader()
        self._has_design = False

    # --- snapshot helpers -------------------------------------------------
    def _scoped_kind(self, kind: str) -> IbisTable:
        tbl = self.ledger.table
        filtered = tbl.filter(tbl["kind"] == kind)
        if not self.ledger.labels:
            return filtered
        lbl = filtered["labels"]
        predicates = [
            lbl[key].unwrap_as("string") == ibis.literal(str(value))
            for key, value in self.ledger.labels.items()
        ]
        return filtered.filter(predicates) if predicates else filtered

    def _snapshot_rows(self) -> IbisTable:
        tbl = self._scoped_kind("snapshot")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_snapshot=tbl.ts,
            nA=json_get_f64(payload, "nA").cast("float64"),
            mA=json_get_f64(payload, "mA").cast("float64"),
            nB=json_get_f64(payload, "nB").cast("float64"),
            mB=json_get_f64(payload, "mB").cast("float64"),
        )
        default = ibis.memtable(
            [
                {
                    "ts_snapshot": datetime(1970, 1, 1),
                    "nA": 0.0,
                    "mA": 0.0,
                    "nB": 0.0,
                    "mB": 0.0,
                }
            ],
            schema=ibis.schema(
                {
                    "ts_snapshot": "timestamp(6)",
                    "nA": "float64",
                    "mA": "float64",
                    "nB": "float64",
                    "mB": "float64",
                }
            ),
        )
        return typed.union(default)

    def _snapshot_desc_rows(self) -> IbisTable:
        return self._snapshot_rows().order_by(ibis.desc("ts_snapshot"))

    def _snapshot_expr(self, offset: int = 0) -> IbisTable:
        rows = self._snapshot_desc_rows()
        return rows.limit(1, offset=offset)

    def _latest_design_expr(self) -> IbisTable:
        tbl = self._scoped_kind("design")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_design=tbl.ts,
            planned_max_n=json_get_f64(payload, "planned_max_n").cast("float64"),
            alpha=json_get_f64(payload, "alpha").cast("float64"),
            family=json_get_str(payload, "spending_family"),
            has_design=ibis.literal(1, type="int64"),
        )
        default = ibis.memtable(
            [
                {
                    "ts_design": datetime(1970, 1, 1),
                    "planned_max_n": 0.0,
                    "alpha": 0.05,
                    "family": "obrien_fleming",
                    "has_design": 0,
                }
            ],
            schema=ibis.schema(
                {
                    "ts_design": "timestamp(6)",
                    "planned_max_n": "float64",
                    "alpha": "float64",
                    "family": "string",
                    "has_design": "int64",
                }
            ),
        )
        return typed.union(default).order_by(ibis.desc("ts_design")).limit(1)

    def _looks_table(self) -> IbisTable:
        tbl = self._scoped_kind("design-look")
        payload = tbl["payload"]
        return tbl.select(
            look=json_get_i64(payload, "look").cast("int64"),
            planned_t=json_get_f64(payload, "planned_t").cast("float64"),
        ).order_by("planned_t")

    def _info_expr(self) -> IbisTable:
        snapshot = self._snapshot_expr(offset=0)
        design = self._latest_design_expr()
        return snapshot.cross_join(design).select(
            info_time=(snapshot.nA + snapshot.nB) / design.planned_max_n
        )

    def _stat_expr(self) -> IbisTable:
        snapshots = self._snapshot_desc_rows()
        order_window = ibis.window(order_by=[snapshots.ts_snapshot.desc()])
        enriched = snapshots.mutate(
            nA_prev=snapshots.nA.lead().over(order_window).fill_null(0.0),
            mA_prev=snapshots.mA.lead().over(order_window).fill_null(0.0),
            nB_prev=snapshots.nB.lead().over(order_window).fill_null(0.0),
            mB_prev=snapshots.mB.lead().over(order_window).fill_null(0.0),
        )
        curr = enriched.limit(1)
        design = self._latest_design_expr()
        return curr.cross_join(design).select(
            info_time=(curr.nA + curr.nB) / design.planned_max_n,
            I0=(curr.nA_prev + curr.nB_prev) / design.planned_max_n,
            I1=(curr.nA + curr.nB) / design.planned_max_n,
            z_value=_pooled_z_expr(curr.nA, curr.mA, curr.nB, curr.mB),
            boundary=_boundary_from_spending_expr(
                (curr.nA_prev + curr.nB_prev) / design.planned_max_n,
                (curr.nA + curr.nB) / design.planned_max_n,
                alpha=design.alpha,
                family=design.family,
            ),
        )

    def _latest_stat_expr(self) -> IbisTable:
        tbl = self._scoped_kind("stat")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_stat=tbl.ts,
            info_time=json_get_f64(payload, "info_time").cast("float64"),
            I0=json_get_f64(payload, "I0").cast("float64"),
            I1=json_get_f64(payload, "I1").cast("float64"),
            z_value=json_get_f64(payload, "z").cast("float64"),
            boundary=json_get_f64(payload, "boundary").cast("float64"),
        )
        default = ibis.memtable(
            [
                {
                    "ts_stat": datetime(1970, 1, 1),
                    "info_time": 0.0,
                    "I0": 0.0,
                    "I1": 0.0,
                    "z_value": 0.0,
                    "boundary": 0.0,
                }
            ],
            schema=ibis.schema(
                {
                    "ts_stat": "timestamp(6)",
                    "info_time": "float64",
                    "I0": "float64",
                    "I1": "float64",
                    "z_value": "float64",
                    "boundary": "float64",
                }
            ),
        )
        return typed.union(default).order_by(ibis.desc("ts_stat")).limit(1)

    def _decision_expr(self) -> IbisTable:
        stat = self._latest_stat_expr()
        looks = self._looks_table()
        return (
            looks.cross_join(stat)
            .filter((looks.planned_t > stat.I0) & (looks.planned_t <= stat.I1))
            .order_by(looks.planned_t)
            .limit(1)
            .select(
                look=looks.look,
                planned_t=looks.planned_t,
                info_time=stat.info_time,
                z=stat.z_value,
                boundary=stat.boundary,
                action=ibis.ifelse(
                    stat.z_value.abs() >= stat.boundary,
                    ibis.literal("stop_efficacy"),
                    ibis.literal("continue"),
                ),
            )
        )

    # --- public API -------------------------------------------------------
    def set_design(
        self,
        *,
        max_n: int,
        looks: Sequence[float],
        alpha: float = 0.05,
        spending: str = "obrien_fleming",
    ) -> None:
        looks = [float(x) for x in looks]
        with self.ledger.session() as sess:
            sess.insert(
                kind="design",
                payload=dict(
                    planned_max_n=float(max_n),
                    alpha=float(alpha),
                    spending_family=str(spending),
                ),
            )
            for idx, planned_t in enumerate(looks, start=1):
                sess.insert(
                    kind="design-look",
                    payload=dict(look=int(idx), planned_t=float(planned_t)),
                )
        self._has_design = True

    def update(self, payload: Dict[str, int]) -> None:
        if not self._has_design:
            raise RuntimeError("Design must be configured before calling update().")
        delta = ibis.memtable(
            [
                {
                    "nA": float(payload["nA"]),
                    "mA": float(payload["mA"]),
                    "nB": float(payload["nB"]),
                    "mB": float(payload["mB"]),
                }
            ],
            schema=ibis.schema(
                {"nA": "float64", "mA": "float64", "nB": "float64", "mB": "float64"}
            ),
        )

        with self.ledger.session() as sess:
            obs_struct = ibis.struct(
                dict(nA=delta.nA, mA=delta.mA, nB=delta.nB, mB=delta.mB)
            )
            sess.insert(kind="observation", payload_expr=obs_struct, source=delta)
            sess.flush()

            prev_snapshot = self._snapshot_expr(offset=0)
            snapshot_expr = prev_snapshot.cross_join(delta).select(
                nA=prev_snapshot.nA + delta.nA,
                mA=prev_snapshot.mA + delta.mA,
                nB=prev_snapshot.nB + delta.nB,
                mB=prev_snapshot.mB + delta.mB,
            )
            snapshot_struct = ibis.struct(
                dict(
                    nA=snapshot_expr.nA,
                    mA=snapshot_expr.mA,
                    nB=snapshot_expr.nB,
                    mB=snapshot_expr.mB,
                )
            )
            sess.insert(
                kind="snapshot", payload_expr=snapshot_struct, source=snapshot_expr
            )
            sess.flush()

            info_expr = self._info_expr()
            info_struct = ibis.struct(dict(info_time=info_expr.info_time))
            sess.insert(kind="info", payload_expr=info_struct, source=info_expr)
            sess.flush()

            stat_expr = self._stat_expr()
            stat_struct = ibis.struct(
                dict(
                    info_time=stat_expr.info_time,
                    I0=stat_expr.I0,
                    I1=stat_expr.I1,
                    z=stat_expr.z_value,
                    boundary=stat_expr.boundary,
                )
            )
            sess.insert(kind="stat", payload_expr=stat_struct, source=stat_expr)
            sess.flush()

            decision_rows = self._decision_expr()
            decision_struct = ibis.struct(
                dict(
                    look=decision_rows.look,
                    planned_t=decision_rows.planned_t,
                    info_time=decision_rows.info_time,
                    z=decision_rows.z,
                    boundary=decision_rows.boundary,
                    action=decision_rows.action,
                )
            )
            sess.insert(
                kind="decision", payload_expr=decision_struct, source=decision_rows
            )
            sess.flush()


# ---------------------------------------------------------------------------
# E-process (staged ibis-native workflow)
# ---------------------------------------------------------------------------


class ENormalMixtureV19:
    def __init__(self, ledger: Ledger, labels: Dict[str, Any]):
        self.ledger = ledger.bind(**labels)
        self.reader = self.ledger.reader()
        self.design: Dict[str, Any] = {"alpha": 0.05, "thetas": [0.5]}

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        self.design = {"alpha": float(alpha), "thetas": list(map(float, thetas))}
        with self.ledger.session() as sess:
            sess.insert(
                kind="e_design",
                payload=dict(alpha=float(alpha), thetas=list(map(float, thetas))),
            )

    def update(self, *, x: float) -> None:
        obs_delta = ibis.memtable([{"x": float(x)}])
        with self.ledger.session() as sess:
            sess.insert(
                kind="e_observation",
                payload_expr=ibis.struct(dict(x=obs_delta.x)),
                source=obs_delta,
            )
            sess.flush()

            obs_tbl = self.reader.table_of_kind("e_observation")
            payload = obs_tbl["payload"]
            x_col = json_get_f64(payload, "x").cast("float64").name("x_val")
            agg = obs_tbl.select(x_col).aggregate(
                sum_x=x_col.sum().fill_null(0.0), n_obs=x_col.count()
            )
            theta_tbl = ibis.memtable(
                [{"theta": float(theta)} for theta in self.design["thetas"]]
            )
            parts = theta_tbl.cross_join(agg).select(
                term=(
                    (theta_tbl.theta * agg.sum_x)
                    - 0.5 * theta_tbl.theta * theta_tbl.theta * agg.n_obs
                ).exp()
            )
            mix = parts.aggregate(
                e_value=parts.term.sum() / float(len(self.design["thetas"]))
            )
            alpha_val = ibis.literal(self.design["alpha"], type="float64")
            state_expr = mix.select(
                e_value=mix.e_value,
                alarm=(mix.e_value >= (1 / alpha_val)).ifelse(1, 0),
            )
            sess.insert(
                kind="e_state",
                payload_expr=ibis.struct(
                    dict(e_value=state_expr.e_value, alarm=state_expr.alarm)
                ),
                source=state_expr,
            )
            sess.flush()


# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------


def _ab_workload() -> BinomialABTestV19:
    con = ibis.duckdb.connect()
    ledger = Ledger(con, "ledger_v19_profile", overwrite=True)
    ab = BinomialABTestV19(ledger, labels={"experiment_id": "demo"})
    ab.set_design(
        max_n=1000, looks=[0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="obrien_fleming"
    )
    batches = [
        {"nA": 100, "mA": 10, "nB": 100, "mB": 12},
        {"nA": 50, "mA": 5, "nB": 50, "mB": 6},
        {"nA": 150, "mA": 10, "nB": 150, "mB": 20},
        {"nA": 100, "mA": 5, "nB": 100, "mB": 35},
    ]
    for batch in batches:
        ab.update(batch)
    return ab


def _profile_run() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    ab = _ab_workload()
    profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
    print(ab.ledger.con.execute(ab.ledger.table))


if __name__ == "__main__":
    _profile_run()
