"""EarlySign DSL Design v21 - lazy-read define-by-run DSL with explicit DuckDB transactions.

Doctest (end-to-end workflows)
------------------------------
>>> import ibis
>>> con = ibis.duckdb.connect()
>>> ledger = Ledger(con, "ledger_dsl21_doctest", overwrite=True)
>>> ab = BinomialABTestV21(ledger, labels={"experiment_id": "exp_ab4"})
>>> ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="obrien_fleming")
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> ab.update({"nA": 50, "mA": 5, "nB": 50, "mB": 6})
>>> ab.update({"nA": 150, "mA": 10, "nB": 150, "mB": 20})
>>> ab.update({"nA": 100, "mA": 5, "nB": 100, "mB": 35})
>>> tbl = ledger.table
>>> decisions = (
...     tbl.filter(
...         (tbl.kind == "decision")
...         & (tbl.labels["experiment_id"].str == "exp_ab4")
...         & (tbl.payload["look"].unwrap_as("int64") > 0)
...     )
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
>>> eledger = Ledger(con, "ledger_dsl21_eproc", overwrite=True)
>>> ep = ENormalMixtureV21(eledger, labels={"run": "mixture"})
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

# NOTE: v21 builds on v20 by wrapping every update in a DuckDB transaction so the
#       staged reads/inserts share the same connection scope without re-opening
#       implicit transactions per statement.
#       ibis-native. The update logic mirrors the light-weight Python arithmetic from
#       v18 while emitting expressions that read the ledger lazily so we can benchmark
#       the impact without the massive recursive plans seen in v19.

from __future__ import annotations

import cProfile
import json
import math
import pstats
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ibis
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

__version__ = "21.0.0"


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

    @contextmanager
    def transaction(self):
        """Yield a DuckDB transaction scope using raw SQL."""
        try:
            self.con.raw_sql("BEGIN")
            yield
        except Exception:
            self.con.raw_sql("ROLLBACK")
            raise
        else:
            self.con.raw_sql("COMMIT")


class LedgerSession:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def __enter__(self) -> "LedgerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # inserts execute immediately; nothing to flush
        return None

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
        merged_labels = dict(self.ledger.labels)
        if labels:
            merged_labels.update(labels)
        if payload_expr is None:
            payload_expr = ibis.struct({k: ibis.literal(v) for k, v in payload.items()})
        if source is None:
            source = ibis.memtable([{"_anchor": 1}])
        row_select = source.select(
            id=ibis.uuid(),
            ts=ibis.now(),
            pkg_version=ibis.literal(__version__),
            kind=ibis.literal(kind),
            labels=_json_literal(merged_labels),
            payload=payload_expr.cast("json"),
        )

        # NOTE: row_select often references the same ledger table (e.g., latest snapshot).
        # DuckDB/SQLGlot currently recurse when compiling INSERT ... SELECT that reads the
        # target table, so we eagerly execute to a pandas row and insert literals instead.
        row_df = self.ledger.con.execute(row_select)
        rows = row_df.to_dict("records")
        if not rows:
            return
        literal_rows = []
        for row in rows:
            ts_value = row["ts"]
            if hasattr(ts_value, "to_pydatetime"):
                ts_value = ts_value.to_pydatetime()
            literal_rows.append(
                dict(
                    id=str(row["id"]),
                    ts=ts_value,
                    pkg_version=row["pkg_version"],
                    kind=row["kind"],
                    labels=row["labels"],
                    payload=row["payload"],
                )
            )
        self.ledger.con.insert(self.ledger.table_name, literal_rows)


class Stage:
    """Helper that materializes intermediate rows in the ledger."""

    def __init__(
        self,
        ledger: Ledger,
        *,
        kind: str,
        mapping: Dict[str, str],
        default_payload: Optional[Dict[str, Any]] = None,
    ):
        self.ledger = ledger
        self.kind = kind
        self.mapping = mapping
        self.default_payload = default_payload or {}
        self._seeded = False
        self._reader = self.ledger.reader()
        self._ensure_seed()

    def _ensure_seed(self) -> None:
        if self._seeded:
            return
        tbl = self._reader.table_of_kind(self.kind)
        count = self.ledger.con.execute(tbl.count())
        if hasattr(count, "iat"):
            count_value = count.iat[0, 0]
        else:
            count_value = int(count)
        if count_value == 0:
            with self.ledger.transaction():
                with self.ledger.session() as sess:
                    sess.insert(kind=self.kind, payload=self.default_payload)
        self._seeded = True

    def latest_expr(self) -> IbisTable:
        self._ensure_seed()
        tbl = self._reader.table_of_kind(self.kind)
        return tbl.order_by(tbl.ts.desc()).limit(1)

    def latest_values(self) -> IbisTable:
        tbl = self.latest_expr()
        payload = tbl["payload"]
        return tbl.select(
            **{
                alias: payload[alias].unwrap_as(dtype)
                for alias, dtype in self.mapping.items()
            }
        )

    def emit_from_expr(self, sess: LedgerSession, expr: IbisTable) -> None:
        payload_struct = ibis.struct({alias: expr[alias] for alias in self.mapping})
        sess.insert(kind=self.kind, payload_expr=payload_struct, source=expr)

    def emit_from_dict(self, sess: LedgerSession, payload: Dict[str, Any]) -> None:
        sess.insert(kind=self.kind, payload=payload)

    def latest_payload(self) -> Dict[str, Any]:
        df = self.ledger.con.execute(self.latest_values())
        if df.empty:
            return dict(self.default_payload)
        return df.to_dict("records")[0]


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

    def latest_struct_expr(
        self,
        kind: str,
        mapping: Dict[str, Tuple[str, str]],
        default: Optional[Dict[str, Any]] = None,
    ) -> IbisTable:
        tbl = self.table_of_kind(kind)
        payload = tbl["payload"]
        selects = {"ts_latest": tbl.ts}
        for alias, (field, dtype) in mapping.items():
            selects[alias] = payload[field].unwrap_as(dtype)
        typed = tbl.select(**selects)
        if default is None:
            default_tbl = None
        else:
            default_struct = dict(ts_latest=datetime(1970, 1, 1), **default)
            schema = ibis.schema(
                {"ts_latest": "timestamp"}
                | {alias: dtype for alias, (_, dtype) in mapping.items()}
            )
            default_tbl = ibis.memtable([default_struct], schema=schema)
        unioned = typed if default_tbl is None else typed.union(default_tbl)
        return unioned.order_by(ibis.desc("ts_latest")).limit(1)

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
            if order_by in mapping:
                field, dtype = mapping[order_by]
                order_expr = payload[field].unwrap_as(dtype)
            else:
                order_expr = tbl[order_by]
            tbl = tbl.order_by(order_expr)
        selects = [
            payload[field].unwrap_as(dtype).name(alias)
            for alias, (field, dtype) in mapping.items()
        ]
        df = self.ledger.con.execute(tbl.select(*selects))
        return df.to_dict("records")

    def emit_rows(self, rows: Iterable[Tuple[str, Dict[str, Any]]]) -> None:
        with self.ledger.session() as sess:
            for kind, payload in rows:
                sess.insert(kind=kind, payload=payload)


# ---------------------------------------------------------------------------
# Math helpers (same as earlier designs)
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


def _boundary_from_spending(
    I0: float, I1: float, *, alpha: float, family: str
) -> float:
    A1 = _alpha_spent(I1, alpha, family)
    A0 = _alpha_spent(I0, alpha, family) if I0 > 0 else 0.0
    local = max(A1 - A0, 1e-16)
    return _phi_inv(1.0 - local / 2.0)


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


def _boundary_from_spending_expr(
    I0: FloatingValue, I1: FloatingValue, *, alpha: FloatingValue, family: StringValue
) -> FloatingValue:
    eps = ibis.literal(1e-16)
    A1 = _alpha_spent_expr(I1, alpha, family)
    A0 = ibis.ifelse(I0 > 0, _alpha_spent_expr(I0, alpha, family), ibis.literal(0.0))
    local = ibis.greatest(A1 - A0, eps)
    return _phi_inv_expr(ibis.literal(1.0) - local / 2.0)


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


# ---------------------------------------------------------------------------
# Binomial AB test (define-by-run style)
# ---------------------------------------------------------------------------


class BinomialABTestV21:
    def __init__(self, ledger: Ledger, labels: Dict[str, Any]):
        self.ledger = ledger.bind(**labels)
        self.reader = self.ledger.reader()
        self._has_design = False
        self._looks_cache: List[Tuple[int, float]] = []
        self._design_default = dict(
            planned_max_n=0.0,
            alpha=0.05,
            spending_family="obrien_fleming",
        )
        self.snapshot_stage = Stage(
            self.ledger,
            kind="snapshot",
            mapping={
                "nA": "float64",
                "mA": "float64",
                "nB": "float64",
                "mB": "float64",
            },
            default_payload={"nA": 0.0, "mA": 0.0, "nB": 0.0, "mB": 0.0},
        )
        self.info_stage = Stage(
            self.ledger,
            kind="info",
            mapping={"info_time": "float64"},
            default_payload={"info_time": 0.0},
        )
        self.stat_stage = Stage(
            self.ledger,
            kind="stat",
            mapping={
                "info_time_prev": "float64",
                "info_time": "float64",
                "z": "float64",
            },
            default_payload={"info_time_prev": 0.0, "info_time": 0.0, "z": 0.0},
        )
        self.decision_stage = Stage(
            self.ledger,
            kind="decision",
            mapping={
                "look": "int64",
                "planned_t": "float64",
                "info_time": "float64",
                "z": "float64",
                "boundary": "float64",
                "action": "string",
            },
            default_payload={
                "look": 0,
                "planned_t": 0.0,
                "info_time": 0.0,
                "z": 0.0,
                "boundary": 0.0,
                "action": "continue",
            },
        )

    def _latest_snapshot_expr(self) -> IbisTable:
        return self.snapshot_stage.latest_values()

    def _latest_design_expr(self) -> IbisTable:
        tbl = self.reader.table_of_kind("design")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_design=tbl.ts,
            planned_max_n=json_get_f64(payload, "planned_max_n").cast("float64"),
            alpha=json_get_f64(payload, "alpha").cast("float64"),
            family=json_get_str(payload, "spending_family"),
        )
        default_tbl = ibis.memtable(
            [
                dict(
                    ts_design=datetime(1970, 1, 1),
                    planned_max_n=self._design_default["planned_max_n"],
                    alpha=self._design_default["alpha"],
                    family=self._design_default["spending_family"],
                )
            ],
            schema=ibis.schema(
                {
                    "ts_design": "timestamp(6)",
                    "planned_max_n": "float64",
                    "alpha": "float64",
                    "family": "string",
                }
            ),
        )
        return typed.union(default_tbl).order_by(ibis.desc("ts_design")).limit(1)

    def _delta_expr(self, payload: Dict[str, int]) -> IbisTable:
        return ibis.memtable(
            [
                dict(
                    nA=float(payload["nA"]),
                    mA=float(payload["mA"]),
                    nB=float(payload["nB"]),
                    mB=float(payload["mB"]),
                )
            ],
            schema=ibis.schema(
                {"nA": "float64", "mA": "float64", "nB": "float64", "mB": "float64"}
            ),
        )

    def _latest_design_payload(self) -> Dict[str, Any]:
        expr = self._latest_design_expr()
        df = self.ledger.con.execute(expr)
        if df.empty:
            return dict(self._design_default)
        row = df.to_dict("records")[0]
        return dict(
            planned_max_n=row["planned_max_n"], alpha=row["alpha"], family=row["family"]
        )

    def _planned_looks(self) -> List[Tuple[int, float]]:
        if self._looks_cache:
            return self._looks_cache
        tbl = self.reader.table_of_kind("design-look")
        payload = tbl["payload"]
        df = self.ledger.con.execute(
            tbl.select(
                look=json_get_i64(payload, "look").cast("int64"),
                planned_t=json_get_f64(payload, "planned_t").cast("float64"),
            ).order_by("planned_t")
        )
        looks = [
            (int(row["look"]), float(row["planned_t"])) for row in df.to_dict("records")
        ]
        self._looks_cache = looks
        return looks

    def set_design(
        self, *, max_n: int, looks: Sequence[float], alpha: float, spending: str
    ) -> None:
        looks = [float(x) for x in looks]
        with self.ledger.transaction():
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
                        payload=dict(look=idx, planned_t=float(planned_t)),
                    )
        self._has_design = True
        self._looks_cache = [
            (idx, planned_t) for idx, planned_t in enumerate(looks, start=1)
        ]

    def update(self, payload: Dict[str, int]) -> None:
        if not self._has_design:
            raise RuntimeError("Design must be configured before calling update().")
        with self.ledger.transaction():
            snapshot_prev = self.snapshot_stage.latest_payload()
            design = self._latest_design_payload()
            looks = self._planned_looks()

            nA_add = float(payload["nA"])
            mA_add = float(payload["mA"])
            nB_add = float(payload["nB"])
            mB_add = float(payload["mB"])

            snapshot_next = dict(
                nA=snapshot_prev["nA"] + nA_add,
                mA=snapshot_prev["mA"] + mA_add,
                nB=snapshot_prev["nB"] + nB_add,
                mB=snapshot_prev["mB"] + mB_add,
            )
            Nmax = design["planned_max_n"]
            info_prev = self.info_stage.latest_payload().get("info_time", 0.0)

            z_value = _pooled_z(
                snapshot_next["nA"],
                snapshot_next["mA"],
                snapshot_next["nB"],
                snapshot_next["mB"],
            )
            boundary = None
            action = None
            due = None

            observation_payload = dict(
                nA=int(payload["nA"]),
                mA=int(payload["mA"]),
                nB=int(payload["nB"]),
                mB=int(payload["mB"]),
            )

            with self.ledger.session() as sess:
                sess.insert(kind="observation", payload=observation_payload)
                self.snapshot_stage.emit_from_dict(sess, snapshot_next)
                snapshot_curr = self.snapshot_stage.latest_payload()
                info_payload = {
                    "info_time": (
                        (snapshot_curr["nA"] + snapshot_curr["nB"]) / Nmax
                        if Nmax
                        else 0.0
                    )
                }
                self.info_stage.emit_from_dict(sess, info_payload)
                info_curr = self.info_stage.latest_payload()
                stat_payload = {
                    "info_time_prev": info_prev,
                    "info_time": info_curr.get("info_time", 0.0),
                    "z": z_value,
                }
                self.stat_stage.emit_from_dict(sess, stat_payload)
                stat_curr = self.stat_stage.latest_payload()
                due = next(
                    (
                        (idx, t)
                        for idx, t in looks
                        if info_prev < t <= stat_curr["info_time"]
                    ),
                    None,
                )
                if due is not None:
                    boundary = _boundary_from_spending(
                        stat_curr["info_time_prev"],
                        stat_curr["info_time"],
                        alpha=design["alpha"],
                        family=design["family"],
                    )
                    action = (
                        "stop_efficacy"
                        if abs(stat_curr["z"]) >= boundary
                        else "continue"
                    )
                if due is not None and boundary is not None and action is not None:
                    look_idx, planned_t = due
                    self.decision_stage.emit_from_dict(
                        sess,
                        dict(
                            look=int(look_idx),
                            planned_t=float(planned_t),
                            info_time=stat_curr["info_time"],
                            z=float(stat_curr["z"]),
                            boundary=float(boundary),
                            action=action,
                        ),
                    )


# ---------------------------------------------------------------------------
# E-process (define-by-run version)
# ---------------------------------------------------------------------------


class ENormalMixtureV21:
    def __init__(self, ledger: Ledger, labels: Dict[str, Any]):
        self.ledger = ledger.bind(**labels)
        self.reader = self.ledger.reader()
        self.design: Dict[str, Any] = {"alpha": 0.05, "thetas": [0.5]}

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        self.design = {"alpha": float(alpha), "thetas": list(map(float, thetas))}
        with self.ledger.transaction():
            with self.ledger.session() as sess:
                sess.insert(
                    kind="e_design",
                    payload=dict(alpha=float(alpha), thetas=list(map(float, thetas))),
                )

    def _latest_state_expr(self) -> IbisTable:
        tbl = self.reader.table_of_kind("e_state")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_state=tbl.ts,
            sum_x=json_get_f64(payload, "sum_x").cast("float64"),
            n_obs=json_get_i64(payload, "n_obs").cast("int64"),
        )
        default_tbl = ibis.memtable(
            [
                dict(
                    ts_state=datetime(1970, 1, 1),
                    sum_x=0.0,
                    n_obs=0,
                )
            ],
            schema=ibis.schema(
                {"ts_state": "timestamp(6)", "sum_x": "float64", "n_obs": "int64"}
            ),
        )
        return typed.union(default_tbl).order_by(ibis.desc("ts_state")).limit(1)

    def update(self, *, x: float) -> None:
        obs_delta = ibis.memtable(
            [{"x": float(x)}], schema=ibis.schema({"x": "float64"})
        )
        state_prev = self._latest_state_expr()
        theta_tbl = ibis.memtable(
            [{"theta": float(theta)} for theta in self.design["thetas"]],
            schema=ibis.schema({"theta": "float64"}),
        )
        alpha_literal = ibis.literal(self.design["alpha"], type="float64")
        theta_count = ibis.literal(float(len(self.design["thetas"])), type="float64")

        with self.ledger.transaction():
            with self.ledger.session() as sess:
                sess.insert(
                    kind="e_observation",
                    payload_expr=ibis.struct(dict(x=obs_delta.x)),
                    source=obs_delta,
                )

                state_next = state_prev.cross_join(obs_delta).select(
                    sum_x=state_prev.sum_x + obs_delta.x,
                    n_obs=state_prev.n_obs + ibis.literal(1, type="int64"),
                )
                mix_terms = theta_tbl.cross_join(state_next).select(
                    term=(
                        (theta_tbl.theta * state_next.sum_x)
                        - 0.5 * theta_tbl.theta * theta_tbl.theta * state_next.n_obs
                    ).exp()
                )
                mix_value = mix_terms.aggregate(e_sum=mix_terms.term.sum())
                state_full = state_next.cross_join(mix_value).select(
                    sum_x=state_next.sum_x,
                    n_obs=state_next.n_obs,
                    e_value=mix_value.e_sum / theta_count,
                )
                state_payload = state_full.select(
                    sum_x=state_full.sum_x,
                    n_obs=state_full.n_obs,
                    e_value=state_full.e_value,
                    alarm=(state_full.e_value >= (1 / alpha_literal)).ifelse(1, 0),
                )
                sess.insert(
                    kind="e_state",
                    payload_expr=ibis.struct(
                        dict(
                            sum_x=state_payload.sum_x,
                            n_obs=state_payload.n_obs,
                            e_value=state_payload.e_value,
                            alarm=state_payload.alarm,
                        )
                    ),
                    source=state_payload,
                )


# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------


def _ab_workload() -> BinomialABTestV21:
    con = ibis.duckdb.connect()
    ledger = Ledger(con, "ledger_v20_profile", overwrite=True)
    ab = BinomialABTestV21(ledger, labels={"experiment_id": "demo"})
    ab.set_design(
        max_n=1000, looks=[0.25, 0.5, 0.75, 1.0], alpha=0.05, spending="obrien_fleming"
    )
    for batch in [
        {"nA": 100, "mA": 10, "nB": 100, "mB": 12},
        {"nA": 50, "mA": 5, "nB": 50, "mB": 6},
        {"nA": 150, "mA": 10, "nB": 150, "mB": 20},
        {"nA": 100, "mA": 5, "nB": 100, "mB": 35},
    ]:
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
