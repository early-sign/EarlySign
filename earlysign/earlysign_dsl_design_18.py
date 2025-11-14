"""EarlySign DSL Design v18 - define-by-run ledger DSL demo.

Doctest (end-to-end workflows)
------------------------------
>>> import ibis
>>> con = ibis.duckdb.connect()
>>> ledger = Ledger(con, "ledger_dsl18_doctest", overwrite=True)
>>> ab = BinomialABTestV18(ledger, labels={"experiment_id": "exp_ab4"})
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
>>> eledger = Ledger(con, "ledger_dsl18_eproc", overwrite=True)
>>> ep = ENormalMixtureV18(eledger, labels={"run": "mixture"})
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

# NOTE: This version introduces a define-by-run session DSL so longer procedures can keep
#       their logic inline while still emitting ibis-native expressions. Both AB tests and
#       the e-process example from v9 run through the same primitives.

from __future__ import annotations

import cProfile
import json
import math
import pstats
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import ibis
from ibis.backends import BaseBackend
from ibis.expr.types import JSONValue, Table

__version__ = "18.0.0"


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

# def json_get_str(json_expr: JSONValue, key: str) -> StringValue:
#     return json_expr[key].unwrap_as("string")


# def json_get_i64(json_expr: JSONValue, key: str) -> IntegerValue:
#     return json_expr[key].unwrap_as("int64")


# def json_get_f64(json_expr: JSONValue, key: str) -> FloatingValue:
#     return json_expr[key].unwrap_as("float64")


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
    def table(self) -> Table:
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
        for job in self._jobs:
            job()

    def insert(
        self,
        *,
        kind: str,
        payload: Dict[str, Any],
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        merged_labels = dict(self.ledger.labels)
        if labels:
            merged_labels.update(labels)
        row = ibis.memtable([{"_anchor": 1}]).select(
            id=ibis.uuid(),
            ts=ibis.now(),
            pkg_version=ibis.literal(__version__),
            kind=ibis.literal(kind),
            labels=_json_literal(merged_labels),
            payload=_json_literal(payload),
        )

        def _exec() -> None:
            self.ledger.con.insert(self.ledger.table_name, row)

        self._jobs.append(_exec)


class LedgerReader:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def table(self) -> Table:
        tbl = self.ledger.table
        if not self.ledger.labels:
            return tbl
        lbl = tbl["labels"]
        predicates = [
            lbl[key].unwrap_as("string") == ibis.literal(str(value))
            for key, value in self.ledger.labels.items()
        ]
        return tbl.filter(predicates) if predicates else tbl

    def table_of_kind(self, kind: str) -> Table:
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


# ---------------------------------------------------------------------------
# Binomial AB test (define-by-run style)
# ---------------------------------------------------------------------------


class BinomialABTestV18:
    def __init__(self, ledger: Ledger, labels: Dict[str, Any]):
        self.ledger = ledger.bind(**labels)
        self.reader = self.ledger.reader()

    def set_design(
        self, *, max_n: int, looks: Sequence[float], alpha: float, spending: str
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
                    payload=dict(look=idx, planned_t=float(planned_t)),
                )

    def _planned_looks(self) -> List[Tuple[int, float]]:
        rows = self.reader.list_json(
            "design-look",
            {"look": ("look", "int64"), "planned_t": ("planned_t", "float64")},
            order_by="planned_t",
        )
        return [(int(r["look"]), float(r["planned_t"])) for r in rows]

    def update(self, payload: Dict[str, int]) -> None:
        defaults = {"nA": 0.0, "mA": 0.0, "nB": 0.0, "mB": 0.0}
        snapshot = self.reader.latest_json(
            "snapshot", {k: (k, "float64") for k in defaults}, default=defaults
        )
        design = self.reader.latest_json(
            "design",
            {
                "Nmax": ("planned_max_n", "float64"),
                "alpha": ("alpha", "float64"),
                "family": ("spending_family", "string"),
            },
            default={"Nmax": 0.0, "alpha": 0.05, "family": "obrien_fleming"},
        )
        if design["Nmax"] <= 0:
            raise RuntimeError("Design planned_max_n must be positive.")

        prev_nA = snapshot["nA"]
        prev_mA = snapshot["mA"]
        prev_nB = snapshot["nB"]
        prev_mB = snapshot["mB"]
        nA_now = prev_nA + int(payload["nA"])
        mA_now = prev_mA + int(payload["mA"])
        nB_now = prev_nB + int(payload["nB"])
        mB_now = prev_mB + int(payload["mB"])
        total_prev = prev_nA + prev_nB
        total_now = nA_now + nB_now
        Nmax = design["Nmax"]
        I0 = total_prev / Nmax if Nmax else 0.0
        I1 = total_now / Nmax if Nmax else 0.0
        looks = self._planned_looks()
        due = next(((idx, t) for idx, t in looks if I0 < t <= I1), None)
        z_value = _pooled_z(nA_now, mA_now, nB_now, mB_now)
        boundary = None
        action = None
        if due is not None:
            boundary = _boundary_from_spending(
                I0, I1, alpha=design["alpha"], family=design["family"]
            )
            action = "stop_efficacy" if abs(z_value) >= boundary else "continue"
        rows: List[Tuple[str, Dict[str, Any]]] = []
        rows.append(
            (
                "observation",
                dict(
                    nA=int(payload["nA"]),
                    mA=int(payload["mA"]),
                    nB=int(payload["nB"]),
                    mB=int(payload["mB"]),
                ),
            )
        )
        rows.append(
            (
                "snapshot",
                dict(
                    nA=float(nA_now),
                    mA=float(mA_now),
                    nB=float(nB_now),
                    mB=float(mB_now),
                ),
            )
        )
        rows.append(("info", dict(info_time=float(I1))))
        if due is not None:
            rows.append(("stat", dict(z=float(z_value))))
            rows.append(
                (
                    "decision",
                    dict(
                        look=int(due[0]),
                        planned_t=float(due[1]),
                        info_time=float(I1),
                        z=float(z_value),
                        boundary=float(boundary) if boundary is not None else 0.0,
                        action=str(action),
                    ),
                )
            )
        self.reader.emit_rows(rows)


# ---------------------------------------------------------------------------
# E-process (define-by-run version)
# ---------------------------------------------------------------------------


class ENormalMixtureV18:
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
        fields = {
            "sum_x": ("sum_x", "float64"),
            "n_obs": ("n_obs", "int64"),
            "e_value": ("e_value", "float64"),
            "alarm": ("alarm", "int64"),
        }
        state = self.reader.latest_json(
            "e_state",
            fields,
            default={"sum_x": 0.0, "n_obs": 0, "e_value": 1.0, "alarm": 0},
        )
        sum_x = state["sum_x"] + float(x)
        n_obs = int(state["n_obs"]) + 1
        thetas = self.design["thetas"]
        alpha = self.design["alpha"]
        mix_terms = [
            math.exp(theta * sum_x - 0.5 * theta * theta * n_obs) for theta in thetas
        ]
        e_now = sum(mix_terms) / len(mix_terms)
        alarm = int(e_now >= 1 / alpha)
        rows = [
            ("e_observation", dict(x=float(x))),
            (
                "e_state",
                dict(sum_x=sum_x, n_obs=n_obs, e_value=float(e_now), alarm=alarm),
            ),
        ]
        self.reader.emit_rows(rows)


# ---------------------------------------------------------------------------
# Profiling harness
# ---------------------------------------------------------------------------


def _ab_workload() -> BinomialABTestV18:
    con = ibis.duckdb.connect()
    ledger = Ledger(con, "ledger_v18_profile", overwrite=True)
    ab = BinomialABTestV18(ledger, labels={"experiment_id": "demo"})
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
