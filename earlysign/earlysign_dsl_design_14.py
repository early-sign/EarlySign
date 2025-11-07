"""EarlySign DSL Design v14 - ledger-first group sequential demo.

Doctest (A/B test with on-the-fly spending)
-------------------------------------------
>>> import ibis
>>> con = ibis.duckdb.connect()
>>> ledger = Ledger(con, "ledger_dsl14_doctest", overwrite=True)
>>> ab = BinomialABTest(ledger, labels={"experiment_id": "exp_ab4"})
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
"""

import cProfile
import json
import math
import pstats
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import ibis
from ibis.backends import BaseBackend
from ibis.expr.types import (
    BooleanValue,
    FloatingValue,
    IntegerValue,
    JSONValue,
    StringValue,
)
from ibis.expr.types import (
    Table as IbisTable,
)

__version__ = "14.0.0"


def json_get_str(json_expr: JSONValue, key: str) -> StringValue:
    """Return a JSON string field using ``unwrap_as`` to keep everything typed."""

    return json_expr[key].unwrap_as("string")


def json_get_i64(json_expr: JSONValue, key: str) -> IntegerValue:
    """Return a JSON integer field as ``int64``."""

    return json_expr[key].unwrap_as("int64")


def json_get_f64(json_expr: JSONValue, key: str) -> FloatingValue:
    """Return a JSON numeric field as ``float64``."""

    return json_expr[key].unwrap_as("float64")


def _json_literal(data: Dict[str, Any]) -> JSONValue:
    """Return a JSON value backed by a deterministic string literal."""

    return ibis.literal(json.dumps(data, sort_keys=True), type="string").cast("json")


def _cdf_normal(x: float) -> float:
    """Gaussian CDF via ``math.erf`` (two-sided inference helpers)."""

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv(p: float) -> float:
    """Inverse CDF Φ⁻¹ (Algorithm AS241)."""

    # Constants replicated from AS241
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
    if p <= 0:  # pragma: no cover - defensive guard
        return -math.inf
    if p >= 1:  # pragma: no cover - defensive guard
        return math.inf
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def _alpha_spent(t: float, alpha: float, family: str) -> float:
    """Lan–DeMets cumulative α(t) for the supported spending families."""

    t = min(max(t, 1e-12), 1.0)
    family_key = family.lower()
    if family_key in {"of", "obrien_fleming", "o'brien_fleming", "obrien-fleming"}:
        z = _phi_inv(1.0 - alpha / 2.0)
        return 2.0 - 2.0 * _cdf_normal(z / math.sqrt(t))
    if family_key == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"Unknown spending family: {family}")


def _boundary_from_spending(I0: float, I1: float, *, alpha: float, family: str) -> float:
    """Two-sided single-look z-threshold from the local α increment."""

    A1 = _alpha_spent(I1, alpha, family)
    A0 = _alpha_spent(I0, alpha, family) if I0 > 0 else 0.0
    local = max(A1 - A0, 1e-16)
    return _phi_inv(1.0 - local / 2.0)


def _pooled_z(nA: float, mA: float, nB: float, mB: float) -> float:
    """Classic pooled two-proportion z-statistic."""

    eps = 1e-9
    if nA <= 0 or nB <= 0:
        return 0.0
    pA = mA / max(nA, eps)
    pB = mB / max(nB, eps)
    total_n = nA + nB
    pooled = (mA + mB) / max(total_n, eps)
    denom = math.sqrt(pooled * (1.0 - pooled) * (1.0 / max(nA, eps) + 1.0 / max(nB, eps)) + eps)
    if denom <= eps:
        return 0.0
    return (pB - pA) / denom


class Ledger:
    """Ibis-backed ledger storing immutable JSON payloads."""

    def __init__(self, con: BaseBackend, table_name: str, overwrite: bool = True) -> None:
        self.con = con
        self.table_name = table_name
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

    def session(self) -> "LedgerSession":
        return LedgerSession(self)


class LedgerSession:
    """Queue ``INSERT`` jobs and execute them when the context exits cleanly."""

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

    def _enqueue_insert(self, select_expr: IbisTable) -> None:
        con = self.ledger.con
        table_name = self.ledger.table_name

        def _exec() -> None:
            con.insert(table_name, select_expr)

        self._jobs.append(_exec)

    def insert(self, *, kind: str, labels: Dict[str, Any], payload: Dict[str, Any]) -> None:
        labels_expr = _json_literal({k: str(v) for k, v in labels.items()})
        payload_json = _json_literal(payload)

        row_select = ibis.memtable([{"_anchor": 1}]).select(
            id=ibis.uuid(),
            ts=ibis.now(),
            pkg_version=ibis.literal(__version__),
            kind=ibis.literal(kind),
            labels=labels_expr,
            payload=payload_json,
        )
        self._enqueue_insert(row_select)


class BinomialABTest:
    """Group sequential two-proportion test driven by the ledger."""

    def __init__(self, ledger: Ledger, *, labels: Dict[str, Any]):
        self.ledger = ledger
        self.labels = dict(labels)

    def set_design(
        self,
        *,
        max_n: int,
        looks: Sequence[float],
        alpha: float = 0.05,
        spending: str = "obrien_fleming",
    ) -> None:
        looks = [float(x) for x in looks]
        looks_text = "[" + ",".join(str(x) for x in looks) + "]"
        with self.ledger.session() as sess:
            design_payload = dict(
                planned_max_n=float(max_n),
                alpha=float(alpha),
                spending_family=str(spending),
                looks=looks_text,
            )
            sess.insert(kind="design", labels=self.labels, payload=design_payload)
            for idx, planned_t in enumerate(looks, start=1):
                look_payload = dict(look=int(idx), planned_t=float(planned_t))
                sess.insert(kind="design-look", labels=self.labels, payload=look_payload)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_labels(self, table: IbisTable) -> IbisTable:
        if not self.labels:
            return table
        labels_col = table["labels"]
        predicates: List[BooleanValue] = []
        for key, value in self.labels.items():
            condition = labels_col[key].unwrap_as("string") == ibis.literal(str(value))
            predicates.append(condition)
        return table.filter(predicates) if predicates else table

    def _kind_exists(self, kind: str) -> bool:
        tbl = self.ledger.table
        kind_pred = tbl["kind"] == ibis.literal(kind)
        filtered = tbl.filter(kind_pred)
        filtered = self._apply_labels(filtered)
        count_expr = filtered.count()
        return bool(self.ledger.con.execute(count_expr))

    def _latest_row(self, kind: str) -> IbisTable:
        tbl = self.ledger.table
        kind_pred = tbl["kind"] == ibis.literal(kind)
        filtered = tbl.filter(kind_pred)
        filtered = self._apply_labels(filtered)
        return filtered.order_by(tbl.ts.desc()).limit(1)

    def _planned_looks(self) -> List[Tuple[int, float]]:
        tbl = self.ledger.table
        kind_pred = tbl["kind"] == ibis.literal("design-look")
        looks_tbl = tbl.filter(kind_pred)
        looks_tbl = self._apply_labels(looks_tbl)
        looks_tbl = looks_tbl.order_by(tbl.ts)
        look_expr = json_get_i64(looks_tbl.payload, "look").name("look")
        planned_expr = json_get_f64(looks_tbl.payload, "planned_t").name("planned_t")
        frame = looks_tbl.select(look_expr, planned_expr)
        raw = self.ledger.con.execute(frame)
        if raw.empty:
            return []
        return [(int(r["look"]), float(r["planned_t"])) for r in raw.to_dict("records")]

    def _state_values(self) -> Dict[str, Any]:
        if not self._kind_exists("design"):
            raise RuntimeError("Design must be configured before calling update().")

        snapshot_row = self._latest_row("snapshot")
        design_row = self._latest_row("design")

        snap_df = self.ledger.con.execute(
            snapshot_row.select(
                json_get_f64(snapshot_row.payload, "nA").name("nA"),
                json_get_f64(snapshot_row.payload, "mA").name("mA"),
                json_get_f64(snapshot_row.payload, "nB").name("nB"),
                json_get_f64(snapshot_row.payload, "mB").name("mB"),
            )
        )
        if snap_df.empty:
            prev_nA = prev_mA = prev_nB = prev_mB = 0.0
        else:
            snap_row = snap_df.to_dict("records")[0]
            prev_nA = float(snap_row.get("nA", 0.0) or 0.0)
            prev_mA = float(snap_row.get("mA", 0.0) or 0.0)
            prev_nB = float(snap_row.get("nB", 0.0) or 0.0)
            prev_mB = float(snap_row.get("mB", 0.0) or 0.0)

        design_df = self.ledger.con.execute(
            design_row.select(
                json_get_f64(design_row.payload, "planned_max_n").name("Nmax"),
                json_get_f64(design_row.payload, "alpha").name("alpha"),
                json_get_str(design_row.payload, "spending_family").name("family"),
            )
        )
        if design_df.empty:
            raise RuntimeError("Design row not found for current labels.")
        design_vals = design_df.to_dict("records")[0]
        Nmax = float(design_vals.get("Nmax", 0.0) or 0.0)
        if Nmax <= 0:
            raise RuntimeError("Design planned_max_n must be positive.")

        return {
            "prev_nA": prev_nA,
            "prev_mA": prev_mA,
            "prev_nB": prev_nB,
            "prev_mB": prev_mB,
            "Nmax": Nmax,
            "alpha": float(design_vals.get("alpha", 0.05) or 0.05),
            "family": str(design_vals.get("family", "obrien_fleming") or "obrien_fleming"),
        }

    def _select_due(self, I0: float, I1: float) -> Optional[Tuple[int, float]]:
        candidates = [(look, t) for look, t in self._planned_looks() if I0 < t <= I1]
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1])
        return candidates[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, payload: Dict[str, int]) -> None:
        nA_add = int(payload["nA"])
        mA_add = int(payload["mA"])
        nB_add = int(payload["nB"])
        mB_add = int(payload["mB"])

        state = self._state_values()
        prev_nA = float(state["prev_nA"])
        prev_mA = float(state["prev_mA"])
        prev_nB = float(state["prev_nB"])
        prev_mB = float(state["prev_mB"])
        Nmax = float(state["Nmax"])
        alpha = float(state["alpha"])
        family = str(state["family"])

        nA_now = prev_nA + nA_add
        mA_now = prev_mA + mA_add
        nB_now = prev_nB + nB_add
        mB_now = prev_mB + mB_add

        total_prev = prev_nA + prev_nB
        total_now = nA_now + nB_now
        I0 = total_prev / Nmax if Nmax else 0.0
        I1 = total_now / Nmax if Nmax else 0.0

        due = self._select_due(I0, I1)
        z_value = _pooled_z(nA_now, mA_now, nB_now, mB_now)
        if due is not None:
            boundary = _boundary_from_spending(I0, I1, alpha=alpha, family=family)
            action = "stop_efficacy" if abs(z_value) >= boundary else "continue"
        else:
            boundary = None
            action = None

        with self.ledger.session() as sess:
            obs_payload = dict(nA=nA_add, mA=mA_add, nB=nB_add, mB=mB_add)
            sess.insert(kind="observation", labels=self.labels, payload=obs_payload)

            snapshot_payload = dict(nA=float(nA_now), mA=float(mA_now), nB=float(nB_now), mB=float(mB_now))
            sess.insert(kind="snapshot", labels=self.labels, payload=snapshot_payload)

            info_payload = dict(info_time=float(I1))
            sess.insert(kind="info", labels=self.labels, payload=info_payload)

            if due is not None:
                look_idx, planned_t = due
                stat_payload = dict(z=float(z_value))
                sess.insert(kind="stat", labels=self.labels, payload=stat_payload)

                decision_payload = dict(
                    look=int(look_idx),
                    planned_t=float(planned_t),
                    info_time=float(I1),
                    z=float(z_value),
                    boundary=float(boundary) if boundary is not None else 0.0,
                    action=str(action),
                )
                sess.insert(kind="decision", labels=self.labels, payload=decision_payload)


def _ab_workload() -> BinomialABTest:
    con = ibis.duckdb.connect()
    ledger = Ledger(con, "ledger_v14_profile", overwrite=True)
    ab = BinomialABTest(ledger, labels={"experiment_id": "demo"})
    ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0], alpha=0.05)
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
