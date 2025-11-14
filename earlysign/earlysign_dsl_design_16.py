"""EarlySign DSL Design v16 - cached ibis plan demo.

Doctest (A/B test with on-the-fly spending)
-------------------------------------------
>>> import ibis
>>> con = ibis.duckdb.connect()
>>> ledger = Ledger(con, "ledger_dsl16_doctest", overwrite=True)
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

# NOTE: This variant validates expression-level caching by compiling the ibis plan once
#       and reusing it across updates via placeholder substitution to study planner costs.

import cProfile
import json
import math
import pstats
from datetime import datetime
from typing import Any, Callable, Dict, List, Sequence

import ibis
from ibis.backends import BaseBackend
from ibis.expr.types import (
    BooleanValue,
    FloatingValue,
    IntegerValue,
    JSONValue,
    StringValue,
    Table,
)

__version__ = "16.0.0"


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


def _label_predicates(table: Table, labels: Dict[str, Any]) -> List[BooleanValue]:
    """Return boolean predicates matching the provided labels for the table."""

    labels_col = table["labels"]
    return [
        labels_col[key].unwrap_as("string") == ibis.literal(str(value))
        for key, value in labels.items()
    ]


def _cdf_normal_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inv_scalar(p: float) -> float:
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


def _alpha_spent_scalar(t: float, alpha: float, family: str) -> float:
    t = min(max(t, 1e-12), 1.0)
    family_key = family.lower()
    if family_key in {"of", "obrien_fleming", "o'brien_fleming", "obrien-fleming"}:
        z = _phi_inv_scalar(1.0 - alpha / 2.0)
        return 2.0 - 2.0 * _cdf_normal_scalar(z / math.sqrt(t))
    if family_key == "pocock":
        return alpha * math.log(1.0 + (math.e - 1.0) * t)
    raise ValueError(f"Unknown spending family: {family}")


def _boundary_from_spending_scalar(
    I0: float, I1: float, *, alpha: float, family: str
) -> float:
    A1 = _alpha_spent_scalar(I1, alpha, family)
    A0 = _alpha_spent_scalar(I0, alpha, family) if I0 > 0 else 0.0
    local = max(A1 - A0, 1e-16)
    return _phi_inv_scalar(1.0 - local / 2.0)


def _pooled_z_expr(
    nA: FloatingValue, mA: FloatingValue, nB: FloatingValue, mB: FloatingValue
) -> FloatingValue:
    """Classic pooled two-proportion z-statistic expressed with ibis."""

    eps = ibis.literal(1e-9)
    zero = ibis.literal(0.0)
    valid = (nA > 0) & (nB > 0)
    nA_safe = ibis.greatest(nA, eps)
    nB_safe = ibis.greatest(nB, eps)
    total_n = nA + nB
    total_safe = ibis.greatest(total_n, eps)
    pA = mA / nA_safe
    pB = mB / nB_safe
    pooled = (mA + mB) / total_safe
    denom = (pooled * (1 - pooled) * (1 / nA_safe + 1 / nB_safe) + eps).sqrt()
    return ibis.ifelse(valid & (denom > eps), (pB - pA) / denom, zero)


class Ledger:
    """Ibis-backed ledger storing immutable JSON payloads."""

    def __init__(
        self, con: BaseBackend, table_name: str, overwrite: bool = True
    ) -> None:
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
    def table(self) -> Table:
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

    def _enqueue_insert(self, select_expr: Table) -> None:
        con = self.ledger.con
        table_name = self.ledger.table_name

        def _exec() -> None:
            con.insert(table_name, select_expr)

        self._jobs.append(_exec)

    def insert(
        self, *, kind: str, labels: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        labels_expr = _json_literal({k: str(v) for k, v in labels.items()})
        payload_json = _json_literal(payload)

        ## magic: we want to create an empty table and convert it to a one-row table, but we need to start with at least one dummy column.
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
        self._delta_schema = ibis.schema(
            {
                "nA_add": "float64",
                "mA_add": "float64",
                "nB_add": "float64",
                "mB_add": "float64",
            }
        )
        self._delta_placeholder = ibis.table(
            self._delta_schema, name="dsl16_delta_inputs"
        )
        self._state_plan_op = None

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
                sess.insert(
                    kind="design-look", labels=self.labels, payload=look_payload
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_labels(self, table: Table) -> Table:
        if not self.labels:
            return table
        predicates = _label_predicates(table, self.labels)
        return table.filter(predicates) if predicates else table

    def _scoped_kind(self, kind: str) -> Table:
        tbl = self.ledger.table
        filtered = tbl.filter(tbl["kind"] == ibis.literal(kind))
        return self._apply_labels(filtered)

    def _latest_snapshot_expr(self) -> Table:
        tbl = self._scoped_kind("snapshot")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_snapshot=tbl.ts,
            prev_nA=json_get_f64(payload, "nA").cast("float64"),
            prev_mA=json_get_f64(payload, "mA").cast("float64"),
            prev_nB=json_get_f64(payload, "nB").cast("float64"),
            prev_mB=json_get_f64(payload, "mB").cast("float64"),
        )
        default_snapshot = ibis.memtable(
            [
                {
                    "ts_snapshot": datetime(1970, 1, 1),
                    "prev_nA": 0.0,
                    "prev_mA": 0.0,
                    "prev_nB": 0.0,
                    "prev_mB": 0.0,
                }
            ],
            schema=ibis.schema(
                {
                    "ts_snapshot": "timestamp(6)",
                    "prev_nA": "float64",
                    "prev_mA": "float64",
                    "prev_nB": "float64",
                    "prev_mB": "float64",
                }
            ),
        )
        union = typed.union(default_snapshot)
        return union.order_by(union.ts_snapshot.desc()).limit(1)

    def _latest_design_expr(self) -> Table:
        tbl = self._scoped_kind("design")
        payload = tbl["payload"]
        typed = tbl.select(
            ts_design=tbl.ts,
            planned_max_n=json_get_f64(payload, "planned_max_n").cast("float64"),
            alpha=json_get_f64(payload, "alpha").cast("float64"),
            family=json_get_str(payload, "spending_family"),
            has_design=ibis.literal(1, type="int64"),
        )
        default_design = ibis.memtable(
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
        union = typed.union(default_design)
        return union.order_by(union.has_design.desc(), union.ts_design.desc()).limit(1)

    def _looks_table(self) -> Table:
        tbl = self._scoped_kind("design-look")
        payload = tbl["payload"]
        return tbl.select(
            look=json_get_i64(payload, "look").cast("int64"),
            planned_t=json_get_f64(payload, "planned_t").cast("float64"),
        ).order_by("planned_t")

    def _state_with_due_frame(self, delta: Table) -> Table:
        snapshot = self._latest_snapshot_expr()
        design = self._latest_design_expr()
        base = delta.cross_join(snapshot).cross_join(design)
        state = base.mutate(
            nA_now=lambda t: t.prev_nA + t.nA_add,
            mA_now=lambda t: t.prev_mA + t.mA_add,
            nB_now=lambda t: t.prev_nB + t.nB_add,
            mB_now=lambda t: t.prev_mB + t.mB_add,
        )
        state = state.mutate(
            total_prev=lambda t: t.prev_nA + t.prev_nB,
            total_now=lambda t: t.nA_now + t.nB_now,
        )
        state = state.mutate(
            I0=lambda t: ibis.ifelse(
                t.planned_max_n > 0, t.total_prev / t.planned_max_n, 0.0
            ),
            I1=lambda t: ibis.ifelse(
                t.planned_max_n > 0, t.total_now / t.planned_max_n, 0.0
            ),
        )
        state = state.mutate(
            z_value=lambda t: _pooled_z_expr(t.nA_now, t.mA_now, t.nB_now, t.mB_now),
        )

        state_cols = [
            "nA_add",
            "mA_add",
            "nB_add",
            "mB_add",
            "prev_nA",
            "prev_mA",
            "prev_nB",
            "prev_mB",
            "planned_max_n",
            "alpha",
            "family",
            "has_design",
            "nA_now",
            "mA_now",
            "nB_now",
            "mB_now",
            "I0",
            "I1",
            "z_value",
        ]
        state = state.select(*state_cols)

        looks = self._looks_table()
        due_candidates = looks.cross_join(state).filter(
            (looks.planned_t > state.I0) & (looks.planned_t <= state.I1)
        )
        due_candidates = (
            due_candidates.order_by(looks.planned_t)
            .limit(1)
            .mutate(selector=ibis.literal(1))
        )
        due_candidates = due_candidates.select(
            *[due_candidates[col] for col in state_cols],
            due_look=due_candidates["look"],
            due_planned_t=due_candidates["planned_t"],
            selector=due_candidates.selector,
        )
        default_due = state.select(
            *[state[col] for col in state_cols],
            due_look=ibis.literal(None, type="int64"),
            due_planned_t=ibis.literal(None, type="float64"),
            selector=ibis.literal(0),
        )
        combined = due_candidates.union(default_due)
        result = (
            combined.order_by(combined.selector.desc())
            .limit(1)
            .mutate(
                has_due=combined.selector,
            )
            .drop("selector")
        )
        return result

    def _state_plan_op_cached(self):
        if self._state_plan_op is None:
            expr = self._state_with_due_frame(self._delta_placeholder)
            self._state_plan_op = expr.op()
        return self._state_plan_op

    def _execute_state_plan(self, delta_expr: Table) -> Dict[str, Any]:
        template = self._state_plan_op_cached()
        replaced = template.replace(
            {self._delta_placeholder.op(): delta_expr.op()}
        ).to_expr()
        state_df = self.ledger.con.execute(replaced)
        if state_df.empty:
            return {}
        return state_df.to_dict("records")[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, payload: Dict[str, int]) -> None:
        delta_expr = ibis.memtable(
            [
                dict(
                    nA_add=float(payload["nA"]),
                    mA_add=float(payload["mA"]),
                    nB_add=float(payload["nB"]),
                    mB_add=float(payload["mB"]),
                )
            ],
            schema=self._delta_schema,
        )
        state = self._execute_state_plan(delta_expr)
        if not state:
            raise RuntimeError("Failed to derive state for update.")
        if not state.get("has_design"):
            raise RuntimeError("Design must be configured before calling update().")
        alpha_val = float(state["alpha"])
        family_val = str(state["family"])

        nA_add = int(payload["nA"])
        mA_add = int(payload["mA"])
        nB_add = int(payload["nB"])
        mB_add = int(payload["mB"])

        nA_now = float(state["nA_now"])
        mA_now = float(state["mA_now"])
        nB_now = float(state["nB_now"])
        mB_now = float(state["mB_now"])
        info_time = float(state["I1"])
        z_value = float(state["z_value"])
        I0_val = float(state["I0"])
        due_flag = bool(state.get("has_due"))
        boundary_val = None
        action_val = None
        if due_flag:
            boundary_val = _boundary_from_spending_scalar(
                I0_val,
                info_time,
                alpha=alpha_val,
                family=family_val,
            )
            action_val = "stop_efficacy" if abs(z_value) >= boundary_val else "continue"

        with self.ledger.session() as sess:
            obs_payload = dict(nA=nA_add, mA=mA_add, nB=nB_add, mB=mB_add)
            sess.insert(kind="observation", labels=self.labels, payload=obs_payload)

            snapshot_payload = dict(
                nA=float(nA_now), mA=float(mA_now), nB=float(nB_now), mB=float(mB_now)
            )
            sess.insert(kind="snapshot", labels=self.labels, payload=snapshot_payload)

            info_payload = dict(info_time=info_time)
            sess.insert(kind="info", labels=self.labels, payload=info_payload)

            if due_flag:
                stat_payload = dict(z=z_value)
                sess.insert(kind="stat", labels=self.labels, payload=stat_payload)

                decision_payload = dict(
                    look=int(state["due_look"]),
                    planned_t=float(state["due_planned_t"]),
                    info_time=info_time,
                    z=z_value,
                    boundary=float(boundary_val),
                    action=str(action_val),
                )
                sess.insert(
                    kind="decision", labels=self.labels, payload=decision_payload
                )


def _ab_workload() -> BinomialABTest:
    con = ibis.duckdb.connect()
    ledger = Ledger(con, "ledger_v16_profile", overwrite=True)
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
