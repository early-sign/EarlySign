"""EarlySign DSL Design v23 — layered rewrite of BinomialABTest.

This single file mirrors the intended package structure:

* earlysign.core.ledger            → ``Ledger``, ``LedgerSession``, ``LedgerReader``
* earlysign.framework.dsl          → ``StageEmitter`` (define-by-run insert helper)
* earlysign.stats.applications...  → ``BinomialABExecution`` orchestration
* earlysign.api.ab_tests           → ``BinomialABTest`` public API (unchanged surface)

The goal is to keep ibis-native ledger interactions while showing how the
layers compose without actually importing the real packages. All logic stays
self-contained so the script remains executable.
"""

from __future__ import annotations

import cProfile
import json
import math
import pstats
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Self

import ibis
from ibis.backends import BaseBackend
from ibis.expr.types import JSONValue
from ibis.expr.types import Table as IbisTable

__version__ = "23.0.0"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _json_literal(data: Dict[str, Any]) -> JSONValue:
    return ibis.literal(json.dumps(data, sort_keys=True), type="string").cast("json")


def _pooled_z(nA: float, mA: float, nB: float, mB: float) -> float:
    if min(nA, nB) <= 0:
        return 0.0
    pA = mA / nA
    pB = mB / nB
    p_pool = (mA + mB) / (nA + nB)
    variance = p_pool * (1.0 - p_pool) * (1.0 / nA + 1.0 / nB)
    if variance <= 0:
        return 0.0
    return (pB - pA) / math.sqrt(variance)


def _boundary_from_spending(I0: float, I1: float, *, alpha: float, family: str) -> float:
    # Simple spending implementations (demo only).
    spent = max(I1 - I0, 0.0)
    family = family.lower()
    if family in {"obrien_fleming", "obf", "obrien-fleming"}:
        if spent <= 0:
            return float("inf")
        return math.sqrt(2.0 * math.log(1.0 / (alpha * spent)))
    if family in {"pocock"}:
        return 2.4 - 0.3 * spent
    # Fallback conservative bound
    return 3.0 - spent


# ---------------------------------------------------------------------------
# Core layer (conceptually earlysign.core.ledger)
# ---------------------------------------------------------------------------


class Ledger:
    def __init__(self, backend: BaseBackend, table_name: str, *, labels: Optional[Dict[str, Any]] = None, overwrite: bool = True) -> None:
        self.con = backend
        self.table_name = table_name
        self.labels = dict(labels or {})
        schema = ibis.schema(
            dict(id="uuid", ts="timestamp", pkg_version="string", kind="string", labels="json", payload="json")
        )
        if overwrite:
            try:
                self.con.drop_table(self.table_name)
            except Exception:
                pass
        if self.table_name not in self.con.list_tables():
            self.con.create_table(self.table_name, schema=schema)

    def ensure(self) -> None:
        if self.table_name not in self.con.list_tables():
            raise RuntimeError(f"Ledger table {self.table_name} is missing")

    def table(self) -> IbisTable:
        return self.con.table(self.table_name)

    def bind(self, **labels: Any) -> Ledger:
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
        self._jobs: List[Any] = []

    def __enter__(self) -> LedgerSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return
        for job in self._jobs:
            job()

    def insert(self, *, kind: str, payload: Dict[str, Any], labels: Optional[Dict[str, Any]] = None) -> None:
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

    def _scoped_table(self) -> IbisTable:
        tbl = self.ledger.table()
        if not self.ledger.labels:
            return tbl
        lbl = tbl["labels"]
        predicates = [lbl[k].unwrap_as("string") == ibis.literal(str(v)) for k, v in self.ledger.labels.items()]
        return tbl.filter(predicates)

    def table_of_kind(self, kind: str) -> IbisTable:
        tbl = self._scoped_table()
        return tbl.filter(tbl["kind"] == kind)

    def latest_json(
        self,
        kind: str,
        mapping: Dict[str, Tuple[str, str]],
        default: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tbl = self.table_of_kind(kind).order_by(lambda t: t.ts.desc()).limit(1)
        payload = tbl["payload"]
        selects = [payload[field].unwrap_as(dtype).name(alias) for alias, (field, dtype) in mapping.items()]
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
        selects = [payload[field].unwrap_as(dtype).name(alias) for alias, (field, dtype) in mapping.items()]
        df = self.ledger.con.execute(tbl.select(*selects))
        return df.to_dict("records")


# ---------------------------------------------------------------------------
# Framework layer (conceptually earlysign.framework)
# ---------------------------------------------------------------------------


class StageEmitter:
    """Tiny define-by-run helper exposed by the framework layer."""

    def __init__(self, ledger: Ledger, kind: str):
        self.ledger = ledger
        self.kind = kind

    def emit(self, payload: Dict[str, Any]) -> None:
        with self.ledger.session() as sess:
            sess.insert(kind=self.kind, payload=payload)


# ---------------------------------------------------------------------------
# Stats applications execution layer (conceptually earlysign.stats.applications.execution)
# ---------------------------------------------------------------------------


class BinomialABExecution:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.reader = ledger.reader()
        self.observation_stage = StageEmitter(ledger, "observation")
        self.snapshot_stage = StageEmitter(ledger, "snapshot")
        self.info_stage = StageEmitter(ledger, "info")
        self.stat_stage = StageEmitter(ledger, "stat")
        self.decision_stage = StageEmitter(ledger, "decision")

    # --- design helpers --------------------------------------------------

    def _planned_looks(self) -> List[Tuple[int, float]]:
        rows = self.reader.list_json(
            "design-look",
            {"look": ("look", "int64"), "planned_t": ("planned_t", "float64")},
            order_by="planned_t",
        )
        return [(int(r["look"]), float(r["planned_t"])) for r in rows]

    # --- state accessors -------------------------------------------------

    def _latest_snapshot(self) -> Dict[str, float]:
        defaults = {"nA": 0.0, "mA": 0.0, "nB": 0.0, "mB": 0.0}
        return self.reader.latest_json("snapshot", {k: (k, "float64") for k in defaults}, default=defaults)

    def _latest_info(self) -> Dict[str, float]:
        return self.reader.latest_json("info", {"info_time": ("info_time", "float64")}, default={"info_time": 0.0})

    def _latest_design(self) -> Dict[str, Any]:
        return self.reader.latest_json(
            "design",
            {
                "Nmax": ("planned_max_n", "float64"),
                "alpha": ("alpha", "float64"),
                "family": ("spending_family", "string"),
            },
            default={"Nmax": 0.0, "alpha": 0.05, "family": "obrien_fleming"},
        )

    # --- public operations -----------------------------------------------

    def set_design(self, payload: Dict[str, Any]) -> None:
        planned_max_n = float(payload["planned_max_n"])
        alpha = float(payload.get("alpha", 0.05))
        # Accept nested efficacy payloads or direct string
        efficacy = payload.get("efficacy") or payload.get("spending") or {}
        spending_family = efficacy.get("family", payload.get("spending_family", "obrien_fleming"))
        looks = payload.get("planned_info_times") or payload.get("looks") or []
        looks = [float(x) for x in looks]
        with self.ledger.session() as sess:
            sess.insert(
                kind="design",
                payload=dict(planned_max_n=planned_max_n, alpha=alpha, spending_family=str(spending_family)),
            )
            for idx, planned_t in enumerate(looks, start=1):
                sess.insert(kind="design-look", payload=dict(look=idx, planned_t=float(planned_t)))

    def update(self, payload: Dict[str, Any]) -> None:
        snapshot_prev = self._latest_snapshot()
        design = self._latest_design()
        info_prev = self._latest_info()
        if design["Nmax"] <= 0:
            raise RuntimeError("Call set_design() before update().")

        # 1) observation
        obs_payload = dict(
            nA=int(payload["nA"]),
            mA=int(payload["mA"]),
            nB=int(payload["nB"]),
            mB=int(payload["mB"]),
        )
        self.observation_stage.emit(obs_payload)

        # 2) snapshot (use previous snapshot + delta)
        snapshot_now = dict(
            nA=float(snapshot_prev["nA"] + obs_payload["nA"]),
            mA=float(snapshot_prev["mA"] + obs_payload["mA"]),
            nB=float(snapshot_prev["nB"] + obs_payload["nB"]),
            mB=float(snapshot_prev["mB"] + obs_payload["mB"]),
        )
        self.snapshot_stage.emit(snapshot_now)

        # 3) info (re-read latest snapshot to stay ledger-native)
        snapshot_latest = self._latest_snapshot()
        info_now = (snapshot_latest["nA"] + snapshot_latest["nB"]) / design["Nmax"]
        self.info_stage.emit(dict(info_time=float(info_now)))

        # 4) stat + decision when due
        looks = self._planned_looks()
        I0 = float(info_prev["info_time"])
        I1 = float(info_now)
        due = next(((idx, t) for idx, t in looks if I0 < t <= I1), None)
        if due is None:
            return
        z_value = _pooled_z(
            snapshot_latest["nA"],
            snapshot_latest["mA"],
            snapshot_latest["nB"],
            snapshot_latest["mB"],
        )
        boundary = _boundary_from_spending(I0, I1, alpha=design["alpha"], family=design["family"])
        action = "stop_efficacy" if abs(z_value) >= boundary else "continue"
        self.stat_stage.emit(dict(z=float(z_value)))
        self.decision_stage.emit(
            dict(
                look=int(due[0]),
                planned_t=float(due[1]),
                info_time=float(I1),
                z=float(z_value),
                boundary=float(boundary),
                action=action,
            )
        )


# ---------------------------------------------------------------------------
# API layer (conceptually earlysign.api.ab_tests)
# ---------------------------------------------------------------------------


class BinomialABTest:
    """Public API compatible with ``earlysign.api.ab_tests.BinomialABTest``."""

    def __init__(
        self,
        connector: BaseBackend | str,
        experiment_id: str,
        table_name: Optional[str] = None,
        *,
        ibis_cache: Optional[Any] = None,
    ) -> None:
        if isinstance(connector, str):
            self.connector = ibis.connect(connector)
        elif isinstance(connector, BaseBackend):
            self.connector = connector
        else:
            raise TypeError("connector must be an ibis backend or connection string")
        self.experiment_id = experiment_id
        ledger_name = table_name if table_name is not None else experiment_id
        base_ledger = Ledger(self.connector, ledger_name).bind(experiment_id=experiment_id)
        base_ledger.ensure()
        self.ledger = base_ledger
        self._ibis_cache = ibis_cache  # kept for API parity
        self._execution = BinomialABExecution(self.ledger)

    def set_design(self, payload: Dict[str, Any]) -> None:
        self._execution.set_design(payload)

    def _execute_expr(self, expr: Any) -> Any:
        # Minimal stand-in for the original caching execute helper.
        return self.connector.execute(expr)

    def update(self, payload: Dict[str, Any]) -> None:
        self._execution.update(payload)


# ---------------------------------------------------------------------------
# Profiling harness (mirrors previous versions)
# ---------------------------------------------------------------------------


def _ab_workload() -> BinomialABTest:
    test = BinomialABTest(ibis.duckdb.connect(), "ledger_v23_demo")
    design = {
        "alpha": 0.05,
        "planned_max_n": 1000,
        "planned_info_times": [0.25, 0.5, 0.75, 1.0],
        "efficacy": {"family": "obrien_fleming"},
    }
    test.set_design(design)
    for batch in [
        {"nA": 100, "mA": 10, "nB": 100, "mB": 12},
        {"nA": 50, "mA": 5, "nB": 50, "mB": 6},
        {"nA": 150, "mA": 10, "nB": 150, "mB": 20},
        {"nA": 100, "mA": 5, "nB": 100, "mB": 35},
    ]:
        test.update(batch)
    return test


def _profile_run() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    ab = _ab_workload()
    profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
    print(ab.ledger.con.execute(ab.ledger.table()))


if __name__ == "__main__":
    _profile_run()
