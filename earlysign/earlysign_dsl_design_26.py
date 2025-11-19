"""EarlySign DSL Design v26 — multi-tenant ledger DSL demo.

This single file mirrors the intended package layout:

* ``earlysign.core``        → ``Ledger`` + ``LedgerSession`` (pure ibis plumbing)
* ``earlysign.framework``   → strongly-typed ``Record`` classes that derive their``kind``
                              from the payload schema/class name (similar to
                              ``earlysign.framework.records``).
* ``earlysign.applications.execution`` → composed procedures (here inlined via helpers)
* ``earlysign.api``         → ``BinomialABTest`` (public surface identical to the real API)

Multiple experiments can share a single physical ledger table: each ``BinomialABTest``
binds to a label scope (``experiment_id`` here), and the core layer ensures that all
reads/writes stay inside that scope without framework/application code having to care.
The code remains self-contained so the script can run on its own while illustrating the
layering strategy.
"""

from __future__ import annotations

import cProfile
import json
import math
import pstats
from typing import Any, Dict, List, Optional, Tuple

import ibis
from ibis.backends import BaseBackend
from ibis.expr.types import JSONValue, Table

__version__ = "26.0.0"


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


def _boundary_from_spending(
    I0: float, I1: float, *, alpha: float, family: str
) -> float:
    # Demo-friendly spending functions.
    spent = max(I1 - I0, 1e-9)
    fam = family.lower()
    if fam in {"obrien_fleming", "obf", "obrien-fleming"}:
        return math.sqrt(2.0 * math.log(1.0 / (alpha * spent)))
    if fam == "pocock":
        return 2.4 - 0.3 * spent
    return 3.0 - spent


# ---------------------------------------------------------------------------
# Core layer (conceptually earlysign.core.ledger)
# ---------------------------------------------------------------------------


class Ledger:
    def __init__(
        self,
        backend: BaseBackend,
        table_name: str,
        *,
        labels: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        self.con = backend
        self.table_name = table_name
        self.labels = dict(labels or {})
        self._schema = ibis.schema(
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
        self.ensure()

    def ensure(self) -> None:
        if self.table_name not in self.con.list_tables():
            self.con.create_table(self.table_name, schema=self._schema)

    def bind(self, **labels: Any) -> Ledger:
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table_name, labels=merged)

    def table(self) -> Table:
        return self.con.table(self.table_name)

    def scoped_table(self) -> Table:
        tbl = self.table()
        if not self.labels:
            return tbl
        lbl = tbl["labels"]
        predicate = ibis.literal(True)
        for key, value in self.labels.items():
            predicate &= lbl[key].unwrap_as("string") == ibis.literal(str(value))
        return tbl.filter(predicate)

    def session(self) -> "LedgerSession":
        return LedgerSession(self)

    def _row_expr(
        self,
        *,
        kind: str,
        payload: Dict[str, Any],
        labels: Optional[Dict[str, Any]] = None,
    ) -> Table:
        merged_labels = dict(self.labels)
        if labels:
            merged_labels.update(labels)
        return ibis.memtable([{"_anchor": 1}]).select(
            id=ibis.uuid(),
            ts=ibis.now(),
            pkg_version=ibis.literal(__version__),
            kind=ibis.literal(kind),
            labels=_json_literal(merged_labels),
            payload=_json_literal(payload),
        )


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

    def insert(
        self,
        *,
        kind: str,
        payload: Dict[str, Any],
        labels: Optional[Dict[str, Any]] = None,
    ) -> None:
        row = self.ledger._row_expr(kind=kind, payload=payload, labels=labels)

        def _exec() -> None:
            self.ledger.con.insert(self.ledger.table_name, row)

        self._jobs.append(_exec)


# ---------------------------------------------------------------------------
# Framework layer (conceptually earlysign.framework.records)
# ---------------------------------------------------------------------------


class RecordBase:
    """Typed convenience wrapper that determines ``kind`` and schema per subclass."""

    KIND: Optional[str] = None
    SCHEMA: Dict[str, Tuple[str, str]] = {}
    DEFAULT: Optional[Dict[str, Any]] = None

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.kind = self._resolve_kind()

    def _resolve_kind(self) -> str:
        if self.KIND:
            return self.KIND
        import re

        name = self.__class__.__name__
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        return snake.replace("_record", "").replace("__", "_")

    def schema(self) -> Dict[str, Tuple[str, str]]:
        return self.SCHEMA

    def default(self) -> Optional[Dict[str, Any]]:
        return dict(self.DEFAULT) if self.DEFAULT is not None else None

    def insert(self, payload: Dict[str, Any]) -> None:
        with self.ledger.session() as sess:
            sess.insert(kind=self.kind, payload=payload)

    def latest(self, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tbl = self.ledger.scoped_table()
        tbl = tbl.filter(tbl.kind == self.kind).order_by(tbl.ts.desc()).limit(1)
        mapping = self.schema()
        fallback = default if default is not None else self.default()
        if not mapping:
            return dict(fallback) if fallback else {}
        payload = tbl.payload
        selects = [
            payload[field].unwrap_as(dtype).name(alias)
            for alias, (field, dtype) in mapping.items()
        ]
        df = self.ledger.con.execute(tbl.select(*selects))
        if df.empty:
            return dict(fallback) if fallback else {}
        return df.to_dict("records")[0]

    def list(self, *, order_by: Optional[str] = None) -> List[Dict[str, Any]]:
        tbl = self.ledger.scoped_table()
        tbl = tbl.filter(tbl.kind == self.kind)
        payload = tbl.payload
        mapping = self.schema()
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


class DesignRecord(RecordBase):
    KIND = "design"
    SCHEMA = {
        "planned_max_n": ("planned_max_n", "float64"),
        "alpha": ("alpha", "float64"),
        "spending_family": ("spending_family", "string"),
        "planned_info_times": ("planned_info_times", "string"),
    }
    DEFAULT = {
        "planned_max_n": 0.0,
        "alpha": 0.05,
        "spending_family": "obrien_fleming",
        "planned_info_times": "[]",
    }


class ObservationRecord(RecordBase):
    KIND = "observation"
    SCHEMA = {
        "nA": ("nA", "int64"),
        "mA": ("mA", "int64"),
        "nB": ("nB", "int64"),
        "mB": ("mB", "int64"),
    }


class SnapshotRecord(RecordBase):
    KIND = "snapshot"
    SCHEMA = {
        "nA": ("nA", "int64"),
        "mA": ("mA", "int64"),
        "nB": ("nB", "int64"),
        "mB": ("mB", "int64"),
    }
    DEFAULT = {"nA": 0, "mA": 0, "nB": 0, "mB": 0}


class InfoRecord(RecordBase):
    KIND = "info"
    SCHEMA = {"info_time": ("info_time", "float64")}
    DEFAULT = {"info_time": 0.0}


class StatRecord(RecordBase):
    KIND = "stat"
    SCHEMA = {"z": ("z", "float64")}


class DecisionRecord(RecordBase):
    KIND = "decision"
    SCHEMA = {
        "look": ("look", "int64"),
        "planned_t": ("planned_t", "float64"),
        "info_time": ("info_time", "float64"),
        "z": ("z", "float64"),
        "boundary": ("boundary", "float64"),
        "action": ("action", "string"),
    }


def _build_records(ledger: Ledger) -> Dict[str, RecordBase]:
    return {
        "design": DesignRecord(ledger),
        "observation": ObservationRecord(ledger),
        "snapshot": SnapshotRecord(ledger),
        "info": InfoRecord(ledger),
        "stat": StatRecord(ledger),
        "decision": DecisionRecord(ledger),
    }


# ---------------------------------------------------------------------------
# API layer (conceptually earlysign.api.ab_tests)
# ---------------------------------------------------------------------------


class BinomialABTest:
    """Public API-compatible implementation backed by the new DSL."""

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
        base_ledger = Ledger(self.connector, ledger_name, overwrite=False).bind(
            experiment_id=experiment_id
        )
        base_ledger.ensure()
        self.ledger = base_ledger
        self._records = _build_records(self.ledger)
        self._ibis_cache = ibis_cache

    # --- API helpers -----------------------------------------------------

    def _planned_looks(
        self, design: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, float]]:
        design_data = design or self._latest_design()
        planned_times = json.loads(design_data.get("planned_info_times", "[]"))
        return [(idx + 1, float(t)) for idx, t in enumerate(planned_times)]

    def _latest_snapshot(self) -> Dict[str, float]:
        return self._records["snapshot"].latest()

    def _latest_info(self) -> Dict[str, float]:
        return self._records["info"].latest()

    def _latest_design(self) -> Dict[str, Any]:
        design = self._records["design"].latest(
            {
                "planned_max_n": 0.0,
                "alpha": 0.05,
                "spending_family": "obrien_fleming",
                "planned_info_times": "[]",
            }
        )
        return dict(design)

    # --- Public API (unchanged signatures) -------------------------------

    def set_design(self, payload: Dict[str, Any]) -> None:
        planned_max_n = float(payload["planned_max_n"])
        alpha = float(payload.get("alpha", 0.05))
        efficacy = payload.get("efficacy") or payload.get("spending") or {}
        spending_family = efficacy.get(
            "family", payload.get("spending_family", "obrien_fleming")
        )
        looks = payload.get("planned_info_times") or payload.get("looks") or []
        with self.ledger.session() as sess:
            sess.insert(
                kind="design",
                payload=dict(
                    planned_max_n=planned_max_n,
                    alpha=alpha,
                    spending_family=str(spending_family),
                    planned_info_times=json.dumps([float(x) for x in looks]),
                ),
            )

    def _execute_expr(self, expr: Any) -> Any:
        return self.connector.execute(expr)

    def update(self, payload: Dict[str, Any]) -> None:
        design = self._latest_design()
        info_prev = self._records["info"].latest()
        if design["planned_max_n"] <= 0:
            raise RuntimeError("Call set_design() before update().")

        self._observation_op(self._records, payload)
        self._snapshot_op(self._records, payload)
        self._info_op(self._records, design)
        looks = self._planned_looks(design)
        self._decision_op(self._records, design, info_prev, looks)

    @staticmethod
    def _observation_op(
        records: Dict[str, RecordBase], payload: Dict[str, Any]
    ) -> None:
        obs_payload = dict(
            nA=int(payload["nA"]),
            mA=int(payload["mA"]),
            nB=int(payload["nB"]),
            mB=int(payload["mB"]),
        )
        records["observation"].insert(obs_payload)

    @staticmethod
    def _snapshot_op(records: Dict[str, RecordBase], payload: Dict[str, Any]) -> None:
        prev_snapshot = records["snapshot"].latest()
        snapshot_now = dict(
            nA=int(prev_snapshot["nA"] + payload["nA"]),
            mA=int(prev_snapshot["mA"] + payload["mA"]),
            nB=int(prev_snapshot["nB"] + payload["nB"]),
            mB=int(prev_snapshot["mB"] + payload["mB"]),
        )
        records["snapshot"].insert(snapshot_now)

    @staticmethod
    def _info_op(records: Dict[str, RecordBase], design: Dict[str, Any]) -> None:
        snapshot_latest = records["snapshot"].latest()
        info_now = (snapshot_latest["nA"] + snapshot_latest["nB"]) / design[
            "planned_max_n"
        ]
        records["info"].insert({"info_time": float(info_now)})

    @staticmethod
    def _decision_op(
        records: Dict[str, RecordBase],
        design: Dict[str, Any],
        info_prev: Dict[str, Any],
        looks: List[Tuple[int, float]],
    ) -> None:
        info_latest = records["info"].latest()
        I0 = float(info_prev["info_time"])
        I1 = float(info_latest["info_time"])
        due = next(((idx, t) for idx, t in looks if I0 < t <= I1), None)
        if due is None:
            return
        snapshot_latest = records["snapshot"].latest()
        z_value = _pooled_z(
            snapshot_latest["nA"],
            snapshot_latest["mA"],
            snapshot_latest["nB"],
            snapshot_latest["mB"],
        )
        boundary = _boundary_from_spending(
            I0, I1, alpha=design["alpha"], family=design["spending_family"]
        )
        action = "stop_efficacy" if abs(z_value) >= boundary else "continue"
        records["stat"].insert({"z": float(z_value)})
        records["decision"].insert(
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
# Profiling harness (mirrors previous versions)
# ---------------------------------------------------------------------------


def _ab_workload() -> Tuple[BinomialABTest, BinomialABTest]:
    con = ibis.duckdb.connect()
    shared_table = "ledger_v26_shared"
    con.raw_sql(f"DROP TABLE IF EXISTS {shared_table}")
    test_a = BinomialABTest(con, "exp_alpha", table_name=shared_table)
    test_b = BinomialABTest(con, "exp_beta", table_name=shared_table)
    design = {
        "alpha": 0.05,
        "planned_max_n": 1000,
        "planned_info_times": [0.25, 0.5, 0.75, 1.0],
        "efficacy": {"family": "obrien_fleming"},
    }
    for test in (test_a, test_b):
        test.set_design(design)
    updates_a = [
        {"nA": 80, "mA": 8, "nB": 80, "mB": 9},
        {"nA": 60, "mA": 6, "nB": 60, "mB": 8},
        {"nA": 200, "mA": 18, "nB": 200, "mB": 25},
    ]
    updates_b = [
        {"nA": 100, "mA": 10, "nB": 100, "mB": 12},
        {"nA": 50, "mA": 5, "nB": 50, "mB": 6},
        {"nA": 150, "mA": 10, "nB": 150, "mB": 20},
        {"nA": 100, "mA": 5, "nB": 100, "mB": 35},
    ]
    for payload in updates_a:
        test_a.update(payload)
    for payload in updates_b:
        test_b.update(payload)
    return test_a, test_b


def _profile_run() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    test_a, test_b = _ab_workload()
    profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
    # Print combined ledger table (both experiments share the same table)
    shared_table_expr = test_a.ledger.table()
    print(test_a.ledger.con.execute(shared_table_expr.order_by("labels", "ts")))


if __name__ == "__main__":
    _profile_run()
