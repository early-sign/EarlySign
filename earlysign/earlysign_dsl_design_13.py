# # -*- coding: utf-8 -*-
# """
# EarlySign DSL Design v13 (JSON API only; no JSON helpers)

# Doctest
# -------
# The doctest demonstrates:
# 1) Creating an in-memory DuckDB-backed ledger
# 2) Running a minimal A/B test workflow with three updates
# 3) Reading the latest aggregated counts from JSON payload (no recomputation)
# 4) Ensuring z/e/decision records are derived by reading the latest records

# >>> ab = ABTestLedger()  # in-memory
# >>> # Three small updates
# >>> ab.update({"nA": 4, "mA": 1, "nB": 4, "mB": 2})
# >>> ab.update({"nA": 3, "mA": 1, "nB": 3, "mB": 1})
# >>> ab.update({"nA": 5, "mA": 2, "nB": 5, "mB": 3})
# >>> # Read back latest aggregate via JSON indexing; unwrap to ints
# >>> tbl = ab.ledger.table
# >>> latest_agg = (
# ...     tbl.filter((tbl.kind == "agg") & (tbl.labels["test"].str == "ab"))
# ...        .order_by(tbl.ts.desc())
# ...        .limit(1)
# ... )
# >>> nA = latest_agg.payload["nA"].unwrap_as("int64")
# >>> mA = latest_agg.payload["mA"].unwrap_as("int64")
# >>> nB = latest_agg.payload["nB"].unwrap_as("int64")
# >>> mB = latest_agg.payload["mB"].unwrap_as("int64")
# >>> out = ab.con.execute(latest_agg.select(nA.name("nA"), mA.name("mA"), nB.name("nB"), mB.name("mB")))
# >>> sums = out.to_dict("records")[0]
# >>> sums["nA"] == 12 and sums["mA"] == 4 and sums["nB"] == 12 and sums["mB"] == 6
# True
# >>> # Latest z / e / decision are present and JSON-typed
# >>> latest_z = (
# ...     tbl.filter((tbl.kind == "z") & (tbl.labels["test"].str == "ab"))
# ...        .order_by(tbl.ts.desc()).limit(1)
# ... )
# >>> latest_e = (
# ...     tbl.filter((tbl.kind == "e") & (tbl.labels["test"].str == "ab"))
# ...        .order_by(tbl.ts.desc()).limit(1)
# ... )
# >>> latest_dec = (
# ...     tbl.filter((tbl.kind == "decision") & (tbl.labels["test"].str == "ab"))
# ...        .order_by(tbl.ts.desc()).limit(1)
# ... )
# >>> zdf = ab.con.execute(latest_z.select(latest_z.payload["z"].unwrap_as("float64").name("z")))
# >>> edf = ab.con.execute(latest_e.select(latest_e.payload["e"].unwrap_as("float64").name("e")))
# >>> ddf = ab.con.execute(latest_dec.select(latest_dec.payload["decision"].str.name("decision")))
# >>> isinstance(zdf["z"][0], float) and isinstance(edf["e"][0], float) and isinstance(ddf["decision"][0], str)
# True
# """

# from __future__ import annotations

# import cProfile
# import math
# import pstats
# from typing import Any, Dict, List, Optional

# import ibis
# import ibis.expr.datatypes as dt
# import ibis.expr.schema as sch


# # ----------------------------------------------------------------------
# # Version tag to record in each row (auto-filled by LedgerSession)
# # ----------------------------------------------------------------------
# __version__ = "v13-jsonapi"


# # ----------------------------------------------------------------------
# # Ledger storage (DuckDB + Ibis JSON dtypes)
# # ----------------------------------------------------------------------
# class Ledger:
#     """A minimal ledger backed by DuckDB (in-memory) with JSON columns.

#     Columns
#     -------
#     - id:           string (uuid)
#     - ts:           timestamp (now)
#     - pkg_version:  string (module version)
#     - kind:         string (record type: 'agg'|'z'|'e'|'decision'|...)
#     - labels:       json   (user-specified labels as JSON)
#     - payload:      json   (record body)
#     """

#     def __init__(self, con: Optional[ibis.Backend] = None, table_name: str = "ledger"):
#         self.con = con or ibis.duckdb.connect()  # in-memory
#         self.table_name = table_name
#         self.table = self._ensure_table()

#     def _ensure_table(self):
#         schema = sch.schema(
#             dict(
#                 id=dt.string,
#                 ts=dt.timestamp,
#                 pkg_version=dt.string,
#                 kind=dt.string,
#                 labels=dt.json,
#                 payload=dt.json,
#             )
#         )
#         if self.table_name not in self.con.list_tables():
#             self.con.create_table(self.table_name, schema=schema)
#         return self.con.table(self.table_name)

#     def session(self) -> "LedgerSession":
#         """Open a session that queues INSERT ... SELECT statements built with Ibis.

#         The session:
#         - Auto-fills id/ts/pkg_version on every inserted row.
#         - Queues each insert as an Ibis `table.insert(select_expr)` call.
#         - Executes them in order when exiting the context manager.
#         """
#         return LedgerSession(self)


# # class LedgerSession:
# #     """Queue INSERT FROM SELECT statements and execute them on context exit.

# #     Notes
# #     -----
# #     - We rely on Ibis-native expressions only; no raw SQL strings are built here.
# #     - Each `insert_select(expr)` appends an insert job that will be executed in order.
# #     - Auto fields (uuid/now/version) are populated inside the select expression.
# #     """

# #     def __init__(self, ledger: Ledger):
# #         self.ledger = ledger
# #         self._jobs: List[Any] = []  # list of callables executing insert

# #     def __enter__(self) -> "LedgerSession":
# #         return self

# #     def __exit__(self, exc_type, exc, tb) -> None:
# #         if exc_type is not None:
# #             return  # do not execute queued jobs on error
# #         for job in self._jobs:
# #             job()

# #     def insert_from_select(self, select_expr):
# #         """Queue: INSERT INTO ledger (cols...) SELECT ... (expr)."""
# #         # tbl = self.ledger.table

# #         def _exec():
# #             select_expr.insert()
# #             # self.ledger.con.insert(self.ledger.table, select_expr.execute())

# #         self._jobs.append(_exec)

# #     def insert(self, *, kind: str, labels: Dict[str, Any], payload_expr):
# #         """Build a row-select with auto fields and queue it for insertion.

# #         Parameters
# #         ----------
# #         kind : str
# #             Record kind.
# #         labels : dict
# #             JSON-like labels (keys -> literals). Built via struct(...).cast('json').
# #         payload_expr : Ibis expression of dtype json
# #             JSON object expression; values may depend on subqueries.
# #         """
# #         # tbl = self.ledger.table
# #         # Build JSON labels via struct + cast('json') to avoid json_object()
# #         labels_struct = ibis.struct({k: ibis.literal(v) for k, v in labels.items()})
# #         # labels_json = labels_struct.cast("json")

# #         # row_select = ibis.values(
# #         #     [[ibis.uuid(), ibis.now(), ibis.literal(__version__),
# #         #     ibis.literal(kind), labels, payload_expr]],
# #         #     schema=ibis.schema({
# #         #         "uuid": "uuid",
# #         #         "ts": "timestamp",
# #         #         "pkg_version": "string",
# #         #         "kind": "string",
# #         #         "labels": "json",
# #         #         "payload": "json",
# #         #     })
# #         # )
# #         from ibis import _
# #         row_expr = _.select(
# #             uuid=ibis.uuid(),
# # #             ts=ibis.now(),
# # #             pkg_version=ibis.literal(__version__),
# # #             kind=ibis.literal(kind),
# # #             labels=labels_struct,
# # #             payload=payload_expr.cast("json"),
# # #         )

# # #         # ここでは実行しない：セッションの遅延実行キューに積む
# # #         # self._ops.append(lambda con=self.con, expr=row_expr: con.insert(self.table_name, expr))
# # #         self.insert_from_select(row_expr)
# # class LedgerSession:
# #     """Queue INSERT ... SELECT jobs and run them in order on context exit (Ibis-native)."""

# #     def __init__(self, ledger: "Ledger"):
# #         self.ledger = ledger               # has: .con (backend), .name (str), .table (ibis Table), .pkg_version (str)
# #         self._jobs: List[Any] = []

# #     def __enter__(self) -> "LedgerSession":
# #         return self

# #     def __exit__(self, exc_type, exc, tb) -> None:
# #         if exc_type is not None:
# #             return  # do not execute queued jobs on error
# #         for job in self._jobs:
# #             job()

# #     def insert_from_select(self, select_expr: ibis.ir.Table):
# #         """Queue: INSERT INTO <ledger> SELECT ..."""
# #         def _exec():
# #             # Table.insert(...) 実装差を避けて backend API を直接使う
# #             self.ledger.con.insert(self.ledger.name, obj=select_expr)
# #         self._jobs.append(_exec)

# #     def insert(self, *, kind: str, labels: Dict[str, Any], payload_expr: ibis.ir.Value):
# #         """Build a single-row SELECT with auto fields and queue it for insertion."""
# #         # Build JSON labels via struct -> json
# #         labels_json = ibis.struct({k: ibis.literal(v) for k, v in labels.items()}).cast("json")

# #         # 1行のリレーションを安定に作る（既存テーブルから集約して 1 行に）
# #         one = self.ledger.table.aggregate(dummy=ibis.count())

# #         row_expr = one.select(
# #             id=ibis.literal(str(uuid.uuid4())),                 # backend依存のuuid()は避ける
# #             ts=ibis.now(),                                      # DBサイドの現在時刻
# #             pkg_version=ibis.literal(self.ledger.pkg_version),
# #             kind=ibis.literal(kind),
# #             labels=labels_json,
# #             payload=(payload_expr if payload_expr.type().is_json()
# #                      else payload_expr.cast(dt.json)),
# #         )

# #         self.insert_from_select(row_expr)
# import uuid
# from ibis import _  # 1-row SELECT を組み立てるためのプレースホルダ
# class LedgerSession:
#     """Queue INSERT ... SELECT jobs and run them in order on context exit (Ibis-native)."""

#     def __init__(self, ledger: "Ledger"):
#         self.ledger = ledger                 # has: .con (backend), .table (ibis Table), .pkg_version (str)
#         self._jobs: list = []

#     def __enter__(self) -> "LedgerSession":
#         return self

#     def __exit__(self, exc_type, exc, tb) -> None:
#         if exc_type is not None:
#             return  # do not execute queued jobs on error
#         for job in self._jobs:
#             job()

#     # def insert_from_select(self, select_expr):
#     #     """Queue: INSERT INTO ledger SELECT ..."""
#     #     def _exec():
#     #         # Ibis v10+: backend.insert(table, obj=<table expr>)
#     #         self.ledger.con.insert(self.ledger.table, [(select_expr.execute(),)])
#     #     self._jobs.append(_exec)

#     # def insert(self, *, kind: str, labels: dict, payload_expr):
#     #     labels_json = ibis.struct({k: ibis.literal(v) for k, v in labels.items()}).cast("json")

#     #     # 保険: 列型ならスカラー化（既にスカラーならそのまま通るバックエンドが多い）
#     #     try:
#     #         payload_scalar = payload_expr.scalar_subquery()
#     #     except Exception:
#     #         payload_scalar = payload_expr

#     #     row_expr = self.ledger.table.select(
#     #         payload=payload_expr, # payload=(payload_scalar if payload_scalar.type().is_json() else payload_scalar.cast("json")),
#     #         id=ibis.literal(str(uuid.uuid4())),
#     #         ts=ibis.now(),
#     #         pkg_version=ibis.literal(__version__),
#     #         kind=ibis.literal(kind),
#     #         labels=labels_json,
#     #     )
#     #     self.insert_from_select(row_expr)

#     def insert_from_select(self, select_expr):
#         # ❌ select_expr.insert() ではない
#         # print("compiled:", self.ledger.con.compile(select_expr))
#         def _exec():
#             self.ledger.con.insert(self.ledger.table, from_expr=select_expr)
#         self._jobs.append(_exec)

#     def insert(self, *, kind: str, labels: dict, payload_expr):
#         labels_struct = ibis.struct({k: ibis.literal(v) for k, v in labels.items()})

#         # 他リレーション由来の式ならここでスカラー化してから使う
#         print("compiled payload:", self.ledger.con.compile(payload_expr))
#         payload_json = payload_expr
#         # payload_json = payload_expr.cast("json")

#         print("exec:", payload_json.execute())
#         row_select = ibis.memtable(dict(
#             uuid=[ibis.uuid()],
#             ts=[ibis.now()],
#             pkg_version=[ibis.literal(__version__)],
#             kind=[ibis.literal(kind)],
#             labels=[labels_struct],
#             payload=[payload_json],
#         ))
#         self.insert_from_select(row_select)
#     # def insert(self, *, kind: str, labels: dict, payload_expr):
#     #     """Build a single-row SELECT with auto fields and queue it for insertion."""
#     #     labels_json = ibis.struct({k: ibis.literal(v) for k, v in labels.items()}).cast("json")

#     #     # single-row relation via aggregation (works even when the table is empty)
#     #     one = self.ledger.table.aggregate(dummy=self.ledger.table.count())
#     #     row_expr = labels_json
#     #     # row_expr = ibis.struct(
#     #     #     id=ibis.literal(str(uuid.uuid4())),
#     #     #     ts=ibis.now(),
#     #     #     pkg_version=ibis.literal(__version__),
#     #     #     kind=ibis.literal(kind),
#     #     #     labels=labels_json,
#     #     #     payload=(payload_expr if payload_expr.type().is_json() else payload_expr.cast("json")),
#     #     # )
#     #     self.insert_from_select(row_expr)

# # ----------------------------------------------------------------------
# # Utilities for "latest record by kind and labels"
# # ----------------------------------------------------------------------
# def latest_by_kind_and_labels(tbl, kind: str, labels: Dict[str, Any]):
#     """Return a single-row table (ORDER BY ts DESC LIMIT 1) restricted by kind & labels."""
#     pred = (tbl.kind == kind)
#     for k, v in labels.items():
#         node = tbl.labels[k]
#         if isinstance(v, bool):
#             pred &= (node.unwrap_as("boolean") == ibis.literal(bool(v)))
#         elif isinstance(v, (int, float)):
#             pred &= (node.unwrap_as("float64") == ibis.literal(float(v)))
#         else:
#             pred &= (node.str == ibis.literal(str(v)))
#     return tbl.filter(pred).order_by(tbl.ts.desc()).limit(1)


# # ----------------------------------------------------------------------
# # A/B Test workflow using JSON-only expressions
# # ----------------------------------------------------------------------
# class ABTestLedger:
#     """Minimal A/B test workflow that writes JSON records and reads the latest ones.

#     Kinds
#     -----
#     - "agg": cumulative counts (nA, mA, nB, mB)
#     - "z":   z-statistic computed from the latest "agg"
#     - "e":   a simple E-process value (illustrative Gaussian e-value)
#     - "decision": text decision based on the latest "z" (e.g., "continue" / "efficacy")

#     Labels
#     ------
#     We fix labels to {"test": "ab"} for the demo.
#     """

#     def __init__(self):
#         self.con = ibis.duckdb.connect()
#         self.ledger = Ledger(self.con)
#         self.labels = {"test": "ab"}

#     # ---------- core builders (pure Ibis exp) ----------
#     def _insert_agg(self, sess: LedgerSession, dnA: int, dmA: int, dnB: int, dmB: int):
#         """Insert an 'agg' record by adding deltas to the latest 'agg'."""
#         tbl = self.ledger.table
#         last = latest_by_kind_and_labels(tbl, "agg", self.labels)

#         prev_nA = last.payload["nA"].unwrap_as("int64").as_scalar()
#         prev_mA = last.payload["mA"].unwrap_as("int64").as_scalar()
#         prev_nB = last.payload["nB"].unwrap_as("int64").as_scalar()
#         prev_mB = last.payload["mB"].unwrap_as("int64").as_scalar()

#         # Use fill_null (v9.1+) instead of deprecated fillna
#         nA = prev_nA.fill_null(0) + ibis.literal(int(dnA))
#         mA = prev_mA.fill_null(0) + ibis.literal(int(dmA))
#         nB = prev_nB.fill_null(0) + ibis.literal(int(dnB))
#         mB = prev_mB.fill_null(0) + ibis.literal(int(dmB))

#         # Build JSON payload via struct + cast('json') (portable across backends)
#         payload = ibis.struct({"nA": nA, "mA": mA, "nB": nB, "mB": mB})
#         sess.insert(kind="agg", labels=self.labels, payload_expr=payload)

#     def _insert_z(self, sess: LedgerSession):
#         """Insert a 'z' record computed from the latest 'agg' (pooled z)."""
#         tbl = self.ledger.table
#         agg = latest_by_kind_and_labels(tbl, "agg", self.labels)

#         nA = agg.payload["nA"].unwrap_as("int64")
#         mA = agg.payload["mA"].unwrap_as("int64")
#         nB = agg.payload["nB"].unwrap_as("int64")
#         mB = agg.payload["mB"].unwrap_as("int64")

#         eps = ibis.literal(1e-9, type="float64")

#         pA = (mA.cast("float64") / (nA.cast("float64") + eps))
#         pB = (mB.cast("float64") / (nB.cast("float64") + eps))
#         nAf = nA.cast("float64")
#         nBf = nB.cast("float64")
#         p_pool = ((mA + mB).cast("float64") / ((nA + nB).cast("float64") + eps))
#         denom = (p_pool * (1 - p_pool) * (1 / (nAf + eps) + 1 / (nBf + eps)) + eps).sqrt()
#         z = ((pB - pA) / denom)

#         payload = ibis.struct({"z": z})
#         sess.insert(kind="z", labels=self.labels, payload_expr=payload)

#     def _insert_e(self, sess: LedgerSession):
#         """Insert an 'e' record from the latest 'z' using a Gaussian e-value (illustrative).

#         We use e = exp(λ Z - λ^2/2) with λ = clip(Z, -λmax, λmax) / t
#         and t ≈ 1 for a unit-information approximation (illustrative only).
#         This is for demonstration; do NOT use for formal inference as-is.
#         """
#         tbl = self.ledger.table
#         zt = latest_by_kind_and_labels(tbl, "z", self.labels)
#         Z = zt.payload["z"].unwrap_as("float64")

#         lam_max = ibis.literal(3.0)
#         t = ibis.literal(1.0)  # unit information (demo only)
#         lam = (Z / t)
#         lam = lam.clip(-lam_max, lam_max)
#         e = (lam * Z - (lam * lam) / 2.0).exp()

#         payload = ibis.struct({"e": e})
#         sess.insert(kind="e", labels=self.labels, payload_expr=payload)

#     def _insert_decision(self, sess: LedgerSession, z_thr: float = 1.96):
#         """Insert a 'decision' record from the latest 'z' (two-sided threshold)."""
#         tbl = self.ledger.table
#         zt = latest_by_kind_and_labels(tbl, "z", self.labels)
#         Z = zt.payload["z"].unwrap_as("float64")

#         thr = ibis.literal(float(z_thr))
#         decision = ibis.cases((Z.abs() >= thr, "efficacy"), else_="continue")

#         payload = ibis.struct({"decision": decision})
#         sess.insert(kind="decision", labels=self.labels, payload_expr=payload)

#     # ---------- public API ----------
#     def update(self, deltas: Dict[str, int]):
#         """Append one update (agg -> z -> e -> decision) using Ibis-native expressions.

#         Parameters
#         ----------
#         deltas : dict with int values
#             Keys: 'nA','mA','nB','mB'  (deltas to add to cumulative counts)
#         """
#         dnA = int(deltas.get("nA", 0))
#         dmA = int(deltas.get("mA", 0))
#         dnB = int(deltas.get("nB", 0))
#         dmB = int(deltas.get("mB", 0))

#         with self.ledger.session() as sess:
#             # 1) Aggregate (adds the deltas to the latest cumulative counts)
#             self._insert_agg(sess, dnA, dmA, dnB, dmB)
#             # 2) z from the latest agg
#             self._insert_z(sess)
#             # 3) e from the latest z (illustrative)
#             self._insert_e(sess)
#             # 4) decision from the latest z
#             self._insert_decision(sess)


# # ----------------------------------------------------------------------
# # Simple profiling hook to mimic prior runs
# # ----------------------------------------------------------------------
# def _ab_workload():
#     ab = ABTestLedger()
#     ab.update({"nA": 4, "mA": 1, "nB": 4, "mB": 2})
#     ab.update({"nA": 3, "mA": 1, "nB": 3, "mB": 1})
#     ab.update({"nA": 5, "mA": 2, "nB": 5, "mB": 3})
#     return ab


# def _profile_run() -> None:
#     pr = cProfile.Profile()
#     pr.enable()
#     ab = _ab_workload()
#     print(ab.ledger.table.execute())
#     pr.disable()
#     ps = pstats.Stats(pr).sort_stats("cumulative")
#     ps.print_stats(20)


# # ----------------------------------------------------------------------
# # CLI
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     _profile_run()

"""
EarlySign v13 — Ibis-native INSERT queue with memtable anchor (no raw SQL)

Doctest (smoke test)
--------------------
>>> import ibis
>>> con = ibis.duckdb.connect()
>>> # Prepare ledger
>>> from earlysign.earlysign_dsl_design_13 import Ledger, ABTest
>>> ledger = Ledger(con, "ledger_v13_doctest", overwrite=True)
>>> ab = ABTest(ledger, labels={"test": "ab"})
>>> # 3 small updates
>>> ab.update({"nA": 4, "mA": 1, "nB": 4, "mB": 2})
>>> ab.update({"nA": 2, "mA": 1, "nB": 2, "mB": 1})
>>> ab.update({"nA": 4, "mA": 1, "nB": 4, "mB": 3})
>>> df = con.execute(ledger.table)
>>> set(df["kind"]) >= {"agg","z","e","decision"}
True
"""

from __future__ import annotations

import cProfile
import pstats
from typing import Any, Callable, Dict, List

import ibis
from ibis.expr.types import Table as IbisTable

__version__ = "13.0.0"


# ---------- Utilities ----------


def json_get_str(j, key: str):
    """Return JSON string as string dtype using unwrap_as."""
    return j[key].unwrap_as("string")


def json_get_i64(j, key: str):
    """Return JSON integer as int64 dtype using unwrap_as."""
    return j[key].unwrap_as("int64")


def json_get_f64(j, key: str):
    """Return JSON number as float64 dtype using unwrap_as."""
    return j[key].unwrap_as("float64")


def labels_pred(labels_col, labels: Dict[str, Any]):
    """Build conjunction of JSON label equalities with unwrap."""
    pred = ibis.literal(True)
    for k, v in labels.items():
        pred &= labels_col[k].unwrap_as("string") == ibis.literal(str(v))
    return pred


# ---------- Ledger ----------


class Ledger:
    """A minimal event ledger backed by an Ibis SQL backend.

    Table schema:
      - id: uuid
      - ts: timestamp
      - pkg_version: string
      - kind: string
      - labels: json
      - payload: json
    """

    def __init__(
        self, con: ibis.backends.BaseBackend, table_name: str, overwrite: bool = True
    ):
        self.con = con
        self.table_name = table_name
        # Create or replace ledger table
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
        if table_name not in self.con.list_tables():
            self.con.create_table(self.table_name, schema=schema)

    @property
    def table(self) -> IbisTable:
        return self.con.table(self.table_name)

    def session(self) -> "LedgerSession":
        return LedgerSession(self)


class LedgerSession:
    """Queue `INSERT` jobs and flush them on context exit.

    Notes
    -----
    - No raw SQL is built; we always use `con.insert(table_name: str, obj=df)`.
    - Each job is a closure that executes a SELECT to a pandas DataFrame,
      then inserts it into the ledger. Subqueries are computed in the backend.
    """

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self._jobs: List[Callable[[], None]] = []

    def __enter__(self) -> "LedgerSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            return  # do not flush on error
        for job in self._jobs:
            job()

    def _enqueue_insert(self, select_expr):
        """Enqueue: execute the one-row SELECT in backend, then insert its df."""
        con = self.ledger.con
        table_name = self.ledger.table_name

        def _exec():
            con.insert(table_name, select_expr)

        self._jobs.append(_exec)

    def insert(self, *, kind: str, labels: Dict[str, Any], payload_expr):
        """Build a one-row SELECT using a memtable anchor and enqueue insertion.

        All cross-table references within `payload_expr` must be *scalar* expressions
        (e.g., reductions like max(), or LIMIT 1 then aggregated), so that they can be
        safely projected off a 1-row anchor without integrity violations.
        """
        row_select = ibis.memtable([{"_dummy": 1}]).select(
            id=ibis.uuid(),
            ts=ibis.now(),
            pkg_version=ibis.literal(__version__),
            kind=ibis.literal(kind),
            labels=ibis.struct({k: ibis.literal(v) for k, v in labels.items()}).cast(
                "json"
            ),
            payload=payload_expr.cast("json"),
        )
        self._enqueue_insert(row_select)


# ---------- AB test workflow (agg -> z -> e -> decision) ----------


class ABTest:
    """A/B test pipeline that writes intermediate records and reuses them later.

    For labels L = {"test": "ab"} the following record kinds are written:
      - "agg": cumulative counts {nA, mA, nB, mB}
      - "z"  : test statistic {z}
      - "e"  : e-process value {e}
      - "decision": current decision {"decision": "efficacy" | "continue"}
    """

    def __init__(self, ledger: Ledger, labels: Dict[str, Any]):
        self.ledger = ledger
        self.labels = labels

    # ---- helpers to access latest payload fields as scalar subqueries ----

    def _latest_row(self, kind: str):
        T = self.ledger.table
        return (
            T.filter(T.kind == kind)
            .filter(labels_pred(T.labels, self.labels))
            .order_by(T.ts.desc())
            .limit(1)
        )

    def _latest_payload_i64(self, kind: str, field: str, default: int = 0):
        col = json_get_i64(self._latest_row(kind).payload, field)
        # Reduce to scalar; if relation is empty, result is NULL, then coalesce
        return ibis.coalesce(col.max(), ibis.literal(default, type="int64"))

    def _latest_payload_f64(self, kind: str, field: str, default: float = 0.0):
        col = json_get_f64(self._latest_row(kind).payload, field)
        return ibis.coalesce(col.max(), ibis.literal(default, type="float64"))

    # ---- inserters (each enqueues one job) ----

    def _insert_agg(self, sess: LedgerSession, dnA: int, dmA: int, dnB: int, dmB: int):
        # Previous cumulative counts (scalar subqueries, default 0)
        prev_nA = self._latest_payload_i64("agg", "nA", 0)
        prev_mA = self._latest_payload_i64("agg", "mA", 0)
        prev_nB = self._latest_payload_i64("agg", "nB", 0)
        prev_mB = self._latest_payload_i64("agg", "mB", 0)

        nA = prev_nA + ibis.literal(int(dnA))
        mA = prev_mA + ibis.literal(int(dmA))
        nB = prev_nB + ibis.literal(int(dnB))
        mB = prev_mB + ibis.literal(int(dmB))

        payload = ibis.struct(dict(nA=nA, mA=mA, nB=nB, mB=mB))
        sess.insert(kind="agg", labels=self.labels, payload_expr=payload)

    def _insert_z(self, sess: LedgerSession):
        # Read cumulative counts from latest "agg"
        nA = self._latest_payload_f64("agg", "nA", 0.0)
        mA = self._latest_payload_f64("agg", "mA", 0.0)
        nB = self._latest_payload_f64("agg", "nB", 0.0)
        mB = self._latest_payload_f64("agg", "mB", 0.0)

        eps = ibis.literal(1e-9)
        pA = mA / (nA + eps)
        pB = mB / (nB + eps)

        # Pooled variance for difference in proportions
        m = mA + mB
        n = nA + nB
        p_pool = m / (n + eps)
        var = p_pool * (1 - p_pool) * (1 / (nA + eps) + 1 / (nB + eps)) + eps
        Z = (pB - pA) / var.sqrt()

        payload = ibis.struct(dict(z=Z))
        sess.insert(kind="z", labels=self.labels, payload_expr=payload)

    def _insert_e(self, sess: LedgerSession):
        # e-process via Gaussian N(0,1) likelihood ratio with clipped lambda
        z = self._latest_payload_f64("z", "z", 0.0)
        lam_raw = z / 1.0
        lam = lam_raw.clip(lower=-3.0, upper=3.0)  # avoid explosion
        e = (lam * z - (lam * lam) / 2.0).exp()

        payload = ibis.struct(dict(e=e))
        sess.insert(kind="e", labels=self.labels, payload_expr=payload)

    def _insert_decision(self, sess: LedgerSession, thr: float = 1.96):
        z = self._latest_payload_f64("z", "z", 0.0)
        decision = ibis.cases(
            (z.abs() >= ibis.literal(thr), "efficacy"), else_="continue"
        )
        payload = ibis.struct(dict(decision=decision))
        sess.insert(kind="decision", labels=self.labels, payload_expr=payload)

    # ---- public API ----

    def update(self, delta: Dict[str, int]):
        """Enqueue all inserts inside one session; flush on context exit.

        Parameters
        ----------
        delta: dict
            {"nA": d, "mA": d, "nB": d, "mB": d} increments for this batch.
        """
        dnA = int(delta.get("nA", 0))
        dmA = int(delta.get("mA", 0))
        dnB = int(delta.get("nB", 0))
        dmB = int(delta.get("mB", 0))

        with self.ledger.session() as sess:
            self._insert_agg(sess, dnA, dmA, dnB, dmB)
            self._insert_z(sess)
            # self._insert_e(sess)
            self._insert_decision(sess)


# ---------- Demo / profiling ----------


def _ab_workload() -> ABTest:
    con = ibis.duckdb.connect()
    ledger = Ledger(con, "ledger", overwrite=True)
    ab = ABTest(ledger, labels={"test": "ab"})

    # Three light updates
    ab.update({"nA": 4, "mA": 1, "nB": 4, "mB": 2})
    ab.update({"nA": 2, "mA": 1, "nB": 2, "mB": 1})
    ab.update({"nA": 4, "mA": 1, "nB": 4, "mB": 3})

    # Show resulting rows
    print(ibis.to_sql(ledger.table))
    print(ledger.con.execute(ledger.table))
    return ab


def _profile_run():
    pr = cProfile.Profile()
    pr.enable()
    _ab_workload()
    pr.disable()
    pstats.Stats(pr).strip_dirs().sort_stats("cumulative").print_stats(20)


if __name__ == "__main__":
    _profile_run()
