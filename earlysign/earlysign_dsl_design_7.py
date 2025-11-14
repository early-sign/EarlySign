"""
Framework + Apps: Reader-base refactor; snapshot→info→(stat,decision); inline SQL
+ cProfile main-run profiling (NOT in doctest)
==================================================================================

Overview
--------
- Ledger: bindable append-only JSON ledger (uuid/ts/pkg_version auto).
- LedgerSession: transaction-local staging; batched INSERT on flush.
- Reader hierarchy (LedgerReaderBase):
    * ABDesignLatest   : latest design (planned_max_n, ts)
    * ABDesignLooks    : explode looks at latest design ts -> (look, planned_t)
    * ABSnapshotLatest : latest snapshot (or a single zero row if none)
    * ABObsSince       : sum of observations strictly after given ts
    * ABRowsBase       : typed AB rows (for decisions/stat/inspection)
    * EBaseRows        : typed e_* rows
    * EThetas          : explode theta grid from e_design
- Apps:
    * BinomialABTest: two-proportions GST with a single 'design' record.
      - ALWAYS: snapshot → info
      - IF due (min i with I0 < t_i ≤ I1): stat & decision
    * ENormalMixture: normal-mixture E-process (uniform over given thetas)

Table contract (must exist before running doctest)
--------------------------------------------------
uuid TEXT, ts TIMESTAMP, pkg_version TEXT, payload_type TEXT, payload JSON, labels JSON

Doctest (DuckDB)
----------------
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

# (A) AB test with single design record (4 looks).
#     Updates crafted so: look=1 -> continue, look=2 -> continue, look=3 -> STOP.
>>> ab = BinomialABTest(ab_ledger)
>>> ab.set_design(max_n=1000, looks=[0.25, 0.5, 0.75, 1.0])
>>> # Update-1: I goes to 0.20 (< 0.25), no look
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> # Update-2: I goes to 0.30, triggers look=1 -> continue
>>> ab.update({"nA": 50, "mA": 5, "nB": 50, "mB": 6})
>>> # Update-3: I goes to 0.60, triggers look=2 -> continue
>>> ab.update({"nA": 150, "mA": 10, "nB": 150, "mB": 20})
>>> # Update-4: I goes to 0.80, triggers look=3 -> STOP
>>> ab.update({"nA": 100, "mA": 5, "nB": 100, "mB": 35})
>>>
>>> # Decisions check
>>> V_base = ABRowsBase(ab_ledger).typed_view()
>>> decisions = V_base.filter(lambda r: r.payload_type == "decision").order_by("ts")
>>> df_dec = con.execute(decisions.select(decisions.look.name("look"),
...                                       decisions.planned_t.name("planned_t"),
...                                       decisions.action.name("action")))
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
>>> # Inspect ledger content (example output; doctest uses ellipsis)
>>> ab_ledger.show().execute()  # doctest: +ELLIPSIS
       payload_type                                            payload                        labels
    0        design  {'planned_max_n': 1000, 'looks': [0.25, 0.5, 0...  {'experiment_id': 'exp_ab4'}
    1   observation         {'nA': 100, 'mA': 10, 'nB': 100, 'mB': 12}  {'experiment_id': 'exp_ab4'}
    2      snapshot  {'nA': 100.0, 'mA': 10.0, 'nB': 100.0, 'mB': 1...  {'experiment_id': 'exp_ab4'}
    3          info                                 {'info_time': 0.2}  {'experiment_id': 'exp_ab4'}
    4   observation             {'nA': 50, 'mA': 5, 'nB': 50, 'mB': 6}  {'experiment_id': 'exp_ab4'}
    5      snapshot  {'nA': 150.0, 'mA': 15.0, 'nB': 150.0, 'mB': 1...  {'experiment_id': 'exp_ab4'}
    6          info                                 {'info_time': 0.3}  {'experiment_id': 'exp_ab4'}
    7          stat                          {'z': 0.5535658388085484}  {'experiment_id': 'exp_ab4'}
    8      decision  {'look': 1, 'planned_t': 0.25, 'info_time': 0....  {'experiment_id': 'exp_ab4'}
    9   observation         {'nA': 150, 'mA': 10, 'nB': 150, 'mB': 20}  {'experiment_id': 'exp_ab4'}
    10     snapshot  {'nA': 300.0, 'mA': 25.0, 'nB': 300.0, 'mB': 3...  {'experiment_id': 'exp_ab4'}
    11         info                                 {'info_time': 0.6}  {'experiment_id': 'exp_ab4'}
    12         stat                          {'z': 1.7312570698610248}  {'experiment_id': 'exp_ab4'}
    13     decision  {'look': 2, 'planned_t': 0.5, 'info_time': 0.6...  {'experiment_id': 'exp_ab4'}
    14  observation          {'nA': 100, 'mA': 5, 'nB': 100, 'mB': 35}  {'experiment_id': 'exp_ab4'}
    15     snapshot  {'nA': 400.0, 'mA': 30.0, 'nB': 400.0, 'mB': 7...  {'experiment_id': 'exp_ab4'}
    16         info                                 {'info_time': 0.8}  {'experiment_id': 'exp_ab4'}
    17         stat                          {'z': 4.5391908987315395}  {'experiment_id': 'exp_ab4'}
    18     decision  {'look': 3, 'planned_t': 0.75, 'info_time': 0....  {'experiment_id': 'exp_ab4'}

# (B) E-process with single design record
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
>>> (round(rowE["e_value"], 2), bool(rowE["alarm"]))
(26.84, True)
"""

from __future__ import annotations

import uuid as _py_uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import ibis

# ===================================
# Common JSON / labels utilities
# ===================================

_PKG_VERSION = "earlysign==dev"


def _json_from_kv(
    items: Dict[str, ibis.Expr], *, raw_keys: set[str] | None = None
) -> ibis.Expr:
    """Build a JSON object purely from Ibis expressions (no Python dumps)."""
    raw_keys = raw_keys or set()
    if not items:
        return ibis.literal("{}").cast("json")
    parts: List[ibis.Expr] = []
    for k, v in items.items():
        key = ibis.literal(f'"{k}":')
        if k in raw_keys:
            val = v.cast("string")
        else:
            val = (
                (ibis.literal('"') + v.cast("string") + ibis.literal('"'))
                if v.type().is_string()
                else v.cast("string")
            )
        parts.append(key + val)
    out = parts[0]
    for p in parts[1:]:
        out = out + ibis.literal(",") + p
    return (ibis.literal("{") + out + ibis.literal("}")).cast("json")


# =========================
# Ledger (bindable facade)
# =========================


@dataclass(frozen=True)
class Ledger:
    """Append-only JSON ledger with bindable labels (scope)."""

    con: Any
    table: str = "ledger"
    labels: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def bind(self, **labels: Union[str, int, float, bool]) -> "Ledger":
        merged = dict(self.labels)
        merged.update(labels)
        return Ledger(self.con, self.table, merged)

    def view_json(self) -> ibis.Expr:
        """Return scoped raw view with (uuid, ts, pkg_version, payload_type, payload, labels)."""
        where_labels = LedgerReaderBase._labels_where_sql(self.labels)
        sql = f"""
            SELECT
              uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
            FROM {self.table}
            WHERE {where_labels}
        """
        return self.con.sql(sql)

    def show(self) -> ibis.Expr:
        """Readable doctest view: order by ts, drop auto columns. Execute outside."""
        t = self.view_json().order_by("ts")
        try:
            t = t.drop("uuid", "ts", "pkg_version")
        except Exception:
            pass
        return t


# =======================================
# Per-row UUID/TS auto fillers (portable)
# =======================================


def _has_ibis_uuid() -> bool:
    return hasattr(ibis, "uuid")


def _uuid_single() -> ibis.Expr:
    """Return a single-row UUID expression (prefer ibis.uuid())."""
    if _has_ibis_uuid():
        return ibis.uuid().cast("string")
    return ibis.literal(_py_uuid.uuid4().hex)


def _uuid_multi(src: ibis.Expr) -> ibis.Expr:
    """
    Return a multi-row UUID expression based on a scalar base + row_number suffix.
    This stays backend-agnostic and deterministic per `src` materialization.
    """
    base = (
        ibis.uuid().cast("string")
        if _has_ibis_uuid()
        else ibis.literal(_py_uuid.uuid4().hex)
    )
    first_col = src[list(src.schema().names)[0]]
    w = ibis.window(order_by=[first_col])
    rn = ibis.row_number().over(w)
    rn_str = rn.cast("int64").cast("string").lpad(12, "0")
    return (base + ibis.literal("-")) + rn_str


# ======================================
# Transaction-local staging session DSL
# ======================================


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
    """Labels-bound staging session; INSERT happens only at flush/commit."""

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

    def _labels_json_expr(
        self, extra: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> ibis.Expr:
        merged = dict(self.ledger.labels)
        if extra:
            merged.update(extra)
        parts = {k: ibis.literal(str(v)) for k, v in merged.items()}
        return _json_from_kv(parts)

    def write(
        self,
        payload_type: str,
        payload: Dict[str, Union[int, float, str, ibis.Expr]],
        *,
        extra_labels: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        raw_keys: set[str] | None = None,
    ) -> None:
        """Stage one row built from Python scalars / Ibis Expr values."""
        anchor = ibis.memtable([{"one": 1}])
        items = {
            k: (v if isinstance(v, ibis.Expr) else ibis.literal(v))
            for k, v in payload.items()
        }
        row = anchor.select(
            uuid=_uuid_single(),
            ts=ibis.now().cast("timestamp(6)"),
            pkg_version=ibis.literal(_PKG_VERSION),
            payload_type=ibis.literal(payload_type),
            payload=_json_from_kv(items, raw_keys=raw_keys or set()),
            labels=self._labels_json_expr(extra_labels),
        )
        self.staged.add(row)

    def write_from(
        self,
        src: ibis.Expr,
        payload_type: str,
        payload: Dict[str, Union[int, float, str, ibis.Expr]],
        *,
        extra_labels: Optional[Dict[str, Union[str, int, float, bool]]] = None,
        raw_keys: set[str] | None = None,
    ) -> None:
        """Stage 0..N rows derived from an Ibis source."""
        items = {
            k: (v if isinstance(v, ibis.Expr) else ibis.literal(v))
            for k, v in payload.items()
        }
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


# =================================
# Reader base + AB/E readers
# =================================


@dataclass(frozen=True)
class LedgerReaderBase:
    """Base for typed/projected readers bound to a Ledger scope."""

    ledger: Ledger

    @staticmethod
    def _labels_where_sql(labels: Dict[str, Union[str, int, float, bool]]) -> str:
        if not labels:
            return "TRUE"
        conds: List[str] = []
        for k, v in labels.items():
            sv = str(v).replace("'", "''")  # escape for SQL
            conds.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
        return " AND ".join(conds)


class ABRowsBase(LedgerReaderBase):
    """Typed view for AB payloads (reused by multiple small readers)."""

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
    """Latest design (planned_max_n, ts)."""

    def latest(self) -> ibis.Expr:
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
SELECT
  b.ts::TIMESTAMP(6) AS ts,
  TRY_CAST(json_extract(b.payload, '$.planned_max_n') AS DOUBLE) AS planned_max_n
FROM {self.ledger.table} b
WHERE {where_labels} AND b.payload_type = 'design'
ORDER BY ts DESC
LIMIT 1
"""
        return self.ledger.con.sql(sql)


class ABDesignLooks(LedgerReaderBase):
    """Explode 'looks' from latest design ts -> (look, planned_t)."""

    def looks(self) -> ibis.Expr:
        D = ABDesignLatest(self.ledger).latest()
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
WITH latest AS (
  SELECT ts FROM ({D.compile()})
)
SELECT
  b.ts::TIMESTAMP(6)             AS ts,
  TRY_CAST(je.key AS BIGINT) + 1 AS look,
  TRY_CAST(je.value AS DOUBLE)   AS planned_t
FROM {self.ledger.table} b,
     LATERAL json_each(b.payload, '$.looks') AS je
WHERE {where_labels} AND b.payload_type = 'design'
  AND b.ts = (SELECT ts FROM latest)
"""
        return self.ledger.con.sql(sql)


class ABSnapshotLatest(LedgerReaderBase):
    """Latest snapshot row; if none, return a single 0-initialized row with epoch ts."""

    def latest(self) -> ibis.Expr:
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
WITH s AS (
  SELECT
    b.ts::TIMESTAMP(6) AS ts,
    TRY_CAST(json_extract(b.payload, '$.nA') AS DOUBLE) AS nA,
    TRY_CAST(json_extract(b.payload, '$.mA') AS DOUBLE) AS mA,
    TRY_CAST(json_extract(b.payload, '$.nB') AS DOUBLE) AS nB,
    TRY_CAST(json_extract(b.payload, '$.mB') AS DOUBLE) AS mB
  FROM {self.ledger.table} b
  WHERE {where_labels} AND b.payload_type = 'snapshot'
  ORDER BY ts DESC
  LIMIT 1
)
SELECT * FROM s
UNION ALL
SELECT
  TIMESTAMP '1970-01-01 00:00:00' AS ts, 0.0 AS nA, 0.0 AS mA, 0.0 AS nB, 0.0 AS mB
WHERE NOT EXISTS (SELECT 1 FROM s)
"""
        return self.ledger.con.sql(sql)


class ABObsSince(LedgerReaderBase):
    """Aggregate observations strictly after given ts0."""

    def sum_after(self, ts0: ibis.Expr) -> ibis.Expr:
        where_labels = self._labels_where_sql(self.ledger.labels)
        sql = f"""
WITH base AS (
  SELECT uuid, ts::TIMESTAMP(6) AS ts, payload
  FROM {self.ledger.table}
  WHERE {where_labels} AND payload_type = 'observation'
),
t0 AS ({ts0.compile()})
SELECT
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.nA') AS DOUBLE)), 0.0) AS d_nA,
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.mA') AS DOUBLE)), 0.0) AS d_mA,
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.nB') AS DOUBLE)), 0.0) AS d_nB,
  COALESCE(SUM(TRY_CAST(json_extract(payload, '$.mB') AS DOUBLE)), 0.0) AS d_mB
FROM base, t0
WHERE base.ts > t0.ts
"""
        return self.ledger.con.sql(sql)


# ==========================
# AB application (two-props)
# ==========================


class BinomialABTest:
    """Two-proportions group-seq test (single 'design' row)."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, max_n: int, looks: Sequence[float]) -> None:
        """Write one 'design' record containing {"planned_max_n": int, "looks": [floats]}."""
        looks_text = "[" + ",".join(str(float(t)) for t in looks) + "]"
        looks_json = ibis.literal(looks_text)
        with LedgerSession(self.ledger) as s:
            s.write(
                "design",
                {"planned_max_n": int(max_n), "looks": looks_json},
                raw_keys={"looks"},
            )

    def update(self, payload: Dict[str, int]) -> None:
        """
        Record observation delta; ALWAYS write snapshot and info first; then if a *new* look is due,
        write z/boundary/decision for the smallest such look i with I0 < t_i <= I1.

        I0 = (nA_before + nB_before) / Nmax
        I1 = (nA_before + nB_before + nA_add + nB_add) / Nmax
        where "before" = latest snapshot + obs since that snapshot (excluding current delta).
        """
        nA_add = float(payload.get("nA", 0))
        mA_add = float(payload.get("mA", 0))
        nB_add = float(payload.get("nB", 0))
        mB_add = float(payload.get("mB", 0))

        with LedgerSession(self.ledger) as s:
            # (0) observation (delta)
            s.write("observation", payload)

            # Readers bound to the same ledger scope
            D_latest = ABDesignLatest(self.ledger).latest()
            Looks = ABDesignLooks(self.ledger).looks()
            S_prev = ABSnapshotLatest(self.ledger).latest()
            Obs_since = ABObsSince(self.ledger).sum_after(S_prev.select("ts"))

            # Nmax from latest design
            Dmax = D_latest.select(Nmax=D_latest.planned_max_n)

            # "before" cumulative (previous snapshot + obs since snapshot)
            before = S_prev.cross_join(Obs_since).select(
                nA_before=S_prev.nA + Obs_since.d_nA,
                mA_before=S_prev.mA + Obs_since.d_mA,
                nB_before=S_prev.nB + Obs_since.d_nB,
                mB_before=S_prev.mB + Obs_since.d_mB,
            )

            # Add current delta to get "now"
            add_tbl = ibis.memtable(
                [
                    {
                        "nA_add": nA_add,
                        "mA_add": mA_add,
                        "nB_add": nB_add,
                        "mB_add": mB_add,
                    }
                ]
            )
            now = (
                before.cross_join(add_tbl)
                .cross_join(Dmax)
                .select(
                    nA=before.nA_before + add_tbl.nA_add,
                    mA=before.mA_before + add_tbl.mA_add,
                    nB=before.nB_before + add_tbl.nB_add,
                    mB=before.mB_before + add_tbl.mB_add,
                    Nmax=Dmax.Nmax,
                    n_before=before.nA_before + before.nB_before,
                )
            )

            # (1) snapshot ALWAYS
            s.write_from(
                now,
                "snapshot",
                {"nA": now.nA, "mA": now.mA, "nB": now.nB, "mB": now.mB},
            )

            # (2) info ALWAYS (I1)
            I1 = (now.nA + now.nB) / now.Nmax.nullif(0)
            s.write_from(now, "info", {"info_time": I1})

            # (3) due detection using I0 < t_i ≤ I1 (I0 excludes current delta)
            I0_tbl = now.select(I0=(now.n_before / now.Nmax.nullif(0)))
            due_candidates = (
                Looks.cross_join(I0_tbl)
                .cross_join(now.select(I1=I1))
                .select(look=Looks.look, planned_t=Looks.planned_t, I0=I0_tbl.I0, I1=I1)
                .filter(lambda r: (r.I0 < r.planned_t) & (r.planned_t <= r.I1))
            )

            min_due = due_candidates.aggregate(
                min_planned_t=due_candidates.planned_t.min()
            )
            due = due_candidates.join(
                min_due, predicates=[due_candidates.planned_t == min_due.min_planned_t]
            ).limit(1)

            # (4) if due, compute Z/boundary/decision at I1
            p_all = (now.mA + now.mB) / (now.nA + now.nB)
            se_all = (p_all * (1 - p_all) * (1 / now.nA + 1 / now.nB)).sqrt().nullif(0)
            z_all = ((now.mB / now.nB) - (now.mA / now.nA)) / se_all

            Z_join = now.cross_join(due).select(
                look=due.look, planned_t=due.planned_t, info_time=I1, z=z_all
            )

            bnd = ibis.cases(
                (Z_join.info_time <= 0.25, ibis.literal(3.5)),
                (Z_join.info_time <= 0.5, ibis.literal(2.963)),
                (Z_join.info_time >= 1.0, ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                + (ibis.literal(1.96) - ibis.literal(2.963))
                * (Z_join.info_time - 0.5)
                / 0.5,
            )
            action = (Z_join.z.abs() >= bnd).ifelse("stop_efficacy", "continue")

            s.write_from(Z_join, "stat", {"z": Z_join.z})
            s.write_from(
                Z_join,
                "decision",
                {
                    "look": Z_join.look,
                    "planned_t": Z_join.planned_t,
                    "info_time": Z_join.info_time,
                    "z": Z_join.z,
                    "boundary": bnd,
                    "action": action,
                },
            )


# =========================
# E readers (inline SQL)
# =========================


class EBaseRows(LedgerReaderBase):
    """Project e_design/e_obs/e_state with only needed typed columns."""

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
    """Explode theta grid from the single 'e_design' row -> (ts, theta)."""

    def grid(self) -> ibis.Expr:
        where_labels = self._labels_where_sql(self.ledger.labels)
        # latest e_design ts
        sql_latest = f"""
SELECT b.ts::TIMESTAMP(6) AS ts
FROM {self.ledger.table} b
WHERE {where_labels} AND b.payload_type = 'e_design'
ORDER BY ts DESC
LIMIT 1
"""
        latest = self.ledger.con.sql(sql_latest)
        sql = f"""
WITH latest AS ({latest.compile()})
SELECT
  b.ts::TIMESTAMP(6)           AS ts,
  TRY_CAST(je.value AS DOUBLE) AS theta
FROM {self.ledger.table} b,
     LATERAL json_each(b.payload, '$.thetas') AS je
WHERE {where_labels} AND b.payload_type = 'e_design'
  AND b.ts = (SELECT ts FROM latest)
"""
        return self.ledger.con.sql(sql)


# ==========================
# E-process application impl
# ==========================


class ENormalMixture:
    """Normal-mixture E-process (uniform over given thetas; single 'e_design' row)."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        """Write one 'e_design' record with {"alpha": float, "thetas": [floats]}."""
        thetas_text = "[" + ",".join(str(float(t)) for t in thetas) + "]"
        thetas_json = ibis.literal(thetas_text)
        with LedgerSession(self.ledger) as s:
            s.write(
                "e_design",
                {"alpha": float(alpha), "thetas": thetas_json},
                raw_keys={"thetas"},
            )

    def update(self, *, x: float) -> None:
        """Record x; update running mixture E-value and write e_state with alarm."""
        x_add = float(x)
        with LedgerSession(self.ledger) as s:
            s.write("e_obs", {"x": x_add})

            Vb = EBaseRows(self.ledger).typed_view()
            Theta = EThetas(self.ledger).grid()
            # latest design alpha
            D = Vb.filter(lambda r: r.payload_type == "e_design")
            Dts = D.aggregate(ts_max=D.ts.max())
            Dcur = D.join(Dts, predicates=[D.ts == Dts.ts_max]).select(alpha=D.alpha)

            # K and Obs aggregates before current delta (current delta not yet committed)
            K = Theta.aggregate(k=Theta.theta.count())
            Obs_prev = Vb.filter(lambda r: r.payload_type == "e_obs").select(
                x_prev=Vb.x
            )
            Agg_prev = Obs_prev.aggregate(
                S_prev=Obs_prev.x_prev.sum().fill_null(0.0),
                t_prev=Obs_prev.x_prev.count(),
            )

            add_tbl = ibis.memtable([{"x_add": x_add, "one": 1}])
            Agg = Agg_prev.cross_join(add_tbl).select(
                S=Agg_prev.S_prev + add_tbl.x_add,
                t=Agg_prev.t_prev + add_tbl.one,
            )

            parts = (
                Theta.cross_join(Agg)
                .cross_join(K)
                .select(
                    term=(
                        (ibis.literal(1.0) / K.k.nullif(0))
                        * (
                            (Theta.theta * Agg.S)
                            - (0.5 * Theta.theta * Theta.theta * Agg.t)
                        ).exp()
                    )
                )
            )
            E = parts.aggregate(e_value=parts.term.sum())

            Thr = Dcur.select(thr=(ibis.literal(1.0) / Dcur.alpha.nullif(0)))
            S_all = E.cross_join(Thr).select(
                e_value=E.e_value, alarm=(E.e_value >= Thr.thr)
            )

            s.write_from(
                S_all,
                "e_state",
                {"e_value": S_all.e_value, "alarm": S_all.alarm.cast("boolean")},
            )


# ==========================
# Profiling helper (callable)
# ==========================


def run_profile_demo(con: Any) -> str:
    """
    Run a short AB scenario under profiling and return stats as a string.

    - Uses a separate label scope (experiment_id="exp_prof") so it doesn't
      interfere with other runs.
    - Filters stats to functions matching /(insert|raw_sql|flush|write_from|write|uuid|now)/.
    - Sorts by cumulative time.
    """
    import cProfile
    import io
    import pstats

    base = Ledger(con, table="ledger")
    prof_ledger = base.bind(experiment_id="exp_prof")
    ab = BinomialABTest(prof_ledger)

    pr = cProfile.Profile()
    pr.enable()

    # Minimal work: design + two updates that trigger one or two looks
    ab.set_design(max_n=400, looks=[0.5, 1.0])
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})  # I≈0.5
    ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})  # I≈1.0

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("cumtime")
    ps.print_stats(r"(insert|raw_sql|flush|write_from|write|uuid|now)")
    return s.getvalue()


# ==========================
# Script entry: profiling
# ==========================

if __name__ == "__main__":
    # When run as a script: create an in-memory DuckDB, ensure table, run profiling,
    # and print the pstats summary to stdout (previous style).
    con = ibis.connect("duckdb://")
    con.raw_sql(
        """
        CREATE TABLE IF NOT EXISTS ledger (
          uuid         TEXT,
          ts           TIMESTAMP,
          pkg_version  TEXT,
          payload_type TEXT,
          payload      JSON,
          labels       JSON
        );
    """
    )
    out = run_profile_demo(con)
    print("=== cProfile (filtered) ===")
    print("Ordered by: cumulative time")
    print(out)
