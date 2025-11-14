"""
Single-ledger JSON DSL + Sequential Procedures (labels-bound, backend-agnostic UUID/TS)
=======================================================================================

What this file provides
-----------------------
1) A payload-agnostic ledger session with:
   - Table contract: (uuid, ts, pkg_version, payload_type, payload, labels)
   - Auto-filled fields on write:
       * uuid  : single-row -> ibis.uuid(); multi-row -> ibis.uuid() base + row_number()
       * ts    : per-row via ibis.now() (no Python-side timestamps)
       * pkg_version: constant string
   - Labels binding via constructor: all reads/writes are scoped by labels
   - JSON projection is centralized in a backend SQL view (DuckDB json_extract* demo)

2) Applications built on top:
   (A) Two-proportions group-sequential test (design precommitted)
   (B) Normal-mean E-process anomaly detector (theta-grid mixture)

Portability notes
-----------------
- No backend-specific UDF names are hard-coded. Per-row UUID is achieved by
  concatenating a scalar `ibis.uuid()` base with a windowed `row_number()`.
- `ts` uses `ibis.now()` so timestamps are evaluated by the backend per row.
- JSON projection in `_real_flat_view` uses DuckDB SQL for the doctest demo.
  For other backends, implement a sibling `_real_flat_view_*` accordingly.

Ledger table contract
---------------------
Columns:
  - uuid:        STRING  (auto)
  - ts:          TIMESTAMP(UTC-like) (auto via ibis.now())
  - pkg_version: STRING  (e.g. "earlysign==dev")
  - payload_type:STRING  (framework-level type name; replaces "kind")
  - payload:     JSON
  - labels:      JSON    (used for scoping/search; e.g. {"experiment_id":"..."} )

Doctest (DuckDB/Ibis)
---------------------
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
>>> # (A) Group-seq A/B test under labels={"experiment_id":"exp_ab"}
>>> ab = BinomialABTest(con, labels={"experiment_id": "exp_ab"}, table="ledger")
>>> ab.set_design(max_n=1000, looks=[0.5, 1.0])
>>> ab.update({"nA": 100, "mA": 10, "nB": 100, "mB": 12})
>>> # No decision at first update
>>> con.execute(
...   _real_flat_view(con, "ledger", labels={"experiment_id":"exp_ab"})
...     .filter(lambda r: r.payload_type == "decision")
...     .count()
... )
0
>>> # Cross the first planned look at the second update
>>> ab.update({"nA": 150, "mA": 15, "nB": 150, "mB": 18})
>>> qA = (
...   _real_flat_view(con, "ledger", labels={"experiment_id":"exp_ab"})
...     .filter(lambda r: r.payload_type == "decision")
...     .order_by("ts")
...     .limit(1)
... )
>>> dfA = con.execute(qA.select(
...   qA.look.name("look"),
...   qA.planned_t.name("planned_t"),
...   qA.info_time.name("info_time"),
...   qA.action.name("action"),
... ))
>>> rowA = dfA.to_dict("records")[0]
>>> (rowA["look"], round(rowA["planned_t"], 3), round(rowA["info_time"], 3), rowA["action"])  # doctest: +ELLIPSIS
(1, 0.5, 0.5, 'continue')
>>>
>>> # (B) Normal E-process under labels={"experiment_id":"exp_e"}
>>> ep = ENormalMixture(con, labels={"experiment_id": "exp_e"}, table="ledger")
>>> ep.set_design(alpha=0.05, thetas=[0.25, 0.5, 0.75])  # uniform weights 1/K
>>> for _ in range(12):
...     ep.update(x=0.8)
>>> qE = (
...   _real_flat_view(con, "ledger", labels={"experiment_id":"exp_e"})
...     .filter(lambda r: r.payload_type == "e_state")
...     .order_by(lambda r: r.ts.desc())
...     .limit(1)
... )
>>> dfE = con.execute(qE.select(qE.e_value.name("e_value"), qE.alarm.name("alarm")))
>>> rowE = dfE.to_dict("records")[0]
>>> (round(rowE["e_value"], 2), bool(rowE["alarm"]))
(26.84, True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import ibis

# Package version for auto-filled pkg_version.
_PKG_VERSION = "earlysign==dev"


# ============================================================================
# Helpers
# ============================================================================


def _json_from_kv(items: Dict[str, ibis.Expr]) -> ibis.Expr:
    """Build a JSON object from Ibis expressions only (no Python json.dumps)."""
    if not items:
        return ibis.literal("{}").cast("json")
    parts: List[ibis.Expr] = []
    for k, v in items.items():
        key = ibis.literal(f'"{k}":')
        val = (
            ibis.literal('"') + v.cast("string") + ibis.literal('"')
            if v.type().is_string()
            else v.cast("string")
        )
        parts.append(key + val)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined + ibis.literal(",") + p
    return (ibis.literal("{") + combined + ibis.literal("}")).cast("json")


def _labels_where_sql(labels: Dict[str, Union[str, int, float, bool]]) -> str:
    """Return a DuckDB SQL predicate that matches exact stringified labels."""
    if not labels:
        return "TRUE"
    conds = []
    for k, v in labels.items():
        sv = str(v).replace("'", "''")
        conds.append(f"json_extract_string(labels, '$.{k}') = '{sv}'")
    return " AND ".join(conds)


def _real_flat_view(
    con: ibis.Client,
    table: str,
    labels: Dict[str, Union[str, int, float, bool]],
) -> ibis.Expr:
    """
    Project JSON payload into typed columns using DuckDB SQL json_extract*.

    NOTE: For non-DuckDB backends, create a sibling function that uses the
    backend's JSON projection facility and keep the same output columns.
    """
    where_labels = _labels_where_sql(labels)
    sql = f"""
        SELECT
          uuid,
          ts::TIMESTAMP(6) AS ts,
          pkg_version,
          payload_type,
          payload,
          labels,

          -- A/B test fields
          TRY_CAST(json_extract(payload, '$.nA')            AS DOUBLE)   AS nA,
          TRY_CAST(json_extract(payload, '$.mA')            AS DOUBLE)   AS mA,
          TRY_CAST(json_extract(payload, '$.nB')            AS DOUBLE)   AS nB,
          TRY_CAST(json_extract(payload, '$.mB')            AS DOUBLE)   AS mB,
          TRY_CAST(json_extract(payload, '$.planned_max_n') AS DOUBLE)   AS planned_max_n,
          TRY_CAST(json_extract(payload, '$.planned_t')     AS DOUBLE)   AS planned_t,
          TRY_CAST(json_extract(payload, '$.look')          AS BIGINT)   AS look,
          TRY_CAST(json_extract(payload, '$.info_time')     AS DOUBLE)   AS info_time,
          TRY_CAST(json_extract(payload, '$.z')             AS DOUBLE)   AS z,
          TRY_CAST(json_extract(payload, '$.boundary')      AS DOUBLE)   AS boundary,
          json_extract_string(payload, '$.action')                      AS action,

          -- E-process fields
          TRY_CAST(json_extract(payload, '$.x')             AS DOUBLE)   AS x,
          TRY_CAST(json_extract(payload, '$.alpha')         AS DOUBLE)   AS alpha,
          TRY_CAST(json_extract(payload, '$.theta')         AS DOUBLE)   AS theta,
          TRY_CAST(json_extract(payload, '$.e_value')       AS DOUBLE)   AS e_value,
          TRY_CAST(json_extract(payload, '$.alarm')         AS BOOLEAN)  AS alarm

        FROM {table}
        WHERE {where_labels}
    """
    return con.sql(sql)


def _uuid_for_single_row() -> ibis.Expr:
    """Return a literal UUID string for single-row inserts using ibis.uuid()."""
    return ibis.uuid().cast("string")


def _uuid_for_multi_rows(src: ibis.Expr) -> ibis.Expr:
    """Return a per-row unique string using base ibis.uuid() + row_number() bound to `src`."""
    base = ibis.uuid().cast("string")
    # Bind the window to `src` by ordering on its first column (any column anchors the window).
    first_col_name = list(src.schema().names)[0]
    first_col = src[first_col_name]
    w = ibis.window(order_by=[first_col])
    rn = ibis.row_number().over(w)
    rn_str = rn.cast("int64").cast("string").lpad(12, "0")
    return (base + ibis.literal("-")) + rn_str


# ============================================================================
# Transaction-local staging (payload-agnostic; labels-bound)
# ============================================================================


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
    """A lightweight, labels-bound staging session over a ledger table.

    - Table schema: (uuid, ts, pkg_version, payload_type, payload, labels)
    - Auto-fill on write:
        * uuid  -> single-row: ibis.uuid(); multi-row: base-uuid + row_number()
        * ts    -> ibis.now() per row
        * pkg_version -> constant string
    - All reads/writes are implicitly scoped by `labels`
    """

    con: ibis.Client
    table: str
    labels: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    staged: _Staged = field(default_factory=_Staged)

    def __enter__(self):
        self.con.raw_sql("BEGIN;")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.flush()
                self.con.raw_sql("COMMIT;")
            else:
                self.con.raw_sql("ROLLBACK;")
        finally:
            self.staged = _Staged()

    def _labels_json_expr(
        self, extra: Optional[Dict[str, Union[str, int, float, bool]]] = None
    ) -> ibis.Expr:
        """Return labels JSON (bound labels merged with extra)."""
        merged = dict(self.labels)
        if extra:
            merged.update(extra)
        parts = {k: ibis.literal(str(v)) for k, v in merged.items()}
        return _json_from_kv(parts)

    def write(
        self,
        payload_type: str,
        payload: Dict[str, Union[int, float, str]],
        *,
        extra_labels: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    ) -> None:
        """Stage a single JSON row (uuid via ibis.uuid; ts via ibis.now)."""
        anchor = ibis.memtable([{"one": 1}])
        row = anchor.select(
            uuid=_uuid_for_single_row(),
            ts=ibis.now().cast("timestamp(6)"),
            pkg_version=ibis.literal(_PKG_VERSION),
            payload_type=ibis.literal(payload_type),
            payload=_json_from_kv({k: ibis.literal(v) for k, v in payload.items()}),
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
    ) -> None:
        """
        Stage rows produced from an Ibis source (0..N rows).
        - UUIDs are unique per row via (ibis.uuid() base) + row_number() bound to `src`.
        - TS is per-row via ibis.now().
        """
        row = src.select(
            uuid=_uuid_for_multi_rows(src),
            ts=ibis.now().cast("timestamp(6)"),
            pkg_version=ibis.literal(_PKG_VERSION),
            payload_type=ibis.literal(payload_type),
            payload=_json_from_kv(
                {
                    k: (v if isinstance(v, ibis.Expr) else ibis.literal(v))
                    for k, v in payload.items()
                }
            ),
            labels=self._labels_json_expr(extra_labels),
        )
        self.staged.add(row)

    def view_real(self) -> ibis.Expr:
        """Return the real committed flat view scoped by bound labels."""
        return _real_flat_view(self.con, self.table, self.labels)

    def view_json(self) -> ibis.Expr:
        """Return (uuid, ts, pkg_version, payload_type, payload, labels) scoped by labels."""
        where_labels = _labels_where_sql(self.labels)
        sql = f"""
            SELECT
              uuid, ts::TIMESTAMP(6) AS ts, pkg_version, payload_type, payload, labels
            FROM {self.table}
            WHERE {where_labels}
        """
        return self.con.sql(sql)

    def flush(self) -> None:
        union = self.staged.union()
        if union is not None:
            self.con.insert(self.table, union)

    def read(self, fn: Callable[[ibis.Expr], ibis.Expr]) -> ibis.Expr:
        """Convenience for DSL-like reads over the JSON scoped view."""
        return fn(self.view_json())


# ============================================================================
# (A) Two-proportions Group-Sequential Test (labels-bound; payload_type-based)
# ============================================================================


class BinomialABTest:
    """Two-proportions group-sequential test (OF-like demo boundary).

    Contract:
      - Observations: payload {"nA","mA","nB","mB"}, payload_type="observation"
      - Design rows:  payload_type in {"design_max_n","design_look"}
      - Derived rows: payload_type in {"snapshot","stat","info","decision"}
      - All reads/writes are scoped by `labels`
    """

    def __init__(
        self,
        con: ibis.Client,
        labels: Dict[str, Union[str, int, float, bool]],
        table: str = "ledger",
    ):
        self.con, self.labels, self.table = con, dict(labels), table

    def set_design(self, *, max_n: int, looks: Sequence[float]) -> None:
        """Commit design rows immediately under current labels (batch, portable UUID/TS)."""
        with LedgerSession(self.con, self.table, self.labels) as s:
            # Single row
            s.write("design_max_n", {"planned_max_n": int(max_n)})
            # Multi rows: vector-safe via base-uuid + row_number()
            tbl = ibis.memtable(
                [{"look": i + 1, "planned_t": float(t)} for i, t in enumerate(looks)]
            )
            s.write_from(
                tbl,
                "design_look",
                {"look": tbl["look"], "planned_t": tbl["planned_t"]},
            )

    def update(self, payload: Dict[str, int]) -> None:
        """One update: write observation and derive info/stat/decision rows."""
        nA_add = float(payload.get("nA", 0))
        mA_add = float(payload.get("mA", 0))
        nB_add = float(payload.get("nB", 0))
        mB_add = float(payload.get("mB", 0))

        with LedgerSession(self.con, self.table, self.labels) as s:
            # (1) Stage raw observation
            s.write("observation", payload)

            # (2) Read committed design/obs and compute derived stats using expressions
            V = s.view_real()

            design_ts = V.filter(
                lambda r: r.payload_type.isin(("design_max_n", "design_look"))
            ).aggregate(ts_max=V.ts.max())

            Dmax = (
                V.filter(lambda r: r.payload_type == "design_max_n")
                .join(design_ts, predicates=[V.ts == design_ts.ts_max])
                .select(Nmax=V.planned_max_n)
            )

            Looks = (
                V.filter(lambda r: r.payload_type == "design_look")
                .join(design_ts, predicates=[V.ts == design_ts.ts_max])
                .select(look=V.look.cast("int64"), planned_t=V.planned_t)
            )

            Dec = V.filter(lambda r: r.payload_type == "decision")
            next_look_tbl = Dec.aggregate(next_look=(Dec.payload_type.count() + 1))

            planned = Looks.join(
                next_look_tbl, predicates=[Looks.look == next_look_tbl.next_look]
            ).select(look=Looks.look, planned_t=Looks.planned_t)

            Obs_prev = V.filter(lambda r: r.payload_type == "observation")
            agg_prev = Obs_prev.aggregate(
                nA=Obs_prev.nA.sum().fill_null(0.0),
                mA=Obs_prev.mA.sum().fill_null(0.0),
                nB=Obs_prev.nB.sum().fill_null(0.0),
                mB=Obs_prev.mB.sum().fill_null(0.0),
            )

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

            base = (
                agg_prev.cross_join(add_tbl)
                .cross_join(Dmax)
                .cross_join(planned)
                .select(
                    nA=(agg_prev.nA + add_tbl.nA_add),
                    mA=(agg_prev.mA + add_tbl.mA_add),
                    nB=(agg_prev.nB + add_tbl.nB_add),
                    mB=(agg_prev.mB + add_tbl.mB_add),
                    Nmax=Dmax.Nmax,
                    look=planned.look,
                    planned_t=planned.planned_t,
                )
            )

            I0_tbl = agg_prev.cross_join(Dmax).select(
                ((agg_prev.nA + agg_prev.nB) / Dmax.Nmax.nullif(0)).name("I0")
            )
            X = base.cross_join(I0_tbl).select(
                nA=base.nA,
                mA=base.mA,
                nB=base.nB,
                mB=base.mB,
                Nmax=base.Nmax,
                look=base.look,
                planned_t=base.planned_t,
                I0=I0_tbl.I0,
            )

            I_expr = (X.nA + X.nB) / X.Nmax.nullif(0)

            p_expr = (X.mA + X.mB) / (X.nA + X.nB)
            se_expr = (p_expr * (1 - p_expr) * (1 / X.nA + 1 / X.nB)).sqrt().nullif(0)
            z_expr = ((X.mB / X.nB) - (X.mA / X.nA)) / se_expr

            ibis.cases(
                (I_expr <= 0.5, ibis.literal(2.963)),
                (I_expr >= 1.0, ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                + (ibis.literal(1.96) - ibis.literal(2.963)) * (I_expr - 0.5) / 0.5,
            )

            is_due = (X.I0 < X.planned_t) & (X.planned_t <= I_expr)

            s.write_from(
                X, "snapshot", {"nA": X.nA, "mA": X.mA, "nB": X.nB, "mB": X.mB}
            )
            s.write_from(X, "stat", {"z": z_expr})
            s.write_from(X, "info", {"info_time": I_expr})

            due = X.filter(is_due)
            I_due = (due.nA + due.nB) / due.Nmax.nullif(0)
            p_due = (due.mA + due.mB) / (due.nA + due.nB)
            se_due = (p_due * (1 - p_due) * (1 / due.nA + 1 / due.nB)).sqrt().nullif(0)
            z_due = ((due.mB / due.nB) - (due.mA / due.nA)) / se_due
            bnd_due = ibis.cases(
                (I_due <= 0.5, ibis.literal(2.963)),
                (I_due >= 1.0, ibis.literal(1.96)),
                else_=ibis.literal(2.963)
                + (ibis.literal(1.96) - ibis.literal(2.963)) * (I_due - 0.5) / 0.5,
            )
            action = (z_due.abs() >= bnd_due).ifelse("stop_efficacy", "continue")

            s.write_from(
                due,
                "decision",
                {
                    "look": due.look,
                    "planned_t": due.planned_t,
                    "info_time": I_due,
                    "z": z_due,
                    "boundary": bnd_due,
                    "action": action,
                },
            )


# ============================================================================
# (B) Normal-mixture E-process (labels-bound; expressions only)
# ============================================================================


class ENormalMixture:
    """
    E-process for detecting mean shifts against i.i.d. N(0,1) null using a theta-grid mixture.

    Design rows (committed once):
      - payload_type='e_design' : {"alpha": ...}
      - payload_type='e_theta'  : {"theta": ...}  (multiple rows; uniform weights 1/K)

    Update step:
      - Stage payload_type='e_obs' with {"x": ...}
      - Read alpha, theta grid, and previous S,t from the committed flat view
      - Compute E_t = (1/K) * Σ exp(theta*S - 0.5*theta^2*t)
      - Emit payload_type='e_state' with {"e_value": ..., "alarm": bool}
    """

    def __init__(
        self,
        con: ibis.Client,
        labels: Dict[str, Union[str, int, float, bool]],
        table: str = "ledger",
    ):
        self.con, self.labels, self.table = con, dict(labels), table

    def set_design(self, *, alpha: float, thetas: Sequence[float]) -> None:
        with LedgerSession(self.con, self.table, self.labels) as s:
            s.write("e_design", {"alpha": float(alpha)})
            tbl = ibis.memtable([{"theta": float(th)} for th in thetas])
            s.write_from(tbl, "e_theta", {"theta": tbl["theta"]})

    def update(self, *, x: float) -> None:
        x_add = float(x)
        with LedgerSession(self.con, self.table, self.labels) as s:
            s.write("e_obs", {"x": x_add})

            V = s.view_real()

            D = V.filter(lambda r: r.payload_type == "e_design")
            Dts = D.aggregate(ts_max=D.ts.max())
            Dcur = D.join(Dts, predicates=[D.ts == Dts.ts_max]).select(alpha=D.alpha)

            Theta = V.filter(lambda r: r.payload_type == "e_theta").select(
                theta=V.theta
            )
            K = Theta.aggregate(k=Theta.theta.count())

            Obs_prev = V.filter(lambda r: r.payload_type == "e_obs").select(x_prev=V.x)
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
