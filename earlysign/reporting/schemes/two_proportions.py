"""
Two-proportions (binary A/B) reporters.

Ibis-first APIs:
- interim_report / interim_plotdata            : snapshot tables & tidy data
- interim_summary_exprs / interim_summary_sql  : GST progress (history summary)
- interim_summary_plot                         : Matplotlib plot from tidy data
- safe_report / safe_plotdata                  : Safe (Ville) snapshot

Reporter class:
- TwoProportionsReporter (GST/Safe)
- get_reporter(scoped, ids)
"""

from typing import Any, Dict, Literal, Optional, Union

import ibis
from matplotlib.figure import Figure

from earlysign.core.ledger import Ledger
from earlysign.integration.execution.methods.group_sequential.records.boundary import (
    GroupSequentialBoundaryRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.decision import (
    GroupSequentialDecisionSignalRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.info import (
    InformationTimeRecord,
)
from earlysign.integration.execution.methods.group_sequential.records.statistics import (
    WaldZStatisticRecord,
)
from earlysign.integration.execution.schemes.two_proportions.records import (
    BinomialCountsRecord,
)
from earlysign.reporting.reportkit import (
    ReporterBase,
    compose_markdown,
    plot_tidy,
    to_pandas_tables,
    to_sql_dict,
)
from earlysign.stats_old.common.anytime_valid.records import (
    EProcessRecord,
    SafeDecisionRecord,
    SafeDesignRecord,
)

# ----------------------------- helpers (counts) ----------------------------- #


def _latest_counts_expr(scoped: Ledger, counts_id: str) -> Any:
    counts = BinomialCountsRecord(name=counts_id).attach(scoped)
    t = counts.latest().select(
        nA=counts.t.payload["nA"].cast("int64"),
        mA=counts.t.payload["mA"].cast("int64"),
        nB=counts.t.payload["nB"].cast("int64"),
        mB=counts.t.payload["mB"].cast("int64"),
    )
    # Avoid division by zero; align types in coalesce
    one = ibis.literal(1.0).cast("float64")
    return t.mutate(
        pA=(t["mA"].cast("float64") / ibis.coalesce(t["nA"].cast("float64"), one)),
        pB=(t["mB"].cast("float64") / ibis.coalesce(t["nB"].cast("float64"), one)),
    )


# ----------------------------- Interim (GS) snapshot ------------------------ #


def interim_report(
    scoped: Ledger,
    ids: Dict[str, str],
    *,
    format: Literal["ibis", "pandas", "markdown"] = "ibis",
) -> Union[Dict[str, Any], str]:
    """
    Snapshot tables (or Markdown) for GS interim.

    Returns keys (ibis/pandas):
      snapshot_counts, snapshot_stats, snapshot_info, snapshot_boundary, snapshot_decision

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demo"); scoped.ensure()
    >>> # prepare minimal rows
    >>> BinomialCountsRecord(name="C").attach(scoped).insert(payload={"nA": 10, "mA": 3, "nB": 12, "mB": 5})
    >>> WaldZStatisticRecord(name="W").attach(scoped).insert(payload={"wald_z": 1.2})
    >>> InformationTimeRecord(name="I").attach(scoped).insert(payload={"info_time": 0.4})
    >>> GroupSequentialBoundaryRecord(name="B").attach(scoped).insert(payload={"upper": 2.0, "lower": -2.0, "scale": "z"})
    >>> GroupSequentialDecisionSignalRecord(name="D").attach(scoped).insert(payload={"signal": "continue", "reason": "start"})
    >>> out = interim_report(scoped, {"counts":"C","wald":"W","info":"I","boundary":"B","decision":"D"})
    >>> set(out.keys()) == {"snapshot_counts","snapshot_stats","snapshot_info","snapshot_boundary","snapshot_decision"}
    True
    """
    counts = _latest_counts_expr(scoped, ids["counts"])
    w = WaldZStatisticRecord(name=ids["wald"]).attach(scoped)
    i = InformationTimeRecord(name=ids["info"]).attach(scoped)
    b = GroupSequentialBoundaryRecord(name=ids["boundary"]).attach(scoped)
    d = GroupSequentialDecisionSignalRecord(name=ids["decision"]).attach(scoped)

    wald = w.latest().select(wald_z=w.t.payload["wald_z"].cast("float64"))
    info = i.latest().select(t=i.t.payload["t"].cast("float64"))
    bound = b.latest().select(
        upper=b.t.payload["upper"].cast("float64"),
        lower=b.t.payload["lower"].cast("float64"),
        scale=b.t.payload["scale"],
    )
    dec = d.latest().select(
        signal=d.t.payload["signal"],
        reason=d.t.payload["reason"],
    )

    tables: Dict[str, Any] = {
        "snapshot_counts": counts,
        "snapshot_stats": wald,
        "snapshot_info": info,
        "snapshot_boundary": bound,
        "snapshot_decision": dec,
    }

    if format == "ibis":
        return tables
    if format == "pandas":
        return to_pandas_tables(tables)
    if format == "markdown":
        p = to_pandas_tables(tables)
        return compose_markdown([("Interim Analysis (Two Proportions)", p)])
    raise ValueError("format must be 'ibis', 'pandas', or 'markdown'")


def interim_plotdata(
    scoped: Ledger,
    ids: Dict[str, str],
    *,
    format: Literal["ibis", "pandas"] = "ibis",
) -> Dict[str, Any]:
    """
    Plot-ready tidy table for GS interim (single-look snapshot).

    Returns
    -------
    {"wald_vs_boundary": (look=1, value, series ∈ {lower, point, upper})}

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demo2"); scoped.ensure()
    >>> WaldZStatisticRecord(name="W2").attach(scoped).insert(payload={"wald_z": 1.5})
    >>> GroupSequentialBoundaryRecord(name="B2").attach(scoped).insert(upper=2.0, lower=-2.0, scale="z", alpha=0.05, tails=2, info_time=0.5)
    >>> out = interim_plotdata(scoped, {"wald":"W2","boundary":"B2"})
    >>> list(out.keys()) == ["wald_vs_boundary"]
    True
    """
    w = WaldZStatisticRecord(name=ids["wald"]).attach(scoped)
    b = GroupSequentialBoundaryRecord(name=ids["boundary"]).attach(scoped)
    wdf = w.latest().select(
        look=ibis.literal(1).cast("int64"),
        value=w.t.payload["wald_z"].cast("float64"),
        series=ibis.literal("point"),
    )
    bup = b.latest().select(
        look=ibis.literal(1).cast("int64"),
        value=b.t.payload["upper"].cast("float64"),
        series=ibis.literal("upper"),
    )
    blo = b.latest().select(
        look=ibis.literal(1).cast("int64"),
        value=b.t.payload["lower"].cast("float64"),
        series=ibis.literal("lower"),
    )
    tidy = wdf.union(bup).union(blo)
    tables: Dict[str, Any] = {"wald_vs_boundary": tidy}
    return tables if format == "ibis" else to_pandas_tables(tables)


# ----------------------------- Interim (GS) progress ------------------------ #
def interim_summary_exprs(scoped: Ledger, ids: Dict[str, str]) -> Dict[str, Any]:
    """
    Build Ibis expressions for the “GST Progress” plot (history summary).

    Returns
    -------
    dict with keys:
      - bounds_t, wald_t, stop_ts, tidy  # (t, y, series) in tidy
    """
    b = GroupSequentialBoundaryRecord(name=ids["boundary"]).attach(scoped)
    i = InformationTimeRecord(name=ids["info"]).attach(scoped)
    w = WaldZStatisticRecord(name=ids["wald"]).attach(scoped)
    d = GroupSequentialDecisionSignalRecord(name=ids["decision"]).attach(scoped)

    # Information time with next_ts via column .lead() over a window
    info = i.t.select(ts=i.t.ts, t=i.t.payload["t"].cast("float64"))
    wwin = ibis.window(order_by="ts")
    info = info.mutate(next_ts=lambda t: t["ts"].lead().over(wwin))

    bounds = b.t.select(
        ts=b.t.ts,
        upper=b.t.payload["upper"].cast("float64"),
        lower=b.t.payload["lower"].cast("float64"),
    )
    wald = w.t.select(ts=w.t.ts, wz=w.t.payload["wald_z"].cast("float64"))

    # ✅ cast signal to string to avoid JSON comparison/convert issues
    decision = d.t.select(
        ts=d.t.ts,
        signal=d.t.payload["signal"].cast("string"),
    )

    # Interval join via CROSS JOIN + filter
    bi = (
        bounds.cross_join(info)
        .filter(
            [
                (info.ts <= bounds.ts)
                & (info.next_ts.isnull() | (bounds.ts < info.next_ts))
            ]
        )
        .select(t=info.t, ts=bounds.ts, upper=bounds.upper, lower=bounds.lower)
    )

    wi = (
        wald.cross_join(info)
        .filter(
            [(info.ts <= wald.ts) & (info.next_ts.isnull() | (wald.ts < info.next_ts))]
        )
        .select(t=info.t, ts=wald.ts, wz=wald.wz)
    )

    # Stop timestamp: any non-"continue" decision
    stop_ts = decision.filter(
        decision.signal.notnull() & (decision.signal != "continue")
    ).agg(ts=decision.ts.max())

    upper_series = bi.select(t=bi.t, y=bi.upper, series=ibis.literal("upper"))
    lower_series = bi.select(t=bi.t, y=bi.lower, series=ibis.literal("lower"))
    wald_series = wi.select(t=wi.t, y=wi.wz, series=ibis.literal("wald"))
    stop_series = wi.join(stop_ts, predicates=[wi.ts == stop_ts.ts]).select(
        t=wi.t, y=wi.wz, series=ibis.literal("stop")
    )

    tidy = upper_series.union(lower_series).union(wald_series).union(stop_series)

    return {"bounds_t": bi, "wald_t": wi, "stop_ts": stop_ts, "tidy": tidy}


def interim_summary_sql(
    scoped: Ledger, ids: Dict[str, str], *, backend: Any
) -> Dict[str, str]:
    """
    Compile the GST “progress” expressions to SQL with the given backend.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demo4"); scoped.ensure()
    >>> # minimal rows
    >>> InformationTimeRecord(name="I4").attach(scoped).insert(info_time=0.5)
    >>> GroupSequentialBoundaryRecord(name="B4").attach(scoped).insert(upper=2.0, lower=-2.0, scale="z")
    >>> WaldZStatisticRecord(name="W4").attach(scoped).insert(wald_z=1.7)
    >>> GroupSequentialDecisionSignalRecord(name="D4").attach(scoped).insert(signal="continue", reason="n/a")
    >>> sqls = interim_summary_sql(scoped, {"info":"I4","boundary":"B4","wald":"W4","decision":"D4"}, backend=con)
    >>> set(sqls.keys()) == {"bounds_t","wald_t","stop_ts","tidy"}
    True
    """
    exprs = interim_summary_exprs(scoped, ids)
    return to_sql_dict(backend, exprs)


def interim_summary_plot(
    scoped: Ledger, ids: Dict[str, str], *, savepath: Optional[str] = None
) -> Figure:
    """
    Render the “GST Progress (Two-Proportions)” summary plot from tidy data.

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demo5"); scoped.ensure()
    >>> InformationTimeRecord(name="I5").attach(scoped).insert(info_time=0.4)
    >>> GroupSequentialBoundaryRecord(name="B5").attach(scoped).insert(upper=2.0, lower=-2.0, scale="z")
    >>> WaldZStatisticRecord(name="W5").attach(scoped).insert(wald_z=1.1)
    >>> GroupSequentialDecisionSignalRecord(name="D5").attach(scoped).insert(signal="continue", reason="n/a")
    >>> fig = interim_summary_plot(scoped, {"info":"I5","boundary":"B5","wald":"W5","decision":"D5"})
    >>> hasattr(fig, "savefig")
    True
    """
    df = to_pandas_tables({"tidy": interim_summary_exprs(scoped, ids)["tidy"]})["tidy"]
    df = df.dropna(subset=["t", "y"]).sort_values("t")
    return plot_tidy(
        df,
        x="t",
        y="y",
        series="series",
        kind="hline_mark",
        title="GST Progress (Two-Proportions)",
        xlabel="Information fraction (t)",
        ylabel="Z",
        savepath=savepath,
    )


# ----------------------------- Safe (Ville) snapshot ------------------------ #
def safe_report(
    scoped: Ledger,
    ids: Dict[str, str],
    *,
    format: Literal["ibis", "pandas", "markdown"] = "ibis",
) -> Union[Dict[str, Any], str]:
    """
    Snapshot tables (or Markdown) for Safe testing (Ville).

    Keys (present if provided in ids): snapshot_counts, snapshot_eproc, snapshot_design, snapshot_decision
    """
    tables: Dict[str, Any] = {}

    # counts is optional in Safe "smoke" scenarios
    if ids.get("counts"):
        tables["snapshot_counts"] = _latest_counts_expr(scoped, ids["counts"])

    if ids.get("eproc"):
        e = EProcessRecord(name=ids["eproc"]).attach(scoped)
        tables["snapshot_eproc"] = e.latest().select(
            E=e.t.payload["E"].cast("float64"),
            logE=e.t.payload["logE"].cast("float64"),
        )

    if ids.get("design"):
        dsg = SafeDesignRecord(name=ids["design"]).attach(scoped)
        tables["snapshot_design"] = dsg.latest().select(
            alpha=dsg.t.payload["alpha"].cast("float64")
        )

    if ids.get("decision"):
        dec = SafeDecisionRecord(name=ids["decision"]).attach(scoped)
        tables["snapshot_decision"] = dec.latest().select(
            signal=dec.t.payload["signal"],
            reason=dec.t.payload["reason"],
            threshold=dec.t.payload["threshold"].cast("float64"),
        )

    if format == "ibis":
        return tables
    if format == "pandas":
        return to_pandas_tables(tables)
    if format == "markdown":
        p = to_pandas_tables(tables)
        return compose_markdown([("Safe Testing (Two Proportions)", p)])
    raise ValueError("format must be 'ibis', 'pandas', or 'markdown'")


def safe_plotdata(
    scoped: Ledger,
    ids: Dict[str, str],
    *,
    format: Literal["ibis", "pandas"] = "ibis",
) -> Dict[str, Any]:
    """
    Plot-ready tidy table for Safe testing: E-value & threshold.

    Returns
    -------
    {"evalue_snapshot": tidy table (x, value, series ∈ {E, threshold})}

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demo7"); scoped.ensure()
    >>> EProcessRecord(name="E7").attach(scoped).insert(E=10.0, logE=2.3)
    >>> SafeDesignRecord(name="S7").attach(scoped).insert(alpha=0.05)
    >>> out = safe_plotdata(scoped, {"eproc":"E7","design":"S7"})
    >>> list(out.keys()) == ["evalue_snapshot"]
    True
    """
    e = EProcessRecord(name=ids["eproc"]).attach(scoped)
    d = SafeDesignRecord(name=ids["design"]).attach(scoped)
    e_tbl = e.latest().select(
        x=ibis.literal("E"),
        value=e.t.payload["E"].cast("float64"),
        series=ibis.literal("E"),
    )
    t_tbl = d.latest().select(
        x=ibis.literal("Threshold"),
        value=(1.0 / d.t.payload["alpha"].cast("float64")),
        series=ibis.literal("threshold"),
    )
    tidy = e_tbl.union(t_tbl)
    tables: Dict[str, Any] = {"evalue_snapshot": tidy}
    return tables if format == "ibis" else to_pandas_tables(tables)


# ----------------------------- Reporter & factory --------------------------- #


class TwoProportionsReporter(ReporterBase):
    """
    Reporter for the two-proportions scheme (GST/Safe).

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demoR"); scoped.ensure()
    >>> # Safe mode smoke
    >>> EProcessRecord(name="ER").attach(scoped).insert(E=5.0, logE=1.6)
    >>> SafeDesignRecord(name="SR").attach(scoped).insert(alpha=0.05)
    >>> r2 = TwoProportionsReporter(scoped, {"eproc":"ER","design":"SR"})
    >>> isinstance(r2.report_tables("ibis"), dict)
    True
    """

    def _mode(self) -> str:
        ids = self.ids
        if "boundary" in ids and "wald" in ids:
            return "interim"
        if "eproc" in ids and "design" in ids:
            return "safe"
        raise RuntimeError("Unknown ids for two-proportions reporter.")

    def report_tables(self, format: str = "ibis") -> Union[Dict[str, Any], str]:
        mode = self._mode()
        if mode == "interim":
            return interim_report(self.scoped, self.ids, format=format)  # type: ignore[arg-type]
        return safe_report(self.scoped, self.ids, format=format)  # type: ignore[arg-type]

    def plot(self, kind: str = "auto", savepath: Optional[str] = None) -> Figure:
        mode = self._mode() if kind == "auto" else None
        if kind in ("interim", "interim_summary") or mode == "interim":
            if kind == "interim":
                pdf = interim_plotdata(self.scoped, self.ids, format="pandas")[
                    "wald_vs_boundary"
                ]
                return plot_tidy(
                    pdf,
                    x="look",
                    y="value",
                    series="series",
                    kind="hline_mark",
                    title="Interim Snapshot (Z-scale)",
                    xlabel="look",
                    ylabel="Z",
                    savepath=savepath,
                )
            return interim_summary_plot(self.scoped, self.ids, savepath=savepath)
        # safe
        pdf = safe_plotdata(self.scoped, self.ids, format="pandas")["evalue_snapshot"]
        return plot_tidy(
            pdf,
            x="x",
            y="value",
            series="series",
            kind="bar",
            title="Safe Testing Snapshot (E-value)",
            xlabel="metric",
            ylabel="value",
            savepath=savepath,
        )

    def get_sql(
        self, kind: str = "interim_summary", *, backend: Any = None
    ) -> Dict[str, str]:
        if kind != "interim_summary":
            raise ValueError("Only 'interim_summary' is supported for SQL emission.")
        if backend is None:
            raise ValueError("backend is required to emit SQL.")
        return interim_summary_sql(self.scoped, self.ids, backend=backend)


def get_reporter(scoped: Ledger, ids: Dict[str, str]) -> ReporterBase:
    """
    Factory to obtain a reporter from the template context.

    Examples
    --------
    >>> r = get_reporter(scoped=None, ids={"eproc":"E","design":"S"})
    >>> isinstance(r, ReporterBase)
    True
    """
    return TwoProportionsReporter(scoped, ids)
