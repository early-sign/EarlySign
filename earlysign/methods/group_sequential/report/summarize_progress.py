"""Helpers for group-sequential interim reporting."""

from typing import Any, Dict, Optional

import ibis
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from earlysign.core.ledger import Ledger
from earlysign.methods.group_sequential.boundary import GroupSequentialBoundaryRecord
from earlysign.methods.group_sequential.decision import (
    GroupSequentialDecisionSignalRecord,
)
from earlysign.methods.group_sequential.info_time import InformationTimeRecord
from earlysign.methods.group_sequential.schemes.two_proportions.wald_z import (
    WaldZStatisticRecord,
)


def interim_plotdata(
    ledger: Ledger,
    *,
    wald_id: str,
    boundary_id: str,
) -> Dict[str, Any]:
    """Return tidy data for a single interim snapshot (wald vs. boundary)."""
    w = WaldZStatisticRecord(name=wald_id).attach(ledger)
    b = GroupSequentialBoundaryRecord(name=boundary_id).attach(ledger)

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
    return {"wald_vs_boundary": tidy}


def interim_summary_exprs(
    ledger: Ledger,
    *,
    boundary_id: str,
    info_id: str,
    wald_id: str,
    decision_id: str,
) -> Dict[str, Any]:
    """Build Ibis expressions for the GST progress (history) plot."""
    b = GroupSequentialBoundaryRecord(name=boundary_id).attach(ledger)
    i = InformationTimeRecord(name=info_id).attach(ledger)
    w = WaldZStatisticRecord(name=wald_id).attach(ledger)
    d = GroupSequentialDecisionSignalRecord(name=decision_id).attach(ledger)

    info = i.t.select(ts=i.t.ts, t=i.t.payload["t"].cast("float64"))
    wwin = ibis.window(order_by="ts")
    info = info.mutate(next_ts=lambda t: t["ts"].lead().over(wwin))

    bounds = b.t.select(
        ts=b.t.ts,
        upper=b.t.payload["upper"].cast("float64"),
        lower=b.t.payload["lower"].cast("float64"),
    )
    wald = w.t.select(ts=w.t.ts, wz=w.t.payload["wald_z"].cast("float64"))

    decision = d.t.select(
        ts=d.t.ts,
        signal=d.t.payload["signal"].cast("string"),
    )

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


def interim_summary_plot(
    ledger: Ledger,
    *,
    boundary_id: str,
    info_id: str,
    wald_id: str,
    decision_id: str,
    savepath: Optional[str] = None,
) -> Figure:
    """Render the GST progress plot from tidy data."""
    tidy_expr = interim_summary_exprs(
        ledger,
        boundary_id=boundary_id,
        info_id=info_id,
        wald_id=wald_id,
        decision_id=decision_id,
    )["tidy"]
    df = tidy_expr.execute()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df = df.dropna(subset=["t", "y"]).sort_values("t")

    fig, ax = plt.subplots()
    ax.set_title("GST Progress", fontsize=14)
    ax.set_xlabel("Information fraction (t)")
    ax.set_ylabel("Z")

    lows = df[df["series"] == "lower"]
    ups = df[df["series"] == "upper"]
    pts = df[df["series"].isin(["point", "wald", "stop"])]

    if len(lows) > 0:
        ax.hlines(
            lows["y"].tolist(),
            xmin=df["t"].min(),
            xmax=df["t"].max(),
            linestyles="dashed",
            label="Lower",
        )
    if len(ups) > 0:
        ax.hlines(
            ups["y"].tolist(),
            xmin=df["t"].min(),
            xmax=df["t"].max(),
            linestyles="dashed",
            label="Upper",
        )
    if len(pts) > 0:
        ax.plot(pts["t"], pts["y"], marker="o", label="Observed/Stop")

    ax.legend()
    ax.grid(True, alpha=0.3)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig
