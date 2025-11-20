"""Helpers for Anytime-Valid / Safe testing reports."""

from typing import Any, Dict, Optional

import ibis

from earlysign.core.ledger import Ledger
from earlysign.stats_old.common.anytime_valid.records import (
    EProcessRecord,
    SafeDecisionRecord,
    SafeDesignRecord,
)


def safe_snapshot_tables(
    scoped: Ledger,
    *,
    eproc_id: Optional[str] = None,
    design_id: Optional[str] = None,
    decision_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return snapshot tables for Safe testing.

    Counts data is intentionally omitted to keep this helper scheme-agnostic.
    """
    tables: Dict[str, Any] = {}

    if eproc_id:
        e = EProcessRecord(name=eproc_id).attach(scoped)
        tables["snapshot_eproc"] = e.latest().select(
            E=e.t.payload["E"].cast("float64"),
            logE=e.t.payload["logE"].cast("float64"),
        )

    if design_id:
        dsg = SafeDesignRecord(name=design_id).attach(scoped)
        tables["snapshot_design"] = dsg.latest().select(
            alpha=dsg.t.payload["alpha"].cast("float64")
        )

    if decision_id:
        dec = SafeDecisionRecord(name=decision_id).attach(scoped)
        tables["snapshot_decision"] = dec.latest().select(
            signal=dec.t.payload["signal"],
            reason=dec.t.payload["reason"],
            threshold=dec.t.payload["threshold"].cast("float64"),
        )

    return tables


def safe_plotdata(
    scoped: Ledger,
    *,
    eproc_id: str,
    design_id: str,
) -> Dict[str, Any]:
    """Return tidy tables for plotting an E-value snapshot."""
    e = EProcessRecord(name=eproc_id).attach(scoped)
    d = SafeDesignRecord(name=design_id).attach(scoped)
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
    return {"evalue_snapshot": tidy}
