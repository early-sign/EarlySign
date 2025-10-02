"""
ReportingBridgeMixin

A lightweight, scheme-agnostic bridge that concrete templates can mix in to
expose reporting methods (`report_tables`, `report_markdown`, `plot`, `get_sql`)
without introducing any scheme dependency into the framework layer.

Requirements on the host class (the template that mixes this in):
- has `self.scoped`: a Ledger bound to an experiment_id
- has `self.registry`: a dict-like storing IDs under key "ids"
    e.g. GST/Interim: {'counts','wald','info','boundary','decision'}
         Safe/AVT   : {'counts','eproc','design','decision'}

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.stats.schemes.two_proportions.records import BinomialCountsRecord
>>> from earlysign.stats.common.anytime_valid.records import (
...     EProcessRecord, SafeDesignRecord, SafeDecisionRecord
... )
>>> con = ibis.duckdb.connect(":memory:")
>>> scoped = Ledger(con, "events").bind(experiment_id="mixA"); scoped.ensure()
>>> # minimal Safe-mode rows
>>> BinomialCountsRecord(id="C1").attach(scoped).insert(payload={"nA":10, "mA":3, "nB":12, "mB":5})
>>> EProcessRecord(id="E1").attach(scoped).insert(payload={"E":9.0, "logE":2.2})
>>> SafeDesignRecord(id="S1").attach(scoped).insert(payload={"alpha":0.05})
>>> SafeDecisionRecord(id="D1").attach(scoped).insert(payload={"signal":"continue", "reason":"n/a", "threshold":20.0})
>>>
>>> class _T(ReportingBridgeMixin):
...     def __init__(self, scoped, ids):
...         self.scoped = scoped
...         self.registry = {"ids": ids}
>>>
>>> t = _T(scoped, {"counts":"C1","eproc":"E1","design":"S1","decision":"D1"})
>>> tbls = t.report_tables("ibis")
>>> isinstance(tbls, dict) and "snapshot_eproc" in tbls  # Safe snapshot keys
True
"""

from typing import Any, Dict, Optional, Union

from matplotlib.figure import Figure

from earlysign.reporting.reportkit import ReporterBase


class ReportingBridgeMixin:
    """Opt-in reporting surface for concrete templates."""

    # -------- internal helper -------------------------------------------------
    def _reporter(self) -> ReporterBase:
        """
        Lazily obtain a scheme-appropriate Reporter via the reporting factory.
        """
        # Local import keeps the mixin free from import cycles.
        from earlysign.reporting.factory import get_reporter

        ids: Dict[str, str] = {}
        registry: Dict[str, Any] = getattr(self, "registry", {}) or {}
        maybe_ids = registry.get("ids", {})
        if isinstance(maybe_ids, dict):
            ids = {str(k): str(v) for k, v in maybe_ids.items()}
        scoped = getattr(self, "scoped", None)
        if scoped is None:
            raise RuntimeError("ReportingBridgeMixin requires `self.scoped` to be set.")
        return get_reporter(scoped, ids)

    # -------- public API ------------------------------------------------------
    def report_tables(self, format: str = "ibis") -> Union[Dict[str, Any], str]:
        """
        Return reporting tables (or markdown string) for the current scheme.
        """
        return self._reporter().report_tables(format=format)

    def report_markdown(self) -> str:
        """
        Return a composed Markdown report from the latest ledger state.
        """
        out = self._reporter().report_tables(format="markdown")
        assert isinstance(out, str)
        return out

    def plot(self, kind: str = "auto", savepath: Optional[str] = None) -> Figure:
        """
        Render a scheme-appropriate plot and return a Matplotlib Figure.

        Parameters
        ----------
        kind : str
            "auto" | "interim" | "interim_summary" | "safe" (depends on scheme)
        savepath : Optional[str]
            If provided, saves the figure to the path.
        """
        return self._reporter().plot(kind=kind, savepath=savepath)

    def get_sql(
        self, kind: str = "interim_summary", *, backend: Any = None
    ) -> Dict[str, str]:
        """
        Compile scheme-specific Ibis expressions to SQL (e.g., for Looker/BI).

        Parameters
        ----------
        kind : str
            Currently supports: "interim_summary" (GST progress)
        backend : Ibis backend connection
        """
        return self._reporter().get_sql(kind=kind, backend=backend)
