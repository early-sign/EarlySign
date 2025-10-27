"""
Factory / dispatcher for reporting.

- `framework/templates.py` must depend only on this module (no scheme imports).
- Reporters are lazily imported here so scheme dependencies do not leak into
  the framework layer.

Detection heuristic
-------------------
- Interim (Group-Sequential): ids include {"wald", "boundary"} (and typically "info")
- Safe (Anytime-Valid):       ids include {"eproc", "design"}

Extensibility
-------------
To add a new scheme, implement its Reporter in `earlysign.reporting.schemes.<name>`
and add a new lazy-import branch below.
"""

from typing import Any, Dict

from earlysign.reporting.reportkit import ReporterBase


def _detect_scheme(ids: Dict[str, str]) -> str:
    """Detect a scheme from the id keys.

    Examples
    --------
    >>> _detect_scheme({"wald": "W", "boundary": "B", "info": "I"})
    'two_proportions'
    >>> _detect_scheme({"eproc": "E", "design": "S", "counts": "C", "decision": "D"})
    'two_proportions'
    """
    keys = set(ids.keys())
    if {"wald", "boundary"} <= keys or {"eproc", "design"} <= keys:
        return "two_proportions"
    raise NotImplementedError(
        "Cannot detect reporting scheme from ids. "
        "Provide ids for GST ('wald' & 'boundary') or Safe ('eproc' & 'design')."
    )


def get_reporter(
    scoped: Any, ids: Dict[str, str], *, scheme: str | None = None
) -> ReporterBase:
    """
    Return a scheme-appropriate Reporter without exposing scheme to the framework.

    Parameters
    ----------
    scoped : Ledger bound to an experiment_id
    ids    : Dict[str, str] of record ids (e.g., {'counts': 'C', 'wald': 'W', ...})
    scheme : Optional explicit scheme name; when None, auto-detect via `_detect_scheme`

    Examples
    --------
    >>> import ibis
    >>> from earlysign.core.ledger import Ledger
    >>> from earlysign.stats.applications.execution.schemes.two_proportions.records import BinomialCountsRecord
    >>> from earlysign.stats.common.anytime_valid.records import EProcessRecord, SafeDesignRecord, SafeDecisionRecord
    >>> con = ibis.duckdb.connect(":memory:")
    >>> scoped = Ledger(con, "events").bind(experiment_id="demoF"); scoped.ensure()
    >>> # minimal Safe-mode rows
    >>> BinomialCountsRecord(id="C8").attach(scoped).insert(payload={"nA": 10, "mA": 3, "nB": 11, "mB": 4})
    >>> EProcessRecord(id="E8").attach(scoped).insert(payload={"E":7.0, "logE":1.95})
    >>> SafeDesignRecord(id="S8").attach(scoped).insert(payload={"alpha":0.05})
    >>> SafeDecisionRecord(id="D8").attach(scoped).insert(payload={"signal":"continue", "reason":"n/a", "threshold":20.0})
    >>> r = get_reporter(scoped, {"counts":"C8","eproc":"E8","design":"S8","decision":"D8"})
    >>> isinstance(r, ReporterBase)
    True
    >>> isinstance(r.report_tables("ibis"), dict)
    True
    """
    name = scheme or _detect_scheme(ids)

    if name == "two_proportions":
        # Lazy import keeps framework free of scheme dependencies
        from earlysign.reporting.schemes.two_proportions import TwoProportionsReporter

        return TwoProportionsReporter(scoped, ids)

    raise NotImplementedError(f"Reporter for scheme '{name}' is not registered.")
