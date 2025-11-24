"""Records for Anytime-Valid (Safe) testing, scheme-agnostic."""

from earlysign.framework.records import LedgerRecord, QueryMixin


class EProcessRecord(LedgerRecord, QueryMixin):
    """E-process snapshots for anytime-valid (safe) testing."""

    schema = {
        "E": (float, ...),
        "logE": (float, ...),
    }


class VilleThresholdRecord(LedgerRecord, QueryMixin):
    """Ville threshold rows: 1/alpha."""

    schema = {
        "alpha": (float, ...),
        "threshold": (float, ...),
    }


class SafeDecisionRecord(LedgerRecord, QueryMixin):
    """Decisions based on Ville's inequality (safe testing)."""

    schema = {
        "criterion": (str, ...),
        "signal": (str, ...),
        "E": (float, ...),
        "threshold": (float, ...),
        "alpha": (float, ...),
    }


class SafeDesignRecord(LedgerRecord, QueryMixin):
    """Design for anytime-valid (safe) testing."""

    schema = {
        "alpha": (float, ...),
        "futility": (dict | None, None),
    }
