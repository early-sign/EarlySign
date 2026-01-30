"""Records for one-mean (Gaussian) schemes."""

from earlysign.v0.framework.records import LedgerRecord, QueryMixin


class OneMeanSummaryRecord(LedgerRecord, QueryMixin):
    """Summary stats for a one-mean Gaussian test (n, mean[, sd, look])."""

    schema = {
        "n": (int | None, None),
        "mean": (float | None, None),
        "sd": (float | None, None),
        "look": (int | None, None),
    }


class ZMeanKnownVarRecord(LedgerRecord, QueryMixin):
    """Z statistic for one-mean with known variance."""

    schema = {
        "z": (float | None, None),
        "look": (int | None, None),
    }


class EProcessRecord(LedgerRecord, QueryMixin):
    """E-process snapshots for anytime-valid (safe) testing."""

    schema = {
        "E": (float | None, None),
        "logE": (float | None, None),
        "look": (int | None, None),
        "params": (dict | None, None),
    }
