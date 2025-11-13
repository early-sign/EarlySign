"""
Records for the "one-mean (Gaussian)" scheme.

- OneMeanSummaryRecord
    Snapshot of summary stats:
      {"n": int, "mean": float, "sd": float? (optional), "look": int?}

- ZMeanKnownVarRecord
    Statistic for testing mu = 0 with known variance sigma^2:
      {"z": float, "look": int?}

Doctest (structure only)
------------------------
>>> OneMeanSummaryRecord(name="s1").payload_type
'OneMean/Summary'
>>> ZMeanKnownVarRecord(name="z1").payload_type
'OneMean/ZKnownVar'
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class OneMeanSummaryRecord(LedgerRecord, QueryMixin):
    """Summary stats for a one-mean Gaussian test (n, mean[, sd])."""

    payload_type: str = "OneMean/Summary"


class ZMeanKnownVarRecord(LedgerRecord, QueryMixin):
    """Z statistic for one-mean with known variance."""

    payload_type: str = "OneMean/ZKnownVar"
