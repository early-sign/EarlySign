"""
Records for the "two-proportions" scheme (binary outcomes A vs B).

This module defines minimal, scheme-specific record types that other operators
(estimators, group-sequential, anytime-valid) can read/write.
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class BinomialCountsRecord(LedgerRecord, QueryMixin):
    """Newly arrived counts for two proportions (A vs B)."""

    schema = {
        "nA": (int, ...),  # trials in A
        "mA": (int, ...),  # successes in A
        "nB": (int, ...),  # trials in B
        "mB": (int, ...),  # successes in B
    }


class BinomialCountsSnapshotRecord(LedgerRecord, QueryMixin):
    """Cumulative counts snapshot for two proportions (A vs B).

    This records the cumulative totals up to this point in time.
    """

    schema = {
        "nA": (int, ...),  # cumulative trials in A
        "mA": (int, ...),  # cumulative successes in A
        "nB": (int, ...),  # cumulative trials in B
        "mB": (int, ...),  # cumulative successes in B
    }


class ScoreZStatisticRecord(LedgerRecord, QueryMixin):
    """Score Z statistic record for two-proportions."""

    schema = {
        "score_z": (float, ...),
    }
