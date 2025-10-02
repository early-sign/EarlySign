"""
Records for the "two-proportions" scheme (binary outcomes A vs B).

This module defines minimal, scheme-specific record types that other operators
(estimators, group-sequential, anytime-valid) can read/write.
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class BinomialCountsRecord(LedgerRecord, QueryMixin):
    """Counts snapshot for two proportions (A vs B). See module docstring for schema."""

    payload_type = "TwoProportions/Counts"
    schema = {
        "nA": (int, ...),  # trials in A
        "mA": (int, ...),  # successes in A
        "nB": (int, ...),  # trials in B
        "mB": (int, ...),  # successes in B
        "look": (int, ...),  # optional: interim look index
    }


class WaldZStatisticRecord(LedgerRecord, QueryMixin):
    """Wald Z statistic record for two-proportions."""

    payload_type = "TwoProportions/WaldZ"
    schema = {
        "wald_z": (float, ...),
        "look": (int, ...),
    }


class ScoreZStatisticRecord(LedgerRecord, QueryMixin):
    """Score Z statistic record for two-proportions."""

    payload_type = "TwoProportions/ScoreZ"
    schema = {
        "score_z": (float, ...),
        "look": (int, ...),
    }
