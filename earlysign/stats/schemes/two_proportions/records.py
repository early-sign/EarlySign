"""
Records for the "two-proportions" scheme (binary outcomes A vs B).

This module defines minimal, scheme-specific record types that other operators
(estimators, group-sequential, anytime-valid) can read/write.

- BinomialCountsRecord
    Snapshot of counts for two arms (A, B):
      payload = {
        "nA": int,   # trials in A
        "mA": int,   # successes in A
        "nB": int,   # trials in B
        "mB": int,   # successes in B
        "look": int, # optional: interim look index
        "labelA": str, "labelB": str,  # optional: arm labels
      }

- WaldZStatisticRecord
    Result of a Wald Z computation for (pB - pA):
      payload = {"wald_z": float, "look"?: int}

- ScoreZStatisticRecord
    Result of a Score (Z) computation for (pB - pA):
      payload = {"score_z": float, "look"?: int}
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class BinomialCountsRecord(LedgerRecord, QueryMixin):
    """Counts snapshot for two proportions (A vs B). See module docstring for schema."""

    payload_type: str = "TwoProportions/Counts"


class WaldZStatisticRecord(LedgerRecord, QueryMixin):
    """Wald Z statistic record for two-proportions."""

    payload_type: str = "TwoProportions/WaldZ"


class ScoreZStatisticRecord(LedgerRecord, QueryMixin):
    """Score Z statistic record for two-proportions."""

    payload_type: str = "TwoProportions/ScoreZ"
