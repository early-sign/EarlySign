"""
Core records for scheme-agnostic Group Sequential components.

These records are intentionally minimal and free of scheme-specific payloads.
Operators in this package (e.g., info_time, design, boundary, decision) read
and write these records.
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class InformationTimeRecord(LedgerRecord, QueryMixin):
    """
    Information time snapshots (scheme-agnostic).
    Stores information time t in [0, 1].
    Example payload:
      {"info_time": 0.5}
    """

    payload_type: str = "GroupSequential/InfoTime"
    schema = {
        "info_time": float,
    }


class GroupSequentialBoundaryRecord(LedgerRecord, QueryMixin):
    """
    Nominal Z boundaries resolved from a GS design.
    Stores nominal Z boundaries (symmetric by default) resolved at a given
    information time (and possibly look), plus design metadata.
    Example payload:
      {
        "upper": 2.5,
        "lower": -2.5,
        "alpha": 0.05,
        "style": "alpha_spending" | "significance_level",
        "info_time": 0.5,
        "look": 2   # optional
      }
    """

    payload_type: str = "GroupSequential/Boundary"
    schema = {
        "upper": float,
        "lower": float,
        "alpha": float,
        "style": str,
        "info_time": float,
        "look": int,
        "scale": str,
    }


class GroupSequentialDecisionSignalRecord(LedgerRecord, QueryMixin):
    """
    Stop/continue signal produced by comparing a statistic to boundaries.
    Stores a decision emitted by comparing a statistic (e.g., Wald Z)
    against the boundaries.
    Example payload:
      {
        "signal": "stop" | "continue",
        "z": 2.34,
        "info_time": 0.5   # optional passthrough from info record
      }
    """

    payload_type: str = "GroupSequential/Decision"
    schema = {
        "signal": str,
        "z": float,
        "info_time": float,
    }


class GroupSequentialDesignRecord(LedgerRecord, QueryMixin):
    """Design record for Group Sequential Testing.

    Payload example
    ---------------
    {
      "alpha": 0.05,
      "tails": 2,
      "scale": "z",
      "efficacy": {"style": "alpha_spending", "family": "obf", "alpha_levels": [ ... ]},
      "futility": {"mode": "symmetric"}  # or {"mode": "none"} / {"mode": "binding", ...}
    }
    """

    payload_type: str = "GroupSequential/Design"
    schema = {
        "alpha": float,
        "tails": int,
        "scale": str,
        "efficacy": dict,
        "futility": dict,
    }
