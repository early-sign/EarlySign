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

    Note: This record DOES NOT store look numbers. Look numbers are
    contextual to the analysis and should be tracked separately.

    Payload example:
      {"info_time": 0.5}
    """

    payload_type: str = "GroupSequential/InfoTime"
    schema = {
        "info_time": float,
    }


class GroupSequentialDesignRecord(LedgerRecord, QueryMixin):
    """
    Design record for Group Sequential Testing.

    Stores the complete design specification including spending functions,
    boundaries, and other configuration.

    Payload example:
      {
        "alpha": 0.05,
        "tails": 2,
        "scale": "z",
        "efficacy": {"style": "alpha_spending", "family": "obf"},
        "futility": {"mode": "none"},
        "binding_mode": "non_binding",
        "planned_max_n": 1000
      }
    """

    payload_type: str = "GroupSequential/Design"
    schema = {
        "alpha": float,
        "tails": int,
        "scale": str,
        "efficacy": dict,
        "futility": dict,
        "binding_mode": str,  # optional
        "planned_max_n": int,  # optional
    }


class GroupSequentialBoundaryRecord(LedgerRecord, QueryMixin):
    """
    Nominal boundaries resolved from a GS design.

    Stores efficacy and futility boundaries resolved at a given
    information time, plus design metadata.

    Payload example:
      {
        "upper": 2.5,
        "lower": -2.5,
        "efficacy": {"upper": 2.5},
        "futility": {"lower": -2.5, "binding": false, "mode": "none"},
        "scale": "z",
        "alpha": 0.05,
        "tails": 2,
        "info_time": 0.5,
        "look": 2
      }
    """

    payload_type: str = "GroupSequential/Boundary"
    schema = {
        "upper": float,
        "lower": float,
        "efficacy": dict,
        "futility": dict,
        "scale": str,
        "alpha": float,
        "tails": int,
        "info_time": float,
        "look": int,  # optional
    }


class GroupSequentialDecisionSignalRecord(LedgerRecord, QueryMixin):
    """
    Stop/continue signal produced by comparing a statistic to boundaries.

    Stores a decision emitted by comparing a statistic (e.g., Wald Z)
    against the boundaries.

    Payload example:
      {
        "signal": "stop_efficacy" | "stop_futility" | "continue",
        "reason": "efficacy" | "futility" | "none",
        "value": 2.34,
        "value_scale": "z",
        "upper": 2.5,
        "lower": -2.5,
        "scale": "z",
        "info_time": 0.5
      }
    """

    payload_type: str = "GroupSequential/Decision"
    schema = {
        "signal": str,
        "reason": str,
        "value": float,
        "value_scale": str,
        "upper": float,
        "lower": float,
        "scale": str,
        "info_time": float,  # optional
    }
