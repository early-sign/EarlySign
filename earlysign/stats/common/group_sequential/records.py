# -*- coding: utf-8 -*-
"""
Core records for scheme-agnostic Group Sequential components.

These records are intentionally minimal and free of scheme-specific payloads.
Operators in this package (e.g., info_time, design, boundary, decision) read
and write these records.

Record types
------------
- InformationTimeRecord
    Stores information time t in [0, 1] and optional look index.
    Example payload:
      {"info_time": 0.5, "look": 2}

- GroupSequentialBoundaryRecord
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

- GroupSequentialDecisionSignalRecord
    Stores a decision emitted by comparing a statistic (e.g., Wald Z)
    against the boundaries.
    Example payload:
      {
        "signal": "stop" | "continue",
        "z": 2.34,
        "info_time": 0.5   # optional passthrough from info record
      }

Doctest (structure only)
------------------------
>>> InformationTimeRecord(id="i1").payload_type
'GroupSequential/InfoTime'
>>> GroupSequentialBoundaryRecord(id="b1").payload_type
'GroupSequential/Boundary'
>>> GroupSequentialDecisionSignalRecord(id="d1").payload_type
'GroupSequential/Decision'
"""
from earlysign.framework.records import LedgerRecord, QueryMixin


class InformationTimeRecord(LedgerRecord, QueryMixin):
    """Information time snapshots (scheme-agnostic)."""

    payload_type: str = "GroupSequential/InfoTime"


class GroupSequentialBoundaryRecord(LedgerRecord, QueryMixin):
    """Nominal Z boundaries resolved from a GS design."""

    payload_type: str = "GroupSequential/Boundary"


class GroupSequentialDecisionSignalRecord(LedgerRecord, QueryMixin):
    """Stop/continue signal produced by comparing a statistic to boundaries."""

    payload_type: str = "GroupSequential/Decision"


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
