"""
Generic gate records used to decide whether downstream operators should run.

Records
-------
- GateDecisionRecord
    payload example:
      {"gate": "open" | "closed", "reason": "insufficient_samples" | "ok", ...}

These records are intentionally scheme-agnostic.
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class GateDecisionRecord(LedgerRecord, QueryMixin):
    """Generic gate decision rows."""

    payload_type: str = "Framework/GateDecision"
