"""
Records for Anytime-Valid (Safe) testing, scheme-agnostic.

Record types
------------
- EProcessRecord
    Stores an e-process snapshot (multiplicative supermartingale under H0).
    Example payload:
      {"E": 12.3, "logE": 2.509, "look": 2}

- VilleThresholdRecord
    Stores the Ville threshold for a given alpha: threshold = 1/alpha.
    Example payload:
      {"alpha": 0.05, "threshold": 20.0}

- SafeDecisionRecord
    Stores a safe-testing decision using Ville's inequality:
      {"criterion": "Ville", "signal": "reject"|"continue"|"stop_futility",
       "E": 12.3, "threshold": 20.0, "alpha": 0.05, "look": 2}

Doctest (structure only)
------------------------
>>> from earlysign.stats.common.anytime_valid.records import (
...     EProcessRecord, VilleThresholdRecord, SafeDecisionRecord
... )
>>> EProcessRecord(id="e1").payload_type
'stats.common.anytime_valid.records.EProcessRecord'
>>> VilleThresholdRecord(id="v1").payload_type
'stats.common.anytime_valid.records.VilleThresholdRecord'
>>> SafeDecisionRecord(id="d1").payload_type
'stats.common.anytime_valid.records.SafeDecisionRecord'
"""

from earlysign.framework.records import LedgerRecord, QueryMixin


class EProcessRecord(LedgerRecord, QueryMixin):
    """E-process snapshots for anytime-valid (safe) testing."""

    pass


class VilleThresholdRecord(LedgerRecord, QueryMixin):
    """Ville threshold rows: 1/alpha."""

    pass


class SafeDecisionRecord(LedgerRecord, QueryMixin):
    """Decisions based on Ville's inequality (safe testing)."""

    pass


class SafeDesignRecord(LedgerRecord, QueryMixin):
    """Design for anytime-valid (safe) testing.

    Payload example
    ---------------
    {"alpha": 0.05, "futility": {"mode": "fixed", "tau": 0.1}}
    """

    pass
