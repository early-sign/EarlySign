"""
Ledger operators for group sequential testing.

This module contains LedgerOperator wrappers around the pure functions
in essentials/. These operators:

- Read from and write to the Ledger
- Handle record attachments and queries
- Maintain backward compatibility with existing code

Operators
---------
InfoTimeOperators:
    InformationTime
    InformationTimeFromRatio
    InformationTimeFromVariance
    InformationTimeFromSD
    InformationTimeFromFisher

DesignOperators:
    GroupSequentialDesign
    BoundaryFromDesign

DecisionOperators:
    GSDecision
    GSDecisionFromWaldZ

Examples
--------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.stats.common.group_sequential.operators import InformationTime
>>> con = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(con, "events")
>>> ledger.ensure()
>>> # Use operators as before
"""

from earlysign.stats.common.group_sequential.operators.boundary_op import (
    BoundaryFromDesign,
)
from earlysign.stats.common.group_sequential.operators.decision_op import (
    GSDecision,
    GSDecisionFromWaldZ,
)
from earlysign.stats.common.group_sequential.operators.design_op import (
    GroupSequentialDesign,
)
from earlysign.stats.common.group_sequential.operators.info_op import (
    InformationTime,
    InformationTimeFromFisher,
    InformationTimeFromRatio,
    InformationTimeFromSD,
    InformationTimeFromVariance,
)

__all__ = [
    # Design operators
    "GroupSequentialDesign",
    "BoundaryFromDesign",
    # Information time operators
    "InformationTime",
    "InformationTimeFromRatio",
    "InformationTimeFromVariance",
    "InformationTimeFromSD",
    "InformationTimeFromFisher",
    # Decision operators
    "GSDecision",
    "GSDecisionFromWaldZ",
]
