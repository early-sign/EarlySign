"""
Group Sequential Testing (scheme-agnostic).

This module provides both pure functional components (essentials/) and
Ledger-integrated operators (operators/) for group sequential designs.

Submodules
----------
essentials
    Pure functions for spending, boundaries, information time, etc.
operators
    LedgerOperator wrappers for Ledger-based workflows
records
    LedgerRecord types for group sequential components

Quick imports
-------------
# Pure functions
from earlysign.stats.common.group_sequential.essentials import spending, boundaries, information

# Operators
from earlysign.stats.common.group_sequential.operators import (
    GroupSequentialDesign,
    BoundaryFromDesign,
    InformationTime,
    GSDecision,
)

# Records
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialDesignRecord,
    GroupSequentialBoundaryRecord,
    InformationTimeRecord,
    GroupSequentialDecisionSignalRecord,
)
"""

# Re-export essentials modules for convenience
from earlysign.stats.common.group_sequential import essentials, operators, records

# Re-export commonly used items
from earlysign.stats.common.group_sequential.essentials import (
    boundaries,
    conversions,
    design_schema,
    information,
    spending,
)
from earlysign.stats.common.group_sequential.operators import (
    BoundaryFromDesign,
    GroupSequentialDesign,
    GSDecision,
    InformationTime,
)
from earlysign.stats.common.group_sequential.records import (
    GroupSequentialBoundaryRecord,
    GroupSequentialDecisionSignalRecord,
    GroupSequentialDesignRecord,
    InformationTimeRecord,
)

__all__ = [
    # Submodules
    "essentials",
    "operators",
    "records",
    # Essentials modules
    "spending",
    "boundaries",
    "information",
    "conversions",
    "design_schema",
    # Operators
    "GroupSequentialDesign",
    "BoundaryFromDesign",
    "InformationTime",
    "GSDecision",
    # Records
    "GroupSequentialDesignRecord",
    "GroupSequentialBoundaryRecord",
    "InformationTimeRecord",
    "GroupSequentialDecisionSignalRecord",
]
