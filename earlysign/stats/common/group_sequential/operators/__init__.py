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

Usage
-----
Import operators directly:

    from earlysign.stats.common.group_sequential.operators.info_op import InformationTime
    from earlysign.stats.common.group_sequential.operators.design_op import GroupSequentialDesign
"""
