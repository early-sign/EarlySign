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

Usage
-----
Import what you need directly:

    from earlysign.stats.common.group_sequential.essentials import spending
    from earlysign.stats.common.group_sequential.operators import GroupSequentialDesign
    from earlysign.stats.common.group_sequential.records import GroupSequentialDesignRecord
"""
