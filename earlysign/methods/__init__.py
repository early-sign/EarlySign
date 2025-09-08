"""
earlysign.methods
=================

Sequential testing methods and statistical procedures.

This package contains implementations of various sequential testing methods
following the EarlySign event-sourcing framework.

Available methods:
- `group_sequential`: Group sequential testing with spending functions
- `safe_testing`: E-value based safe testing methods

Each method module provides components (Statistic, Criteria, Signaler) that
integrate with the ledger-based event system.
"""
