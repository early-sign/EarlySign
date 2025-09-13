"""
Statistical methods and theories for sequential testing.

This module follows ADR-002 architecture with clear separation between
generic methods and scheme-specific implementations:

1. **Common** (earlysign.stats.common):
   Generic, reusable statistical methods and algorithms that are independent
   of specific problem domains. Includes group sequential testing foundations,
   e-processes, and basic statistical computations.

2. **Schemes** (earlysign.stats.schemes):
   Problem-specific implementations that apply the generic methods from `common`
   to particular experimental scenarios (two-proportions testing, survival analysis, etc.).

This architecture maximizes code reuse, ensures clear separation of concerns,
and makes it easy to add new statistical schemes by composing existing generic methods.

Example:
--------
>>> # Generic method (reusable across schemes)
>>> from earlysign.stats.common.group_sequential import lan_demets_spending
>>> alpha_spent = lan_demets_spending(0.05, 0.5, "obf")

>>> # Scheme-specific application
>>> from earlysign.stats.schemes.two_proportions.group_sequential import WaldZStatistic
>>> statistic = WaldZStatistic()
"""
