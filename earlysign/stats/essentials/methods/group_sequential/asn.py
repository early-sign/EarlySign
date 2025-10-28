"""ASN calculator protocol for group-sequential methods.

This module centralises the :class:`ASNCalculator` Protocol so various
optimizers and utilities can depend on a stable interface.
"""

from typing import Any, Protocol


class ASNCalculator(Protocol):
    """Protocol describing the minimal ASN-calculator surface used by
    timing optimizers such as :class:`MinimizeASNOptimizer`.

    Implementations are expected to provide an ``evaluate`` method that
    accepts a schedule of information fractions and returns the expected
    sample size, a.k.a. average sample number (ASN). The optimizer also
    relies on several internal helpers (validation, stagewise alpha -> z-boundary conversion and
    a ``_n_max_from_power`` method) as well as a few public attributes
    used for computing stagewise means.
    """

    def evaluate(self, info: Any) -> float: ...

    def _validate_information_rates(self, info: Any) -> Any: ...

    def _per_stage_alpha(self, rates: Any) -> Any: ...

    def _z_boundaries(self, per_stage_alpha: Any) -> Any: ...

    def _n_max_from_power(self) -> float: ...

    # attributes accessed by timing workflows
    st_dev: float
    allocation_ratio: float
    alternative: float
