"""
earlysign.methods.safe_testing
==============================

Safe testing methods based on e-values and e-processes for anytime-valid inference.

This module implements various safe testing approaches that provide anytime-valid statistical
inference without traditional alpha-spending or multiple comparison corrections.

Components:
- `BetaBinomialEValue`: Beta-binomial e-process for two-proportion comparisons
- `SafeThreshold`: Anytime-valid thresholds for e-values
- `SafeSignaler`: Decision logic for safe testing procedures

References:
- Grünwald, P., de Heide, R., & Koolen, W. (2020). Safe testing. arXiv:1906.07801.
- Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2023). Game-theoretic statistics and safe anytime-valid inference.

Doctest (basic usage):
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> L = PolarsLedger()
>>> # Ingest observation
>>> L.write_event(time_index="t1", namespace=Namespace.OBS, kind="observation",
...               experiment_id="exp#1", step_key="s1",
...               payload_type="TwoPropObsBatch", payload={"nA":20,"nB":20,"mA":2,"mB":8})
>>> # Compute e-value (requires scipy for beta functions)
>>> BetaBinomialEValue(alpha_prior=1, beta_prior=1).step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
>>> # Check for significance
>>> SafeSignaler(threshold=20.0).step(L, "exp#1", "s1", "t1")  # doctest: +SKIP
"""

from __future__ import annotations

# Import all public components
from earlysign.methods.safe_testing.two_proportions import (
    BetaBinomialEValue,
    SafeThreshold,
    SafeSignaler,
)

__all__ = [
    "BetaBinomialEValue",
    "SafeThreshold",
    "SafeSignaler",
]
