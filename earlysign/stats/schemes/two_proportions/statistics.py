"""
earlysign.stats.schemes.two_proportions.statistics
==================================================

Statistical components for two-proportions testing - Compatibility Module.

This module maintains backwards compatibility while the codebase transitions
to the new ADR-002 structure. All components have been reorganized into
focused modules:

- Common utilities: .common
- Group Sequential Testing: .group_sequential
- Safe Testing (E-processes): .e_process

New code should import directly from the specific modules rather than this
compatibility layer.

Migration Guide:
---------------
OLD: from earlysign.stats.schemes.two_proportions.statistics import WaldZStatistic
NEW: from earlysign.stats.schemes.two_proportions.group_sequential import WaldZStatistic

OLD: from earlysign.stats.schemes.two_proportions.statistics import BetaBinomialEValue
NEW: from earlysign.stats.schemes.two_proportions.e_process import BetaBinomialEValue
"""

# Backwards compatibility imports - Group Sequential Testing
from earlysign.stats.schemes.two_proportions.group_sequential import (
    WaldZStatistic,
    LanDeMetsBoundary,
    PeekSignaler,
    AdaptiveGSTBoundary,
)

# Backwards compatibility imports - Safe Testing
from earlysign.stats.schemes.two_proportions.e_process import (
    BetaBinomialEValue,
    SafeThreshold,
    SafeSignaler,
)

# Backwards compatibility imports - Common utilities
from earlysign.stats.schemes.two_proportions.common import (
    WaldZPayload,
    GstBoundaryPayload,
    TwoPropObsBatch,
    reduce_counts,
    get_latest_statistic,
    get_latest_criteria,
    TwoProportionsSampleTracker,
)

# Deprecated - will be removed in future versions
__all__ = [
    # Group Sequential Testing
    "WaldZStatistic",
    "LanDeMetsBoundary",
    "PeekSignaler",
    "AdaptiveGSTBoundary",
    # Safe Testing
    "BetaBinomialEValue",
    "SafeThreshold",
    "SafeSignaler",
    # Common utilities
    "WaldZPayload",
    "GstBoundaryPayload",
    "TwoPropObsBatch",
    "reduce_counts",
    "get_latest_statistic",
    "get_latest_criteria",
    "TwoProportionsSampleTracker",
]
