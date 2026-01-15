"""Shared enums and lookup tables for group sequential design payloads."""

from enum import Enum
from typing import Dict, Optional


class BindingMode(str, Enum):
    """Binding vs non-binding futility policy."""

    BINDING = "binding"
    NON_BINDING = "non_binding"


class BoundaryScale(str, Enum):
    """Supported boundary/statistic scales."""

    Z = "z"
    B = "bm"


class HypothesisStructure(str, Enum):
    """Global hypothesis geometry."""

    TWO_SIDED_SYMMETRIC = "two_sided_symmetric"
    ONE_SIDED_UPPER = "one_sided_upper"
    ONE_SIDED_LOWER = "one_sided_lower"
    INNER_WEDGE = "inner_wedge"


class EfficacyStyle(str, Enum):
    """Efficacy boundary configuration styles."""

    ALPHA_SPENDING = "alpha_spending"
    SIGNIFICANCE_LEVEL = "significance_level"
    FIXED_THRESHOLD = "fixed_threshold"


class FutilityMode(str, Enum):
    """Futility boundary configuration styles."""

    NONE = "none"
    SYMMETRIC = "symmetric"
    FIXED_THRESHOLD = "fixed_threshold"
    BETA_SPENDING = "beta_spending"
    CUSTOM = "custom"


class StatisticType(str, Enum):
    """Primary test statistics supported by the runtime."""

    WALD_Z = "wald_z"
    T = "t"
    LIKELIHOOD_RATIO = "likelihood_ratio"
    CUSTOM = "custom"


class SpendingFamily(str, Enum):
    """Canonical spending families."""

    OBF = "obrien_fleming"
    POCOCK = "pocock"
    HSD = "hsd"


STATISTIC_DEFAULT_SCALES: Dict[StatisticType, Optional[BoundaryScale]] = {
    StatisticType.WALD_Z: BoundaryScale.Z,
    StatisticType.T: None,
    StatisticType.LIKELIHOOD_RATIO: None,
    StatisticType.CUSTOM: None,
}
