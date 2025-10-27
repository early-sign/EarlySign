"""Shared enums and lookup tables for group sequential design payloads."""

from enum import Enum
from typing import Dict, Optional, Union


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

    OBF = "obf"
    POCOCK = "pocock"
    HSD = "hsd"


STATISTIC_DEFAULT_SCALES: Dict[StatisticType, Optional[BoundaryScale]] = {
    StatisticType.WALD_Z: BoundaryScale.Z,
    StatisticType.T: None,
    StatisticType.LIKELIHOOD_RATIO: None,
    StatisticType.CUSTOM: None,
}

_SPENDING_FAMILY_ALIASES: Dict[str, SpendingFamily] = {
    "obf": SpendingFamily.OBF,
    "obrien_fleming": SpendingFamily.OBF,
    "o'brien_fleming": SpendingFamily.OBF,
    "pocock": SpendingFamily.POCOCK,
    "hsd": SpendingFamily.HSD,
}


def normalize_spending_family(
    value: Optional[Union[SpendingFamily, str]],
) -> Optional[SpendingFamily]:
    """Coerce user-provided family names (or aliases) to the canonical enum."""

    if value is None:
        return None
    if isinstance(value, SpendingFamily):
        return value
    alias = str(value).strip().lower()
    if alias in _SPENDING_FAMILY_ALIASES:
        return _SPENDING_FAMILY_ALIASES[alias]
    return SpendingFamily(alias)


__all__ = [
    "BindingMode",
    "BoundaryScale",
    "HypothesisStructure",
    "EfficacyStyle",
    "FutilityMode",
    "StatisticType",
    "SpendingFamily",
    "STATISTIC_DEFAULT_SCALES",
    "normalize_spending_family",
]
