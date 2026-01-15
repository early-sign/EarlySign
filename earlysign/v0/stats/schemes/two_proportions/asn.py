"""
Group sequential helpers tailored to two-sample proportion designs.

The utilities in this module adapt the two-proportion setting to the
underlying normal-means ASN calculator by mapping proportions to the
corresponding normal approximation parameters.
"""

import math
from dataclasses import dataclass

from earlysign.v0.methods.group_sequential.spending import (
    SpendingFunction,
)
from earlysign.v0.stats.schemes.two_means.asn import (
    NormalMeansASNCalculator,
)


@dataclass(frozen=True)
class NormalApproximationParameters:
    """Intermediate parameters for approximating two proportions by normal means."""

    effect_size: float
    st_dev: float
    allocation_ratio: float


def _validate_proportion(value: float, label: str) -> None:
    if not (0.0 < value < 1.0):
        raise ValueError(f"{label} must lie in (0, 1), got {value!r}")


def normal_approximation_parameters(
    *,
    p_control: float,
    effect_size: float,
    allocation_ratio: float = 1.0,
) -> NormalApproximationParameters:
    """Return normal approximation parameters for a two-proportion design."""
    _validate_proportion(p_control, "p_control")
    p_treatment = p_control + effect_size
    _validate_proportion(p_treatment, "treatment proportion")

    pooled = 0.5 * (p_control + p_treatment)
    variance = pooled * (1.0 - pooled)
    if variance <= 0.0:
        raise ValueError("pooled variance must be positive")
    if allocation_ratio <= 0.0:
        raise ValueError("allocation_ratio must be positive")

    return NormalApproximationParameters(
        effect_size=float(effect_size),
        st_dev=math.sqrt(variance),
        allocation_ratio=float(allocation_ratio),
    )


def build_asn_calculator(
    *,
    alpha: float,
    beta: float,
    sided: int,
    p_control: float,
    effect_size: float,
    allocation_ratio: float,
    spending: SpendingFunction,
) -> NormalMeansASNCalculator:
    """Factory that returns a normal approximation ASN calculator."""
    params = normal_approximation_parameters(
        p_control=p_control,
        effect_size=effect_size,
        allocation_ratio=allocation_ratio,
    )
    return NormalMeansASNCalculator(
        alpha=alpha,
        beta=beta,
        sided=sided,
        alternative=params.effect_size,
        st_dev=params.st_dev,
        allocation_ratio=params.allocation_ratio,
        spending=spending,
    )
