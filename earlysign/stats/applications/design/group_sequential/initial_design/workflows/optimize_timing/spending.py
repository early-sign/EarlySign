"""
Helper utilities for configuring spending strategies in timing workflows.
"""

from dataclasses import dataclass

from earlysign.stats.essentials.methods.group_sequential.spending import (
    HSDSpending,
    OBFSpending,
    PocockSpending,
    SpendingFunction,
)

SpendingStrategy = SpendingFunction


@dataclass(frozen=True)
class SpendingConfig:
    """Configuration for constructing an alpha-spending strategy."""

    alpha: float
    sided: int = 1
    family: str = "obrien_fleming"
    pocock_alpha: float | None = None
    hsd_gamma: float = -4.0

    def build(self) -> SpendingStrategy:
        family = self.family.lower()
        if family == "pocock":
            return PocockSpending(alpha=self.pocock_alpha or self.alpha)
        if family == "hsd":
            return HSDSpending(alpha=self.alpha, gamma=self.hsd_gamma)
        if family == "obrien_fleming":
            return OBFSpending(alpha=self.alpha, sided=self.sided)
        raise ValueError(f"Unsupported spending family: {self.family!r}")
