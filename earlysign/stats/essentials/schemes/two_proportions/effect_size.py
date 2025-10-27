"""Effect size utilities for two-sample proportion tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from scipy import stats

from earlysign.stats.essentials.schemes.protocols import (
    EffectSizeCalculator,
    FixedDesignEffectCalculator,
)
from earlysign.stats.essentials.schemes.two_proportions.util import (
    compute_standard_error as compute_se_proportions,
)


@dataclass
class ProportionsEffect:
    """Effect specification for binary outcomes."""

    p_control: float = 0.10
    p_treatment: float | None = None
    effect_size: float | None = 0.02
    effect_type: Literal["absolute", "relative", "odds_ratio"] = "absolute"

    def get_treatment_proportion(self) -> float:
        """Resolve treatment proportion based on effect specification."""
        if self.p_treatment is not None:
            return self.p_treatment
        if self.effect_size is None:
            msg = "effect_size must be provided when p_treatment is not set."
            raise ValueError(msg)
        if self.effect_type == "absolute":
            return self.p_control + self.effect_size
        if self.effect_type == "relative":
            return self.p_control * (1 + self.effect_size)
        if self.effect_type == "odds_ratio":
            odds_control = self.p_control / (1 - self.p_control)
            odds_treatment = odds_control * self.effect_size
            return odds_treatment / (1 + odds_treatment)
        msg = f"Unknown effect_type: {self.effect_type}"
        raise ValueError(msg)


@dataclass
class ProportionsSampleSize:
    """Sample size specification for binary outcomes."""

    n_per_analysis: int = 500


@dataclass
class TwoProportionsEffectSizeCalculator(
    EffectSizeCalculator, FixedDesignEffectCalculator
):
    """Effect size calculator for two-proportion Z tests."""

    p_control: float | None = None

    def calculate_sample_size(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.8
    ) -> int:
        """Compute per-group sample size for a fixed design."""
        if self.p_control is None:
            msg = "p_control must be set before calculating sample size."
            raise ValueError(msg)

        p1 = self.p_control + effect_size
        delta = effect_size
        se_alt = np.sqrt(self.p_control * (1 - self.p_control) + p1 * (1 - p1))
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = ((z_alpha + z_beta) * se_alt / delta) ** 2
        return int(np.ceil(n))

    def get_null_value(self) -> float:
        """Return the baseline (null) proportion."""
        if self.p_control is None:
            msg = "p_control must be set before retrieving the null value."
            raise ValueError(msg)
        return self.p_control

    def standardized_effect(
        self,
        effect: ProportionsEffect,
        info_time: float,
        *,
        sample_size: ProportionsSampleSize,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> float:
        """Calculate standardized effect (delta) at the given information fraction."""
        p_control = effect.p_control
        p_treatment = effect.get_treatment_proportion()

        n_total_control = sample_size.n_per_analysis * len(info_times)
        n_control_at_t = int(n_total_control * info_time)
        n_treatment_at_t = int(n_control_at_t * allocation_ratio)

        if n_control_at_t == 0 or n_treatment_at_t == 0:
            return 0.0

        p_pooled = (p_control + p_treatment) / 2.0
        se = compute_se_proportions(
            n_control_at_t, n_treatment_at_t, p_pooled, p_pooled, pooled=True
        )

        if se == 0:
            return 0.0

        return float((p_treatment - p_control) / se)

    def sample_sizes(
        self,
        sample_size: ProportionsSampleSize,
        *,
        allocation_ratio: float,
        info_times: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Return planned sample sizes at each interim analysis."""
        n_total_control = sample_size.n_per_analysis * len(info_times)

        n_control = (n_total_control * info_times).astype(int)
        n_treatment = (n_control * allocation_ratio).astype(int)

        return {
            "n_control": n_control,
            "n_treatment": n_treatment,
            "n_total": n_control + n_treatment,
            "info_fraction": info_times,
        }


__all__ = [
    "ProportionsEffect",
    "ProportionsSampleSize",
    "TwoProportionsEffectSizeCalculator",
]
