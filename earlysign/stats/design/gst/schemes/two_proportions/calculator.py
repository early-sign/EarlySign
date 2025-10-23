"""Sample size calculator for two-proportion z-test (moved from scenarios.py)."""

from dataclasses import dataclass

import numpy as np

from earlysign.stats.common.protocols import EffectSizeCalculator


@dataclass
class TwoProportionsCalculator(EffectSizeCalculator):
    """Sample size calculator for two-proportion z-test.

    Attributes:
        p_control: Control proportion
    """

    p_control: float

    def calculate_sample_size(
        self, effect_size: float, alpha: float = 0.05, power: float = 0.8
    ) -> int:
        from scipy import stats

        p1 = self.p_control + effect_size
        delta = effect_size
        se_alt = np.sqrt(self.p_control * (1 - self.p_control) + p1 * (1 - p1))
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = ((z_alpha + z_beta) * se_alt / delta) ** 2
        return int(np.ceil(n))

    def get_null_value(self) -> float:
        return self.p_control
