from typing import Dict, Optional

import numpy as np


def get_standardized_drift(delta: float, sigma: float, n_arms: int = 2) -> float:
    """
    Calculates standardized drift for a continuous test.
    theta = delta / (2 * sigma) for two-arm balanced.
    theta = delta / sigma for one-arm.
    """
    if sigma <= 0:
        raise ValueError(f"Invalid standard deviation (sigma={sigma}).")
    return float(delta / (2.0 * sigma if n_arms == 2 else sigma))


def calculate_n_max(
    drift: float,
    delta: float,
    sigma: float,
    n_arms: int = 2,
    arm_names: Optional[list[str]] = None,
) -> Dict[str, int]:
    """
    Calculates the maximum sample size per arm for a continuous design.
    """
    theta = get_standardized_drift(delta, sigma, n_arms)
    i_max = (drift / theta) ** 2
    n_total = int(np.ceil(i_max))

    if arm_names is None:
        arm_names = ["control", "treatment"] if n_arms == 2 else ["treatment"]

    n_per_arm = int(np.ceil(n_total / n_arms))
    return {name: n_per_arm for name in arm_names}
