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
    allocation_ratios: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """
    Calculates the maximum sample size per arm for a continuous design.
    """
    theta = get_standardized_drift(delta, sigma, n_arms)
    i_max = (drift / theta) ** 2
    int(np.ceil(i_max))

    if arm_names is None:
        arm_names = ["control", "treatment"] if n_arms == 2 else ["treatment"]

    if allocation_ratios is None:
        n_per_arm = int(np.ceil(i_max / n_arms))
        return {name: n_per_arm for name in arm_names}

    # 2-arm unequal case (r_rel = ratio of treatment / control)
    primary_ratio = (
        allocation_ratios.get(arm_names[1], 1.0) if len(arm_names) > 1 else 1.0
    )
    n_c = int(np.ceil(i_max * (1 + 1.0 / primary_ratio) / 2.0))  # approx scaling
    # TODO: more precise formula based on I_max(r)
    n_max_dict = {arm_names[0]: n_c}
    for i, name in enumerate(arm_names[1:], 1):
        ratio = allocation_ratios.get(name, 1.0)
        n_max_dict[name] = int(np.ceil(n_c * ratio))

    return n_max_dict
