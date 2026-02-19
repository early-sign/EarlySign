from typing import Dict, Optional

import numpy as np


def get_standardized_drift(p_control: float, p_treatment: float) -> float:
    """
    Calculates standardized drift for a two-arm binomial test with equal allocation.
    theta = delta / sqrt(4 * p_bar * (1 - p_bar))
    """
    delta = abs(p_treatment - p_control)
    p_bar = (p_control + p_treatment) / 2.0
    sigma2 = p_bar * (1.0 - p_bar)
    if sigma2 <= 0:
        raise ValueError(
            f"Invalid pooled proportions (p_bar={p_bar}) resulting in zero variance."
        )
    return float(delta / np.sqrt(4.0 * sigma2))


def calculate_n_max(
    drift: float,
    p_control: float,
    p_treatment: float,
    allocation_ratios: Optional[Dict[str, float]] = None,
    control_arm_name: str = "control",
    treatment_arm_name: str = "treatment",
) -> Dict[str, int]:
    """
    Calculates the maximum sample size per arm for a binomial design.
    """
    delta = abs(p_treatment - p_control)
    sigma2 = p_control * (1.0 - p_control)
    # Information at final look: I_max = (drift / delta)^2
    i_max_stat = (drift / delta) ** 2

    if control_arm_name == treatment_arm_name:
        # Single-Arm Case (n = I_max * sigma2)
        n = int(np.ceil(i_max_stat * sigma2))
        return {treatment_arm_name: n}

    if allocation_ratios is None:
        allocation_ratios = {treatment_arm_name: 1.0}

    primary_ratio = allocation_ratios.get(treatment_arm_name, 1.0)
    # n_c = I_max_stat * sigma2 * (1 + 1/rj)
    n_c = int(np.ceil(i_max_stat * sigma2 * (1 + 1.0 / primary_ratio)))

    n_max_dict = {control_arm_name: n_c}
    for arm, ratio in allocation_ratios.items():
        n_max_dict[arm] = int(np.ceil(n_c * ratio))

    return n_max_dict
