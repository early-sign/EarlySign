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

    For a two-arm binomial design with control arm proportion ``p_control`` and
    treatment arm proportion ``p_treatment``, this function computes the
    maximum sample size per arm implied by the supplied standardized drift
    statistic ``drift``.

    When ``control_arm_name == treatment_arm_name``, the design is treated as
    single-arm, and the function returns only a single entry for that arm.

    Parameters
    ----------
    drift:
        Standardized drift parameter for the design.
    p_control:
        Event probability in the control arm.
    p_treatment:
        Event probability in the primary treatment arm.
    allocation_ratios:
        Optional mapping from arm name to allocation ratio relative to the
        control. If ``None``, a 1:1 allocation between control and the
        treatment arm is assumed.
    control_arm_name:
        Name of the control arm key in the returned dictionary.
    treatment_arm_name:
        Name of the primary treatment arm key in the returned dictionary.

    Returns
    -------
    Dict[str, int]
        A mapping from arm name to maximum sample size for that arm.

    Raises
    ------
    ValueError
        If ``p_control`` and ``p_treatment`` are equal, since the resulting
        effect size would be zero and the information statistic undefined.

    Examples
    --------
    A single-arm design (control and treatment share the same name):

    >>> calculate_n_max(drift=1.0, p_control=0.5, p_treatment=0.6, control_arm_name="treatment", treatment_arm_name="treatment")
    {'treatment': 25}

    A two-arm design with 1:1 allocation:

    >>> calculate_n_max(drift=1.0, p_control=0.5, p_treatment=0.6)
    {'control': 50, 'treatment': 50}

    Attempting to use identical control and treatment proportions raises an error:

    >>> calculate_n_max(drift=1.0, p_control=0.5, p_treatment=0.5)
    Traceback (most recent call last):
    ...
    ValueError: p_control and p_treatment must be different to compute maximum sample size; received p_control=0.5 and p_treatment=0.5.
    """
    delta = abs(p_treatment - p_control)
    if delta == 0.0:
        raise ValueError(
            "p_control and p_treatment must be different to compute maximum "
            f"sample size; received p_control={p_control} and p_treatment={p_treatment}."
        )
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
