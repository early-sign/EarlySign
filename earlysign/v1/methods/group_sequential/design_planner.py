from typing import Any, Dict, Optional

import numpy as np

from earlysign.v1.methods.group_sequential.canonical_dist import (
    CanonicalJointDistribution,
)


class DesignPlanner:
    """
    Simplified Planner for group sequential designs (Pattern G v1).
    Ported to support full-stack verification.
    """

    def __init__(self, cjd: Optional[CanonicalJointDistribution] = None):
        self._cjd = cjd or CanonicalJointDistribution()

    def plan_binomial_ab(
        self,
        alpha: float,
        power: float,
        p_control: float,
        delta: float,
        k: int,
        shape_type: str = "obrien_fleming",
    ) -> Dict[str, Any]:
        """
        Plans a binomial A/B design.
        """
        # Average variance under H0 approx: p_control * (1 - p_control)
        sigma2 = p_control * (1.0 - p_control)
        theta = delta  # difference in proportions

        info_times = np.linspace(1 / k, 1.0, k)

        # 1. Solve for boundary constant c
        c_val = self._cjd.solve_boundary_constant(
            info_times.tolist(), alpha, shape_type=shape_type
        )

        if shape_type == "pocock":
            c_shape = np.ones(k)
        elif shape_type == "obrien_fleming":
            c_shape = 1.0 / np.sqrt(info_times)
        else:
            raise ValueError(f"Unsupported shape: {shape_type}")

        boundaries = (c_val * c_shape).tolist()

        # 2. Solve for standardized drift delta = theta * sqrt(I_max)
        drift = self._cjd.solve_drift(
            info_times.tolist(), boundaries, target_power=power
        )

        # 3. Calculate I_max = (drift / theta)^2
        i_max = (drift / theta) ** 2

        # 4. Map to sample size n_max (total for both arms)
        # Information I = n_total / (4 * sigma^2) => n_total = 4 * I * sigma^2
        n_max = 4 * i_max * sigma2
        n_per_look = n_max / k

        return {
            "alpha": alpha,
            "power": power,
            "delta": delta,
            "k": k,
            "boundaries": boundaries,
            "i_max": i_max,
            "n_max": int(np.ceil(n_max)),
            "n_per_look": int(np.ceil(n_per_look)),
            "info_times": info_times.tolist(),
        }
