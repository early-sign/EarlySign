from typing import Any, Dict, Literal, Optional

import numpy as np

from earlysign.v1.methods.group_sequential.canonical_dist import (
    CanonicalJointDistribution,
)
from earlysign.v1.methods.group_sequential.protocol import GSTProtocol


class ProtocolDesigner:
    """
    Designer for group sequential protocols.
    Translates scientific intent (alpha, power, delta) into a realized design (boundaries, sample size).
    """

    def __init__(self, cjd: Optional[CanonicalJointDistribution] = None):
        self._cjd = cjd or CanonicalJointDistribution()

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ProtocolDesigner":
        """
        Creates a ProtocolDesigner instance from a configuration dictionary.

        Args:
            config: A dictionary containing initialization parameters.
                Supported keys:
                - model: (str) Model type to use (default: "canonical_gaussian").
                - model_params: (dict) Parameters to pass to the model constructor (e.g., {"rng_seed": 42}).

        Returns:
            An initialized ProtocolDesigner instance.
        """
        model_type = config.get("model", "canonical_gaussian")
        model_params = config.get("model_params", {})

        if model_type == "canonical_gaussian":
            cjd = CanonicalJointDistribution(**model_params)
            return cls(cjd=cjd)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def plan_binomial_ab(
        self,
        alpha: float,
        power: float,
        delta: float,
        k: int,
        p_control: float,
        shape_type: Literal["obrien_fleming", "pocock"] = "obrien_fleming",
        side: int = 1,
        rho: float = 3.0,
    ) -> GSTProtocol:
        """
        Plans a binomial A/B design and returns a fully populated GSTProtocol.

        Args:
            alpha: Type-1 error rate.
            power: Target power (1 - beta).
            delta: Minimum clinically meaningful difference.
            k: Number of looks.
            p_control: Baseline conversion rate (for variance estimation).
            shape_type: Spending function / boundary shape.
            side: Number of sides (1 or 2).
            rho: Parameter for spending function if applicable.

        Returns:
            GSTProtocol with both intent fields AND realized design fields (n_max, boundaries, etc.) populated.
        """
        # Average variance under H0 approx: p_control * (1 - p_control)
        sigma2 = p_control * (1.0 - p_control)
        theta = delta  # difference in proportions

        info_times = np.linspace(1 / k, 1.0, k)

        # 1. Solve for boundary constant c
        # Note: CanonicalJointDistribution currently takes 'shape_type' string.
        # Ideally, this should map from protocol enums, but for now we pass through.
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
        n_max_float = 4 * i_max * sigma2
        n_max = int(np.ceil(n_max_float))

        # Construct the realized protocol
        return GSTProtocol(
            # Intent
            alpha=alpha,
            power=power,
            K=k,
            delta=delta,
            side=side,
            spending_function=shape_type,  # Mapping string directly for now
            rho=rho,
            # Realization
            n_max=n_max,
            milestones=info_times.tolist(),
            boundaries=boundaries,
        )
