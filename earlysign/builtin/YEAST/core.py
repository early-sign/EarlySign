import numpy as np
from scipy import stats

from earlysign.schema.ES3.YEAST import Protocol


class BoundaryModel:
    """Core mathematical model for YEAST Boundary."""

    @staticmethod
    def calculate_boundary_value(protocol: Protocol) -> float:
        """
        Calculates the YEAST boundary value.

        B = z_{alpha/2} * sqrt(N_max * V_N)

        Args:
            protocol: The YEAST Protocol containing method parameters.

        Returns:
            The calculated boundary value.
        """
        method = protocol.method
        alpha = method.significance_level
        n_max = method.expected_num_observations
        estimated_variance = method.estimated_variance

        # Two-sided critical value (using upper tail)
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # b^* = z_{1 - \alpha/2} \sqrt{N \hat{V}_N}
        boundary_value = z_crit * np.sqrt(n_max * estimated_variance)
        return float(boundary_value)


class TrajectoryModel:
    """Core mathematical model for YEAST Trajectory."""

    @staticmethod
    def calculate_trajectory(n_c: int, n_t: int, diff: float) -> float:
        """
        Calculates the standardized trajectory: S_n / sqrt(n_effective)

        where n_effective = 2 / (1/n_c + 1/n_t)

        Args:
            n_c: Sample size of control arm.
            n_t: Sample size of treatment arm.
            diff: Raw difference (Successes_T - Successes_C) or similar.

        Returns:
            Standardized trajectory value.
        """
        if n_c > 0 and n_t > 0:
            n_eff = 2.0 / (1.0 / n_c + 1.0 / n_t)
            # Avoid division by zero if n_eff is zero (though minimal n checks usually prevent this)
            if n_eff <= 0:
                return 0.0
            return float(diff / (n_eff**0.5))
        return 0.0
