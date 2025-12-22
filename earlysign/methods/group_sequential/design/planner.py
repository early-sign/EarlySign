from typing import Dict, Literal, Optional
import numpy as np

from earlysign.methods.group_sequential.canonical_joint_distribution import CanonicalJointDistribution

TrialType = Literal["normal-mean", "paired", "crossover"]

class DesignPlanner:
    """
    Planner for group sequential designs based on the canonical joint distribution.
    Determines the required information (I_max) and sample size (n_max) to achieve 
    target operating characteristics (alpha, power) for clinical trial scenarios.
    """

    def __init__(self, cjd: Optional[CanonicalJointDistribution] = None):
        self._cjd = cjd or CanonicalJointDistribution()

    def plan_design(
        self,
        alpha: float,
        power: float,
        theta: float,
        sigma2: float,
        k: int,
        shape_type: str,
        trial_type: TrialType = "normal-mean",
    ) -> Dict[str, float]:
        """
        Plans a sequential design by calculating I_max and n_max.
        
        Args:
            alpha: Type I error rate.
            power: Target power (1 - beta).
            theta: Clinical effect size (e.g., difference in means).
            sigma2: Variance parameter (meaning depends on trial_type).
            k: Number of looks.
            shape_type: 'pocock', 'obrien_fleming', or 'wang_tsiatis'.
            trial_type: 'normal-mean', 'paired', or 'crossover'.
            
        Returns:
            Dict containing 'i_max', 'n_max', and 'boundaries'.
        """
        info_times = np.linspace(1/k, 1.0, k)
        
        # 1. Solve for boundary constant c
        c_val = self._cjd.solve_boundary_constant(info_times, alpha, shape_type=shape_type)
        
        if shape_type == "pocock":
            c_shape = np.ones(k)
        elif shape_type == "obrien_fleming":
            c_shape = 1.0 / np.sqrt(info_times)
        elif shape_type == "wang_tsiatis":
            c_shape = info_times ** (-0.25)
        else:
            raise ValueError(f"Unknown shape: {shape_type}")
            
        boundaries = c_val * c_shape
        
        # 2. Solve for standardized drift delta = theta * sqrt(I_max)
        drift = self._cjd.solve_drift(info_times, boundaries, target_power=power)
        
        # 3. Calculate I_max = (drift / theta)^2
        i_max = (drift / theta)**2
        
        # 4. Map to sample size n_max
        if trial_type == "normal-mean":
            # For 2-arm A/B trial: I = n_total / (4 * sigma^2) => n_total = 4 * sigma^2 * I
            n_reported = 4 * i_max * sigma2
        elif trial_type == "paired":
            # For paired: I = n / sigma^2_diff => n = I * sigma^2_diff
            n_reported = i_max * sigma2
        elif trial_type == "crossover":
            # For crossover: I = 2n / sigma^2 => n = I * sigma^2 / 2
            n_reported = i_max * sigma2 / 2.0
        else:
            n_reported = i_max * sigma2

        return {
            "i_max": i_max,
            "n_max": n_reported,
            "boundaries": boundaries
        }
