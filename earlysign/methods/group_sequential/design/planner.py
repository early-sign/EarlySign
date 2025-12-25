from typing import Dict, Literal, Optional
import numpy as np
from scipy.stats import norm, nct

from earlysign.methods.group_sequential.canonical_joint_distribution import CanonicalJointDistribution

TrialType = Literal[
    "normal-mean", "paired", "crossover", "binomial-single", "binomial-ab", "log-rank", "t-test"
]

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
        round_to_k: bool = False,
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
            trial_type: 'normal-mean', 'paired', 'crossover', 'binomial-single', 'binomial-ab'.
            round_to_k: If True, round n_max up to the nearest multiple of k.
            
        Returns:
            Dict containing 'i_max', 'n_max', 'boundaries', and 'n_per_look'.
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

        # 3b. Calculate I_fixed = ( (z_{1-alpha/2} + z_{1-beta}) / theta )^2
        from scipy.stats import norm
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        z_beta = norm.ppf(power)
        i_fixed = ((z_alpha + z_beta) / theta)**2
        
        # 4. Map to sample size n_max
        if trial_type == "normal-mean" or trial_type == "binomial-ab":
            # For 2-arm A/B trial: I = n_total / (4 * sigma^2) => n_total = 4 * sigma^2 * I
            # Here n_reported is n_total if normal-mean, but we often want n_g. 
            # In the previous test implementation for binomial-ab, we used n_g.
            # Let's keep consistency with the existing methods or refine them.
            n_reported = 4 * i_max * sigma2 
        elif trial_type == "paired" or trial_type == "binomial-single":
            # For paired: I = n / sigma^2_diff => n = I * sigma^2_diff
            n_reported = i_max * sigma2
        elif trial_type == "log-rank":
            # For log-rank: I = d / 4 => d = 4 * I (total events)
            # theta is log(HR), sigma2 is usually not needed but we can use it as multiplier if provided
            n_reported = 4.0 * i_max * (sigma2 if sigma2 else 1.0)
        elif trial_type == "crossover":
            # For crossover: I = 2n / sigma^2 => n = I * sigma^2 / 2
            n_reported = i_max * sigma2 / 2.0
        else:
            n_reported = i_max * sigma2

        if round_to_k:
            # If it's a 2-arm trial, n_reported is n_total. n_g = n_total / 2.
            if trial_type in ["normal-mean", "binomial-ab", "t-test"]:
                n_g = n_reported / 2.0
                n_g_rounded = int(np.ceil(n_g / k) * k)
                n_reported = n_g_rounded * 2.0
                n_per_look = n_g_rounded / k
            else:
                n_reported = int(np.ceil(n_reported / k) * k)
                n_per_look = n_reported / k
        else:
            n_per_look = n_reported / k

        res = {
            "i_max": i_max,
            "i_fixed": i_fixed,
            "n_max": n_reported,
            "boundaries": boundaries,
            "n_per_look": n_per_look
        }

        if trial_type == "t-test":
            # For t-test, we might have passed nu_K as sigma2 or similar.
            # J&T Table 3.3 uses nu_K (degrees of freedom at final look).
            # Let's assume sigma2 is used for nu_K here or passed in metadata.
            nu_K = sigma2 # Use sigma2 as a container for nu_K in this context
            approx = self.calculate_t_test_power_approx(alpha, i_max, i_fixed, theta, nu_K)
            res.update(approx)

        return res

    def calculate_t_test_power_approx(
        self,
        alpha: float,
        i_max: float,
        i_fixed: float,
        theta: float,
        df_max: float,
    ) -> Dict[str, float]:
        """
        Calculate power approximations (3.20) and (3.21) for group sequential t-tests.
        """
        # Eq (3.21): Normal approximation
        # phi( theta * sqrt(I_fixed) - z_alpha/2 )
        # Note: I_fixed as defined in plan_design uses normal quantiles.
        z_alpha = norm.ppf(1.0 - alpha / 2.0)
        # Ratio r = i_max / i_fixed
        # Approximation uses drift_actual = theta * sqrt(I_max / r) = theta * sqrt(I_fixed)
        ncp = theta * np.sqrt(i_fixed)
        power_normal = norm.cdf(ncp - z_alpha)
        
        # Eq (3.20): Non-central t approximation
        from scipy.stats import t as t_dist
        t_alpha = t_dist.isf(alpha / 2.0, df_max)
        power_t = nct.sf(t_alpha, df_max, ncp)
        
        return {
            "power_approx_320": power_t,
            "power_approx_321": power_normal
        }
