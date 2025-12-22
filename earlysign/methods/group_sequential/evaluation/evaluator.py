from typing import Dict, List, Optional, Union
import numpy as np

from earlysign.methods.group_sequential.canonical_joint_distribution import CanonicalJointDistribution

class OperatingCharacteristicEvaluator:
    """
    Evaluator for the operating characteristics (Type I error and Power) of 
    a group sequential procedure. Supports evaluating performance under 
    planned or mismatched (robustness check) information schedules.
    """

    def __init__(self, cjd: Optional[CanonicalJointDistribution] = None):
        self._cjd = cjd or CanonicalJointDistribution()

    def evaluate_performance(
        self,
        planned_info_times: np.ndarray,
        actual_info_times: np.ndarray,
        planned_boundaries: np.ndarray,
        theta_for_power: float = 0.0,
        i_max_planned: Optional[float] = None,
        i_max_actual: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Evaluates Type I error and Power when the actual info schedule varies.
        
        Args:
            planned_info_times: Info times used to derive boundaries.
            actual_info_times: Info times at which the procedure is evaluated.
            planned_boundaries: The critical values used at each look.
            theta_for_power: Effect size for power evaluation.
            i_max_planned: Reference maximum information.
            i_max_actual: Attained maximum information.
        """
        alpha_actual = self._cjd.compute_rejection_probability(
            actual_info_times, planned_boundaries, drift=0.0
        )
        return {"alpha_actual": alpha_actual}

    def evaluate_table31_robustness(
        self,
        planned_n: np.ndarray,
        actual_n: np.ndarray,
        alpha: float,
        spending_family: str,
        theta: float,
        var_diff: float
    ) -> Dict[str, float]:
        """Evaluates Type I error and Power for Table 3.1 robustness scenarios."""
        i_plan = planned_n / var_diff
        t_plan = i_plan / i_plan[-1]
        
        c_val = self._cjd.solve_boundary_constant(t_plan, alpha, shape_type=spending_family)
        
        if spending_family == "pocock":
            c_shape = np.ones_like(t_plan)
        elif spending_family == "obrien_fleming":
            c_shape = 1.0 / np.sqrt(t_plan)
        elif spending_family == "wang_tsiatis":
            c_shape = t_plan ** (-0.25)
        else:
            raise ValueError(f"Unknown shape: {spending_family}")
            
        boundaries = c_val * c_shape
        
        i_actual = actual_n / var_diff
        t_actual = i_actual / i_actual[-1]
        
        alpha_actual = self._cjd.compute_rejection_probability(t_actual, boundaries, drift=0.0)
        
        drift_actual = theta * np.sqrt(i_actual[-1])
        power_actual = self._cjd.compute_rejection_probability(t_actual, boundaries, drift=drift_actual)
        
        return {
            "alpha_actual": alpha_actual,
            "power_actual": power_actual
        }

    def evaluate_table32_robustness(
        self,
        k: int,
        alpha: float,
        planned_power: float,
        spending_family: str,
        pi: float,
        r: float
    ) -> Dict[str, float]:
        """Evaluates Type I error and Power for Table 3.2 robustness scenarios."""
        t_plan = np.linspace(1/k, 1.0, k)
        c_val = self._cjd.solve_boundary_constant(t_plan, alpha, shape_type=spending_family)
        
        if spending_family == "pocock":
            c_shape = np.ones(k)
        else:
            c_shape = 1.0 / np.sqrt(t_plan)
            
        boundaries = c_val * c_shape
        drift_planned = self._cjd.solve_drift(t_plan, boundaries, target_power=planned_power)
        
        ks = np.arange(1, k + 1)
        i_actual_fractions = pi * (ks/k)**r
        t_actual = i_actual_fractions / i_actual_fractions[-1]
        
        alpha_actual = self._cjd.compute_rejection_probability(t_actual, boundaries, drift=0.0)
        
        drift_actual = drift_planned * np.sqrt(pi)
        power_actual = self._cjd.compute_rejection_probability(t_actual, boundaries, drift=drift_actual)
        
        return {
            "alpha_actual": alpha_actual,
            "power_actual": power_actual
        }
