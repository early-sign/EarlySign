from typing import Any, Dict, Optional

import numpy as np
from scipy.stats import norm

from earlysign.methods.group_sequential.canonical_joint_distribution import (
    CanonicalJointDistribution,
)
from earlysign.methods.group_sequential.spending import SpendingFunction


class OperatingCharacteristicEvaluator:
    """
    Evaluator for the operating characteristics (Type I error and Power) of
    a group sequential procedure. Supports evaluating performance under
    planned or mismatched (robustness check) information schedules.
    """

    def __init__(self, cjd: Optional[CanonicalJointDistribution] = None):
        self._cjd = cjd or CanonicalJointDistribution()

    def evaluate_rejection_probability(
        self,
        n: np.ndarray,
        boundaries: np.ndarray,
        theta: float,
        sigma2: float,
        trial_type: str = "normal-mean",
    ) -> float:
        """
        Evaluates the rejection probability for a given design and effect size.

        Args:
            n: Array of sample sizes at each look.
            boundaries: Array of critical values at each look.
            theta: Clinical effect size.
            sigma2: Variance parameter.
            trial_type: 'normal-mean', 'paired', 'crossover', 'binomial-single', 'binomial-ab'.
        """
        if trial_type in ["normal-mean", "binomial-ab"]:
            # I = n_total / (4 * sigma2)
            # If n is n_g (per group), then n_total = 2 * n.
            # I = 2n / (4 * sigma2) = n / (2 * sigma2)
            info = n / (2.0 * sigma2)
        elif trial_type in ["paired", "binomial-single"]:
            # I = n / sigma2
            info = n / sigma2
        elif trial_type == "crossover":
            # I = 2n / sigma2
            info = 2.0 * n / sigma2
        else:
            info = n / sigma2

        t = (info / info[-1]).tolist()
        drift = theta * np.sqrt(info[-1])
        return self._cjd.compute_rejection_probability(
            t, boundaries.tolist(), drift=drift
        )

    def evaluate_rejection_probability_canonical(
        self,
        info_times: np.ndarray,
        boundaries: np.ndarray,
        drift: float = 0.0,
    ) -> float:
        """
        Evaluates rejection probability for a given canonical information schedule and boundaries.

        Args:
            info_times: Actual cumulative information fractions (t_1, ..., t_K).
            boundaries: Z-scale boundaries at each look.
            drift: Standardized drift delta = theta * sqrt(I_max).
        """
        return self._cjd.compute_rejection_probability(
            info_times.tolist(), boundaries.tolist(), drift=drift
        )

    def evaluate_performance(
        self,
        actual_info_times: np.ndarray,
        planned_boundaries: np.ndarray,
        drift: float = 0.0,
    ) -> Dict[str, float]:
        """
        Evaluates rejection probability for a given info schedule and boundaries.
        (Kept for backward compatibility or simple usage)
        """
        prob = self.evaluate_rejection_probability_canonical(
            actual_info_times, planned_boundaries, drift=drift
        )
        return {"rejection_probability": prob}

    def evaluate_t_test_rejection_probability(
        self,
        info_times: np.ndarray,
        boundaries: np.ndarray,
        nu: np.ndarray,
        drift: float = 0.0,
        tails: int = 2,
    ) -> float:
        """
        Evaluates the rejection probability of a group sequential t-test.

        Args:
            info_times: Actual information fractions.
            boundaries: t-scale boundaries at each look.
            nu: Degrees of freedom at each look.
            drift: Standardized drift delta = theta * sqrt(I_max).
            tails: 1 or 2.
        """
        t = np.asarray(info_times)
        b = np.asarray(boundaries)
        k = len(t)
        n_sims = self._cjd.n_sims * 2
        rng = self._cjd._rng

        # 1. Generate joint Z distribution
        # Means: E(Z_k) = delta * sqrt(t_k)
        means = drift * np.sqrt(t)
        cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
        np.fill_diagonal(cov, 1.0)
        z_sims = rng.multivariate_normal(means, cov, size=n_sims)

        # 2. Generate joint chi-square distribution for variance estimates
        # nu_k * S_k^2 / sigma^2 ~ chi2_{nu_k}
        # Increments in SSE are independent.
        chi2_sims = np.zeros((n_sims, k))
        current_chi2 = np.zeros(n_sims)
        for i in range(k):
            df_inc = nu[i] - (nu[i - 1] if i > 0 else 0)
            if df_inc > 0:
                current_chi2 += rng.chisquare(df_inc, size=n_sims)
            chi2_sims[:, i] = current_chi2

        # 3. Compute T statistics: T_k = Z_k / sqrt(chi2_k / nu_k)
        t_sims = z_sims / np.sqrt(chi2_sims / nu)

        if tails == 2:
            rejected = np.any(np.abs(t_sims) > b, axis=1)
        else:
            rejected = np.any(t_sims > b, axis=1)

        return float(np.mean(rejected))

    def evaluate_design_characteristics(
        self,
        info_times: np.ndarray,
        spending: SpendingFunction,
        target_power: float = 0.8,
        alpha: float = 0.05,
        tails: int = 2,
    ) -> Dict[str, Any]:
        """Compute boundaries, inflation factor R_LD, and ASN for a spending design.

        Args:
            info_times: Planned information fractions.
            spending: Spending function.
            target_power: Target power at drift eta.
            alpha: Significance level.
            tails: 1 or 2.

        Returns:
            Dictionary containing:
            - boundaries: Z-scale critical values.
            - eta: Required drift for target power.
            - r_ld: Inflation factor (eta / eta_fixed)^2.
            - asn_looks: Expected look number at various drift levels.
        """
        # 1. Solve boundaries
        boundaries = self._cjd.solve_boundaries_from_spending(
            info_times.tolist(), spending, tails=tails
        )

        # 2. Solve for drift eta (the drift that gives target power)
        eta = self._cjd.solve_drift(
            info_times.tolist(),
            boundaries.tolist(),
            target_power=target_power,
            tails=tails,
        )

        # 3. Calculate inflation factor R_LD
        # eta_fixed = Phi^-1(1 - alpha/tails) + Phi^-1(power)
        if tails == 2:
            eta_fixed = norm.ppf(1 - alpha / 2.0) + norm.ppf(target_power)
        else:
            eta_fixed = norm.ppf(1 - alpha) + norm.ppf(target_power)
        r_ld = (eta / eta_fixed) ** 2

        # 4. Calculate ASN (Expected Look) for various drift levels
        # Usually reported for theta=0, 0.5delta, delta, 1.5delta
        # delta = eta
        asn_0 = self._cjd.evaluate_asn(
            info_times.tolist(), boundaries.tolist(), drift=0.0, tails=tails
        )
        asn_05delta = self._cjd.evaluate_asn(
            info_times.tolist(), boundaries.tolist(), drift=0.5 * eta, tails=tails
        )
        asn_delta = self._cjd.evaluate_asn(
            info_times.tolist(), boundaries.tolist(), drift=eta, tails=tails
        )
        asn_15delta = self._cjd.evaluate_asn(
            info_times.tolist(), boundaries.tolist(), drift=1.5 * eta, tails=tails
        )

        return {
            "boundaries": boundaries,
            "eta": eta,
            "r_ld": r_ld,
            "asn_looks": {
                "0": float(asn_0),
                "0.5delta": float(asn_05delta),
                "delta": float(asn_delta),
                "1.5delta": float(asn_15delta),
            },
        }

    def evaluate_mismatched_design(
        self,
        actual_info: np.ndarray,
        planned_i_max: float,
        spending: SpendingFunction,
        theta: float = 0.0,
        tails: int = 2,
    ) -> Dict[str, Any]:
        """
        Evaluate a design where the actual information sequence is unplanned.
        Maintains overall Type I error by forcing final look to spend all alpha.

        Args:
            actual_info: Absolute information values at each actual look.
            planned_i_max: The maximum information I_max used to define the spending function.
            spending: The spending function object.
            theta: The effect size for power evaluation.
            tails: 1 or 2.

        Returns:
            Dictionary containing boundaries and rejection_probability.
        """
        t_actual = np.asarray(actual_info)
        t_spending = t_actual / planned_i_max

        # Determine spending at each look
        cum_alpha = spending.cumulative(t_spending)
        # Lan-DeMets robustness: last look MUST spend remaining alpha to maintain level
        cum_alpha[-1] = spending.alpha

        # Canonical info times for boundary solver
        t_canonical = (t_actual / t_actual[-1]).tolist()

        boundaries = self._cjd.solve_boundaries_from_cumulative_alpha(
            t_canonical, cum_alpha.tolist(), tails=tails
        )

        drift = theta * np.sqrt(t_actual[-1])
        power = self._cjd.compute_rejection_probability(
            t_canonical, boundaries.tolist(), drift=drift, tails=tails
        )

        return {
            "boundaries": boundaries,
            "rejection_probability": power,
            "actual_final_info": t_actual[-1],
        }

    def evaluate_dual_scale_design(
        self,
        info_sequence: np.ndarray,
        spending_fractions: np.ndarray,
        spending: SpendingFunction,
        theta: float = 0.0,
        tails: int = 2,
    ) -> Dict[str, Any]:
        """
        Evaluate a design where spending and info are on different scales (e.g. calendar time vs deaths).

        Args:
            info_sequence: Absolute information values at each look.
            spending_fractions: Fractions of max spending resource (e.g. T/T_max) at each look.
            spending: The spending function object.
            theta: Effect size for power evaluation.
            tails: 1 or 2.

        Returns:
            Dictionary with boundaries and rejection_probability.
        """
        t_info = np.asarray(info_sequence)
        t_spend_frac = np.asarray(spending_fractions)

        # 1. Determine target cumulative alpha at each look
        cum_alpha = spending.cumulative(t_spend_frac)

        # 2. Canonical info times for the joint Z distribution (relative to terminal look)
        t_canonical = (t_info / t_info[-1]).tolist()

        # 3. Solve for boundaries
        boundaries = self._cjd.solve_boundaries_from_cumulative_alpha(
            t_canonical, cum_alpha.tolist(), tails=tails
        )

        # 4. Evaluate rejection probability (power) at the final information level
        drift = theta * np.sqrt(t_info[-1])
        power = self._cjd.compute_rejection_probability(
            t_canonical, boundaries.tolist(), drift=drift, tails=tails
        )

        return {
            "boundaries": boundaries,
            "rejection_probability": power,
            "actual_final_info": t_info[-1],
        }
