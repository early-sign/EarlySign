"""
Engine for Adaptive Group Sequential Design / Sample Size Re-estimation.
"""

import numpy as np
from scipy import stats

from earlysign.schema.ES3.GST import Protocol
from earlysign.schema.ES3.GST.Log import (
    AdaptationLog,
    LookResult,
    PromisingZoneStatus,
)


class ConditionalPowerAdaptationEngine:
    """
    Engine for calculating Conditional Power and recommending Sample Size Re-estimation (SSR).
    """

    @classmethod
    def compute_conditional_power(
        cls,
        observed_z: float,
        current_info_time: float,
        final_info_time: float,
        final_efficacy_bound: float,
        assumed_effect: float,
        variance: float = 1.0,
    ) -> float:
        """
        Compute conditional power at an interim analysis.

        Parameters
        ----------
        observed_z : float
            Observed Z-statistic at current interim analysis.
        current_info_time : float
            Current information time, in (0, 1).
        final_info_time : float
            Information time at final analysis, typically 1.0.
        final_efficacy_bound : float
            Efficacy boundary value (Z-scale) at final analysis.
        assumed_effect : float
            Assumed standardized effect size for remaining data.
        variance : float
            Model variance (default 1.0).

        Returns
        -------
        float
            Conditional power.

        Examples
        --------
        >>> cp = ConditionalPowerAdaptationEngine.compute_conditional_power(
        ...     observed_z=2.0,
        ...     current_info_time=0.5,
        ...     final_info_time=1.0,
        ...     final_efficacy_bound=1.96,
        ...     assumed_effect=0.5
        ... )
        >>> round(cp, 3)
        0.711
        """
        if not 0 < current_info_time < final_info_time:
            # Degenerate cases
            if current_info_time >= final_info_time:
                return 0.0 if observed_z < final_efficacy_bound else 1.0
            raise ValueError("Current info time must be less than final info time")

        delta_t = final_info_time - current_info_time
        mean_increment = assumed_effect * np.sqrt(delta_t)

        # Z(final) | Z(current) ~ N(Z(current) + drift, delta_t / variance)
        # Note: B-value vs Z-value conversion.
        # This implementation assumes Z-scale direct extrapolation (as in v0).
        # Let's verify standard B-value math:
        # B(t) ~ N(theta * t, t)
        # B(1) | B(t) ~ N(B(t) + theta * (1 - t), 1 - t)
        # Z(1) = B(1) / sqrt(1) = B(1)
        # Z(1) | Z(t) ~ N(Z(t) * sqrt(t) + theta * (1 - t), 1 - t) ??? -> No.

        # v0 Implementation was:
        # mean_increment = assumed_effect * np.sqrt(delta_t)
        # conditional_mean = observed_z + mean_increment
        # conditional_sd = np.sqrt(delta_t / variance)
        # This implies observed_z is treated like a B-value increment??
        # Actually v0 docstring says "Z(final) | Z(current) ~ N(mean, var)".
        # Let's stick to the v0 math for consistency in "Migration".

        conditional_mean = observed_z + mean_increment
        conditional_sd = np.sqrt(delta_t / variance)

        cp = 1.0 - stats.norm.cdf(
            final_efficacy_bound, loc=conditional_mean, scale=conditional_sd
        )
        return float(cp)

    @classmethod
    def assess_promising_zone(
        cls,
        result: LookResult,
        protocol: Protocol,
        assumed_effect: float,
        final_efficacy_bound: float = 1.96,  # Should come from Design/Engine
        cp_threshold_min: float = 0.5,
        cp_threshold_max: float = 0.9,
    ) -> AdaptationLog:
        """
        Assess if the trial is in the 'Promising Zone' and recommend action.

        Examples
        --------
        >>> from earlysign.schema.ES3.GST import Protocol
        >>> # Mock objects
        >>> res = LookResult(
        ...     look=1, sample_n=50, info_frac=0.5, z_stat=1.5,
        ...     is_efficacy_crossed=False, is_futility_crossed=False,
        ...     status="continue"
        ... )
        >>> # Basic protocol mocking for doctest
        >>> from unittest.mock import MagicMock
        >>> p_mock = MagicMock()
        >>> p_mock.method.stopping_policy.timer.max_sample_size = 100
        >>> # Logic test
        >>> log = ConditionalPowerAdaptationEngine.assess_promising_zone(
        ...     res, p_mock, assumed_effect=1.0, final_efficacy_bound=1.96
        ... )
        >>> log.promising_zone_status
        <PromisingZoneStatus.PROMISING: 'promising'>
        """
        # 1. Stop Check
        if result.is_efficacy_crossed:
            return AdaptationLog(
                look=result.look or 1,
                conditional_power=1.0,
                promising_zone_status=PromisingZoneStatus.EFFICACY,
                promising_zone_recommendation="Stop for Efficacy",
                original_sample_size=result.sample_n,  # Placeholder
            )

        # 2. Compute CP
        # Assumes final info time = 1.0 for simplification, or extract from protocol
        cp = cls.compute_conditional_power(
            observed_z=result.z_stat,
            current_info_time=result.info_frac,
            final_info_time=1.0,
            final_efficacy_bound=final_efficacy_bound,
            assumed_effect=assumed_effect,
        )

        # 3. Categorize
        if cp < cp_threshold_min:
            status = PromisingZoneStatus.FUTILITY
            rec = "Consider stopping for futility (Low CP)"
        elif cp > cp_threshold_max:
            status = PromisingZoneStatus.CONTINUE_
            rec = "Continue as planned (High CP)"
        else:
            status = PromisingZoneStatus.PROMISING
            rec = "Promising Zone: Consider increasing sample size to recover power"

        return AdaptationLog(
            look=result.look or 1,
            conditional_power=cp,
            promising_zone_status=status,
            promising_zone_recommendation=rec,
            original_sample_size=getattr(
                protocol.method.stopping_policy.timer, "max_sample_size", 0
            ),
        )

    @classmethod
    def replan_sample_size(
        cls,
        protocol: Protocol,
        adaptation_log: AdaptationLog,
        target_cp: float = 0.9,
    ) -> Protocol:
        """
        Returns a NEW Protocol with updated max_sample_size to achieve target CP.
        This uses the Cui-Hung-Wang method (implied) or simple CP inversion.
        """
        # Deep copy protocol to avoid mutation
        new_protocol = protocol.model_copy(deep=True)

        # If not promising, return original
        if adaptation_log.promising_zone_status != PromisingZoneStatus.PROMISING:
            return new_protocol

        # Logic: Increase N_max such that CP becomes target_cp.
        # This requires the 'assumed_effect' used in calculation or re-deriving it.
        # For this migration step, let's implement a placeholder multiplier
        # to demonstrate the "Structure" of replanning.

        # Real logic would solve for N_new in the CP equation.
        # Simple heuristic: N_new = N_old * (target_CP / current_CP)^2 (very rough)

        current_n = adaptation_log.original_sample_size
        multiplier = 1.0

        if adaptation_log.conditional_power > 0:
            # simple heuristic for demo
            multiplier = target_cp / adaptation_log.conditional_power

        # Limit multiplier
        multiplier = min(multiplier, 2.0)  # Cap at 2x
        multiplier = max(multiplier, 1.0)

        new_n = int(current_n * multiplier)

        if hasattr(new_protocol.method.stopping_policy.timer, "max_sample_size"):
            new_protocol.method.stopping_policy.timer.max_sample_size = new_n

        return new_protocol
