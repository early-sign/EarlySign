"""
Engine for Adaptive Group Sequential Design / Sample Size Re-estimation.
"""

import numpy as np
from scipy import stats

import earlysign.schema.ES3.GST as GST
from earlysign.methods.group_sequential.shared.design_utils import (
    get_final_efficacy_boundary,
    get_standardized_drift,
)
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
        if current_info_time >= final_info_time:
            # Degenerate cases: At or beyond final analysis
            return 1.0 if observed_z >= final_efficacy_bound else 0.0

        delta_t = final_info_time - current_info_time
        mean_increment = assumed_effect * np.sqrt(delta_t)

        # Standard Z-scale extrapolation (B-value equivalent)

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
        cp_threshold_min: float = 0.5,
        cp_threshold_max: float = 0.9,
    ) -> AdaptationLog:
        """
        Assess if the trial is in the 'Promising Zone' and recommend action.

        >>> import earlysign.schema.ES3.Base as ES3_BASE
        >>> from earlysign.schema.ES3.GST import (
        ...     Protocol, MethodSpec, StoppingPolicySpec, OBrienFlemingStrategy,
        ...     TaskSpec, HypothesisSpec, BinaryEffectSize, EquidistantSchedule,
        ...     TwoArmBinomialZ, SampleSizeTimer, EqualityHypothesis,
        ...     CanonicalGaussianModel
        ... )
        >>> from earlysign.methods.group_sequential.execution.sample_size_reestimation import ConditionalPowerAdaptationEngine
        >>> from earlysign.schema.ES3.GST.Log import LookResult
        >>> # Mock results
        >>> res = LookResult(
        ...     look=1, sample_n=50, info_frac=0.5, z_stat=2.0,
        ...     is_efficacy_crossed=False, is_futility_crossed=False,
        ...     status="continue"
        ... )
        >>> p = Protocol(
        ...     name="Binomial SSR Protocol",
        ...     task=TaskSpec(
        ...         arms=ES3_BASE.TwoArmComparison(control_arm_name="control", treatment_arm_name="treatment"),
        ...         response_type="binary",
        ...         hypotheses=HypothesisSpec(
        ...             h_null_description="H0",
        ...             h_alt_description="H1",
        ...             test_logic=EqualityHypothesis(),
        ...             target_effect=BinaryEffectSize(proportions={"control": 0.3, "treatment": 0.5})
        ...         )
        ...     ),
        ...     method=MethodSpec(
        ...         stopping_policy=StoppingPolicySpec(
        ...             statistic=TwoArmBinomialZ(variance_estimation="pooled"),
        ...             strategy=OBrienFlemingStrategy(
        ...                 alpha=0.05, sided="two",
        ...                 statistical_model=CanonicalGaussianModel()
        ...             ),
        ...             timer=SampleSizeTimer(
        ...                 unit="individuals",
        ...                 max_sample_size={"control": 50, "treatment": 50}
        ...             ),
        ...             schedule=EquidistantSchedule(n_looks=2)
        ...         )
        ...     )
        ... )
        >>> # Logic test
        >>> log = ConditionalPowerAdaptationEngine.assess_promising_zone(
        ...     res, p
        ... )
        >>> log.promising_zone_status
        <PromisingZoneStatus.PROMISING: 'promising'>
        """
        # Internalize stats derivation
        theta = get_standardized_drift(protocol)
        final_efficacy_bound = get_final_efficacy_boundary(protocol)
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
            assumed_effect=theta,
        )
        if cp < cp_threshold_min:
            status = PromisingZoneStatus.FUTILITY
            rec = "Consider stopping for futility (Low CP)"
        elif cp > cp_threshold_max:
            status = PromisingZoneStatus.CONTINUE_
            rec = "Continue as planned (High CP)"
        else:
            status = PromisingZoneStatus.PROMISING
            rec = "Promising Zone: Consider increasing sample size to recover power"

        # Calculate total original sample size
        n_max_dict = getattr(
            protocol.method.stopping_policy.timer, "max_sample_size", {}
        )
        original_n = sum(n_max_dict.values()) if isinstance(n_max_dict, dict) else 0

        return AdaptationLog(
            look=result.look or 1,
            conditional_power=cp,
            promising_zone_status=status,
            promising_zone_recommendation=rec,
            original_sample_size=original_n,
        )

    @classmethod
    def check_and_adapt(
        cls,
        result: LookResult,
        protocol: Protocol,
        cp_threshold_min: float = 0.5,
        cp_threshold_max: float = 0.9,
        target_cp: float = 0.9,
    ) -> AdaptationLog:
        """
        Convenience method that assesses the promising zone and recalculates
        the sample size if needed.
        """
        log = cls.assess_promising_zone(
            result, protocol, cp_threshold_min, cp_threshold_max
        )

        if log.promising_zone_status == PromisingZoneStatus.PROMISING:
            new_proto = cls.replan_sample_size(protocol, log, result, target_cp)
            new_n_dict = getattr(
                new_proto.method.stopping_policy.timer, "max_sample_size", {}
            )
            log.recommended_sample_size = (
                sum(new_n_dict.values()) if isinstance(new_n_dict, dict) else None
            )

        return log

    @classmethod
    def replan_sample_size(
        cls,
        protocol: Protocol,
        adaptation_log: AdaptationLog,
        look_result: LookResult,
        target_cp: float = 0.9,
    ) -> Protocol:
        """
        Returns a NEW Protocol with updated max_sample_size to achieve target CP.
        This uses the Cui-Hung-Wang method for Sample Size Re-estimation.
        """
        # Deep copy protocol to avoid mutation
        new_protocol = protocol.model_copy(deep=True)

        # If not promising, return original
        if adaptation_log.promising_zone_status != PromisingZoneStatus.PROMISING:
            return new_protocol

        # Extract parameters for inversion from explicit arguments
        z_t = look_result.z_stat
        t = look_result.info_frac
        theta = get_standardized_drift(protocol)
        c = get_final_efficacy_boundary(protocol)
        n_old = adaptation_log.original_sample_size

        # Scaling theta by sqrt(n_old) to match the Z-stat scale
        theta_total = theta * np.sqrt(n_old)

        if t >= 1.0 or theta <= 0:
            return new_protocol

        # Type narrowing
        z_t_val: float = z_t
        t_val: float = t
        c_val: float = c
        n_old_val: int = n_old

        # Cui-Hung-Wang / CP Inversion Logic:
        # We want P(sqrt(t)Z_t + sqrt(1-t)Z_rem' >= c) = target_cp
        # Z_rem' ~ N(theta * sqrt(r(1-t)), 1)
        # Solve for r (inflation factor for remaining sample size)

        # Z_needed from remaining data (independent of r) to reach c
        z_needed_rem = (c_val - np.sqrt(t_val) * z_t_val) / np.sqrt(1 - t_val)

        # z_target = stats.norm.ppf(target_cp)
        # Equation: theta * sqrt(r) * sqrt(1-t) = z_needed_rem + z_target
        z_target = stats.norm.ppf(target_cp)
        numerator = z_needed_rem + z_target

        if numerator <= 0:
            # Already reaching target cp or boundary impossible
            return new_protocol

        # sqrt(r) = numerator / (theta_total * np.sqrt(1 - t))
        r = (numerator / (theta_total * np.sqrt(1 - t_val))) ** 2

        # New max sample size
        # N_new = N_look + r * N_rem = t * n_old + r * (1-t) * n_old
        new_n_float = (t_val + r * (1 - t_val)) * n_old_val
        new_n = int(np.ceil(new_n_float))

        # Often SSR is capped (e.g., at 2x or 4x the original n_max).
        # We respect the inflation_cap from the protocol spec if provided.
        inflation_cap = float("inf")  # Default if not specified (uncapped)
        if (
            protocol.method.adaptation
            and hasattr(protocol.method.adaptation, "inflation_cap")
            and protocol.method.adaptation.inflation_cap is not None
        ):
            inflation_cap = protocol.method.adaptation.inflation_cap

        if np.isfinite(inflation_cap):
            new_n = min(new_n, int(np.ceil(inflation_cap * n_old_val)))
        new_n = max(new_n, n_old_val)

        if hasattr(new_protocol.method.stopping_policy.timer, "max_sample_size"):
            # Redistribute new_n proportionally
            n_old_dict = getattr(
                protocol.method.stopping_policy.timer, "max_sample_size", {}
            )
            if isinstance(n_old_dict, dict) and n_old_val > 0:
                ratio = new_n / n_old_val
                new_n_dict = {k: int(np.ceil(v * ratio)) for k, v in n_old_dict.items()}
                new_protocol.method.stopping_policy.timer.max_sample_size = new_n_dict

        # Attach snapshot for Type I error preservation
        new_protocol.method.adaptation_snapshot = GST.AdaptationSnapshot(
            z_t=z_t_val,
            info_frac=t_val,
            original_max_sample_size=n_old_val,
        )

        return new_protocol
