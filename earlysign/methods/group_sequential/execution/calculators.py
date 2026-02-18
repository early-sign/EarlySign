from typing import Any, Dict, Optional, Protocol as TypingProtocol

import numpy as np

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST


class ZStatisticCalculator(TypingProtocol):
    """Protocol for Z-statistic calculators supporting MAMS and diverse stats models."""

    def calculate(
        self, metrics: Any, protocol: GST.Protocol
    ) -> Optional[Dict[str, float]]:
        """
        Calculates Z-statistics for all treatment arms against the control.
        Returns a mapping of treatment arm name -> Z-statistic,
        or None if data is insufficient/missing.
        """
        ...


class BinomialZCalculator:
    """Binomial Z-statistic calculator supporting Wald (unpooled) and Score (pooled) variants."""

    def calculate(
        self, metrics: Any, protocol: GST.Protocol
    ) -> Optional[Dict[str, float]]:
        task = protocol.task
        arms_struct = task.arms

        # Determine Control Arm
        if isinstance(
            arms_struct, (ES3_BASE.TwoArmComparison, ES3_BASE.MultiArmComparison)
        ):
            control_name = arms_struct.control_arm_name
        else:
            # Single arm logic or unsupported arm structure
            return None

        control_status = metrics.arms.get(control_name)
        if control_status is None or control_status.metrics.total == 0:
            return None

        control_metrics = control_status.metrics

        # Determine Treatment Arms
        treatment_names = []
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            treatment_names = [arms_struct.treatment_arm_name]
        elif isinstance(arms_struct, ES3_BASE.MultiArmComparison):
            treatment_names = arms_struct.treatment_arm_names

        # Determine Variance Estimation Method
        # In ES3.GST.TwoArmBinomialZ, variance_estimation is used.
        # In ES3.GST.OneArmBinomialZ, variance_source is used.
        stopping_stat = protocol.method.stopping_policy.statistic

        variance_method = GST.VarianceEstimation.POOLED
        if hasattr(stopping_stat, "variance_estimation"):
            variance_method = stopping_stat.variance_estimation

        results = {}
        for trtm_name in treatment_names:
            trtm_status = metrics.arms.get(trtm_name)
            if trtm_status is None or trtm_status.metrics.total == 0:
                continue  # Skip arm if no data, effectively returning None for this arm in the dict

            trtm_metrics = trtm_status.metrics

            z = self._compute_single_z(control_metrics, trtm_metrics, variance_method)
            if z is not None:
                results[trtm_name] = z

        return results if results else None

    def _compute_single_z(
        self, control: Any, treatment: Any, method: GST.VarianceEstimation
    ) -> Optional[float]:
        n_c, n_t = control.total, treatment.total
        if n_c == 0 or n_t == 0:
            return None  # Explicitly None for missing data

        p_c = control.successes / n_c
        p_t = treatment.successes / n_t

        if method == GST.VarianceEstimation.POOLED:
            p_pool = (control.successes + treatment.successes) / (n_c + n_t)
            if p_pool <= 0 or p_pool >= 1.0:
                return 0.0  # Or None? Usually 0.0 if both are 0 or 1
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
        else:
            # Unpooled (Wald)
            se_c2 = (p_c * (1 - p_c)) / n_c if p_c > 0 and p_c < 1 else 0
            se_t2 = (p_t * (1 - p_t)) / n_t if p_t > 0 and p_t < 1 else 0
            se = np.sqrt(se_c2 + se_t2)

        if se <= 0:
            return 0.0

        return float((p_t - p_c) / se)


class ContinuousZCalculator:
    """Continuous Z-statistic calculator supporting pooled and unpooled (Welch) variance."""

    def calculate(
        self, metrics: Any, protocol: GST.Protocol
    ) -> Optional[Dict[str, float]]:
        task = protocol.task
        arms_struct = task.arms

        if isinstance(
            arms_struct, (ES3_BASE.TwoArmComparison, ES3_BASE.MultiArmComparison)
        ):
            control_name = arms_struct.control_arm_name
        else:
            return None

        control_status = metrics.arms.get(control_name)
        if control_status is None or control_status.metrics.total == 0:
            return None

        control_metrics = control_status.metrics

        treatment_names = []
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            treatment_names = [arms_struct.treatment_arm_name]
        elif isinstance(arms_struct, ES3_BASE.MultiArmComparison):
            treatment_names = arms_struct.treatment_arm_names

        stat_spec = protocol.method.stopping_policy.statistic

        # Continuous usually has 'variance' spec: KnownVariance or TwoArmEstimatedVariance
        # If TwoArmEstimatedVariance, it has 'method' (pooled/unpooled)
        variance_method = "unpooled"
        if hasattr(stat_spec, "variance") and hasattr(stat_spec.variance, "method"):
            variance_method = stat_spec.variance.method

        results = {}
        for trtm_name in treatment_names:
            trtm_status = metrics.arms.get(trtm_name)
            if trtm_status is None or trtm_status.metrics.total == 0:
                continue

            trtm_metrics = trtm_status.metrics
            z = self._compute_single_z(control_metrics, trtm_metrics, variance_method)
            if z is not None:
                results[trtm_name] = z

        return results if results else None

    def _compute_single_z(
        self, control: Any, treatment: Any, method: str
    ) -> Optional[float]:
        n_c, n_t = control.total, treatment.total
        if n_c < 2 or n_t < 2:
            return None  # Need at least 2 points for variance estimate if not known

        m_c, m_t = control.mean, treatment.mean
        v_c, v_t = control.variance, treatment.variance

        if method == "pooled":
            # Pooled variance estimate
            v_pool = ((n_c - 1) * v_c + (n_t - 1) * v_t) / (n_c + n_t - 2)
            se = np.sqrt(v_pool * (1 / n_c + 1 / n_t))
        else:
            # Unpooled (Welch)
            se = np.sqrt(v_c / n_c + v_t / n_t)

        if se <= 0:
            return 0.0

        return float((m_t - m_c) / se)


class ZStatisticCalculatorFactory:
    """Factory to build the appropriate calculator based on Protocol."""

    @staticmethod
    def build(protocol: GST.Protocol) -> ZStatisticCalculator:
        response_type = protocol.task.response_type
        if response_type == GST.ResponseType.BINARY:
            return BinomialZCalculator()
        elif response_type == GST.ResponseType.CONTINUOUS:
            return ContinuousZCalculator()
        else:
            raise ValueError(
                f"Unsupported response type for Z-calculation: {response_type}"
            )
