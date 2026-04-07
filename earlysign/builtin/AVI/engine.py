from typing import Any, List, Optional

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.builtin.AVI.adapters import BinomialAdapter, ContinuousAdapter
from earlysign.builtin.AVI.core import (
    BinomialEValueModel,
    GAVIBoundaryModel,
    MSPRTBoundaryModel,
)
from earlysign.builtin.AVI.schema import (
    AVILookResult,
    AVIProtocol,
    DecisionStatus,
    GAVIMethodSpec,
    MSPRTMethodSpec,
    Scoreboard as QuantileScoreboard,
    SequentialQuantileLookResult,
    SequentialQuantileMethodSpec,
)
from earlysign.framework.trace import TraceId
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboard
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboard


class BinomialEValueEngine:
    """Engine for 1-sample Binomial E-value monitoring."""

    def __init__(
        self, protocol: Any
    ):  # typed as EProcessProtocol but strict typing might require importing it
        self.protocol = protocol

    def run(self, metrics: BinomialScoreboard, **kwargs: Any) -> AVILookResult:
        n_total, s_total = BinomialAdapter.extract_total_stats(metrics)

        res = BinomialEValueModel.compute(
            n=n_total,
            successes=s_total,
            null_p=self.protocol.null_p,
            alt_p=self.protocol.alt_p,
            alpha=self.protocol.alpha,
        )

        status = DecisionStatus.CONTINUE
        if res.is_rejected:
            status = DecisionStatus.STOP_DETECTED

        return AVILookResult(
            sample_n=n_total,
            trajectory=res.e_value,
            boundary=1.0 / self.protocol.alpha,
            is_crossed=res.is_rejected,
            status=status,
        )


class GAVIEngine:
    """Engine for Generalized Always Valid Inference (GAVI)."""

    def __init__(self, protocol: AVIProtocol):
        if not isinstance(protocol.method, GAVIMethodSpec):
            raise ValueError("Protocol method must be GAVIMethodSpec for GAVIEngine.")
        self.protocol = protocol
        self.method: GAVIMethodSpec = protocol.method

    def _get_alpha_adjusted(self) -> float:
        if self.method.sides == "one":
            return 2 * self.method.alpha
        return self.method.alpha

    def run(
        self,
        metrics: BinomialScoreboard | ContinuousScoreboard,
        **kwargs: Any,
    ) -> AVILookResult:
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError("GAVIEngine requires a TwoArmComparison arm structure.")

        if isinstance(metrics, BinomialScoreboard):
            n_c, n_t, val_c, val_t = BinomialAdapter.extract_stats(
                metrics, control_key, treatment_key
            )
            # Variance estimation for Binomial: p(1-p)
            var_c = val_c * (1 - val_c)
            var_t = val_t * (1 - val_t)
            sigma2_est = (var_c + var_t) / 2.0
        else:
            # Continuous
            n_c, n_t, val_c, val_t, var_c, var_t = ContinuousAdapter.extract_stats(
                metrics, control_key, treatment_key
            )
            sigma2_est = (var_c + var_t) / 2.0

        # Enforce burn-in period to avoid extreme instability of empirical variance plug-in
        # and to wait for asymptotic approximations to hold.
        burn_in = getattr(self.method, "burn_in", 100) or 100
        n_total = n_c + n_t
        if n_c < burn_in or n_t < burn_in:
            return AVILookResult(
                sample_n=n_total,
                trajectory=0.0,
                boundary=float("inf"),
                is_crossed=False,
                status=DecisionStatus.CONTINUE,
            )

        estimate = val_t - val_c
        alpha = self._get_alpha_adjusted()

        # Phi (max_n) is typically required for GAVI boundary.
        # Fallback to extremely large if not specified to allow execution,
        # though this changes the statistical semantics (it becomes a non-budgeted CS).
        max_n_val = getattr(self.method, "max_n", None)
        phi = float(max_n_val) if max_n_val is not None else 1e9

        # Override sigma2 from method if provided
        sigma2 = (
            self.method.variance if self.method.variance is not None else sigma2_est
        )

        ci = GAVIBoundaryModel.calculate_boundary(
            n_total=float(n_total),
            alpha=alpha,
            phi=phi,
            sigma2=sigma2,
            n_c=n_c,
            n_t=n_t,
        )

        is_crossed = False
        if self.method.sides == "two":
            if abs(estimate) > ci:
                is_crossed = True
        else:
            if estimate > ci:
                is_crossed = True

        status = DecisionStatus.CONTINUE
        if is_crossed:
            status = DecisionStatus.STOP_DETECTED
        elif self.method.max_n is not None and (n_total / 2.0) >= self.method.max_n:
            status = DecisionStatus.STOP_PLAN_END_REACHED

        return AVILookResult(
            sample_n=(n_c or 0) + (n_t or 0),
            trajectory=estimate or 0.0,
            boundary=ci or 0.0,
            is_crossed=is_crossed,
            status=status,
        )


class mSPRTEngine:
    """Engine for Always Valid F-test (mSPRT)."""

    def __init__(self, protocol: AVIProtocol):
        if not isinstance(protocol.method, MSPRTMethodSpec):
            raise ValueError("Protocol method must be MSPRTMethodSpec for mSPRTEngine.")
        self.protocol = protocol
        self.method: MSPRTMethodSpec = protocol.method

    def _get_alpha_adjusted(self) -> float:
        if self.method.sides == "one":
            return 2 * self.method.alpha
        return self.method.alpha

    def run(
        self,
        metrics: BinomialScoreboard | ContinuousScoreboard,
        **kwargs: Any,
    ) -> AVILookResult:
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError("mSPRTEngine requires a TwoArmComparison arm structure.")

        if isinstance(metrics, BinomialScoreboard):
            n_c, n_t, val_c, val_t = BinomialAdapter.extract_stats(
                metrics, control_key, treatment_key
            )
            var_c = val_c * (1 - val_c)
            var_t = val_t * (1 - val_t)
        else:
            n_c, n_t, val_c, val_t, var_c, var_t = ContinuousAdapter.extract_stats(
                metrics, control_key, treatment_key
            )

        # Enforce burn-in period to avoid extreme instability of empirical variance plug-in
        # and to wait for asymptotic approximations to hold.
        burn_in = getattr(self.method, "burn_in", 100) or 100
        if n_c < burn_in or n_t < burn_in:
            return AVILookResult(
                sample_n=n_c + n_t,
                trajectory=0.0,
                boundary=float("inf"),
                is_crossed=False,
                status=DecisionStatus.CONTINUE,
            )

        estimate = val_t - val_c
        alpha = self._get_alpha_adjusted()

        # Var(diff) = var_c/n_c + var_t/n_t
        var_diff = (var_c / n_c) + (var_t / n_t)
        if var_diff == 0:
            var_diff = 1e-10

        ci = MSPRTBoundaryModel.calculate_boundary(
            var_diff=var_diff,
            tau_mde=getattr(self.method, "mde", 1.0),
            alpha=alpha,
        )

        is_crossed = False
        if self.method.sides == "two":
            if abs(estimate) > ci:
                is_crossed = True
        else:
            if estimate > ci:
                is_crossed = True

        status = DecisionStatus.CONTINUE
        if is_crossed:
            status = DecisionStatus.STOP_DETECTED

        return AVILookResult(
            sample_n=(n_c or 0) + (n_t or 0),
            trajectory=estimate or 0.0,
            boundary=ci or 0.0,
            is_crossed=is_crossed,
            status=status,
        )


class SequentialQuantileEngine:
    """Engine for Howard & Ramdas (2022) Sequential Quantile estimation."""

    def __init__(self, protocol: AVIProtocol):
        self.protocol = protocol

    def run(
        self, metrics: QuantileScoreboard, trace: Optional[List[TraceId]] = None
    ) -> SequentialQuantileLookResult:
        method = self.protocol.method
        if not isinstance(method, SequentialQuantileMethodSpec):
            # This check might fail if method is generic, but usually strict in engine
            pass  # raise TypeError("Expected SequentialQuantileMethodSpec")

        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            ctrl_id = arms_struct.control_arm_name
            treat_id = arms_struct.treatment_arm_name
        else:
            raise ValueError("SequentialQuantileEngine requires TwoArmComparison")

        arms_dict = metrics.arms or {}
        ctrl_arm = arms_dict.get(ctrl_id)
        treat_arm = arms_dict.get(treat_id)

        ctrl = ctrl_arm.metrics if ctrl_arm else None
        treat = treat_arm.metrics if treat_arm else None

        is_disjoint = False
        if (
            ctrl
            and treat
            and ctrl.ci_upper is not None
            and treat.ci_lower is not None
            and treat.ci_upper is not None
            and ctrl.ci_lower is not None
        ):
            is_disjoint = (ctrl.ci_upper < treat.ci_lower) or (
                treat.ci_upper < ctrl.ci_lower
            )

        status = DecisionStatus.CONTINUE
        if is_disjoint:
            status = DecisionStatus.STOP_DETECTED

        ctrl_n = ctrl.total if ctrl else 0
        treat_n = treat.total if treat else 0
        total_n = ctrl_n + treat_n
        if (
            hasattr(method, "max_n")
            and method.max_n is not None
            and total_n >= method.max_n
            and status == DecisionStatus.CONTINUE
        ):
            status = DecisionStatus.STOP_PLAN_END_REACHED

        return SequentialQuantileLookResult(
            sample_n=total_n,
            trajectory=(
                (treat.quantile_estimate or 0.0) - (ctrl.quantile_estimate or 0.0)
                if ctrl and treat
                else 0.0
            ),
            boundary=float("nan"),
            estimated_quantile=(
                float(treat.quantile_estimate)
                if treat and treat.quantile_estimate
                else 0.0
            ),
            interval_lower=float(treat.ci_lower) if treat and treat.ci_lower else 0.0,
            interval_upper=float(treat.ci_upper) if treat and treat.ci_upper else 0.0,
            is_crossed=is_disjoint,
            status=status,
        )
