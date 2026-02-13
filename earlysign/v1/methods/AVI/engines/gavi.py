from typing import Any

import numpy as np

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.schema.ES3.AVI import GAVIMethodSpec, Protocol
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboard
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboard


class GAVIEngine:
    """Engine for Generalized Always Valid Inference (GAVI).

    Implements the boundary minimization using Lambert W_{-1} approximation
    as described in Waudby-Smith et al. (2021).
    """

    def __init__(self, protocol: Protocol):
        if not isinstance(protocol.method, GAVIMethodSpec):
            raise ValueError("Protocol method must be GAVIMethodSpec for GAVIEngine.")
        self.protocol = protocol
        self.method: GAVIMethodSpec = protocol.method

    def _get_alpha_adjusted(self) -> float:
        # formulas are for two-sided experiments.
        # for one-sided, multiply alpha by 2.
        if self.method.sides == "one":
            return 2 * self.method.alpha
        return self.method.alpha

    def run(
        self,
        metrics: BinomialScoreboard | ContinuousScoreboard,
        **kwargs: Any,
    ) -> LookResult:
        """Runs the GAVI engine to determine if a boundary is crossed.

        Args:
            metrics: The scoreboard containing metrics for control and treatment arms.
            **kwargs: Additional keyword arguments (not used in this method).

        Returns:
            A `LookResult` object indicating the current status of the experiment,
            including trajectory, boundary, and decision status.
        """
        arms_struct = self.protocol.task.arms
        if isinstance(arms_struct, ES3_BASE.TwoArmComparison):
            control_key = arms_struct.control_arm_name
            treatment_key = arms_struct.treatment_arm_name
        else:
            raise ValueError(
                f"GAVIEngine requires a TwoArmComparison arm structure, but got {type(arms_struct).__name__}."
            )

        # Extract metrics
        s_c = metrics.arms.get(control_key)
        s_t = metrics.arms.get(treatment_key)

        if not s_c or not s_t:
            # Not enough data yet
            return LookResult(
                sample_n=0,
                trajectory=0.0,
                boundary=float("inf"),
                is_crossed=False,
                status=DecisionStatus.CONTINUE_,
            )

        n_c = s_c.metrics.n
        n_t = s_t.metrics.n

        if n_c == 0 or n_t == 0:
            return LookResult(
                sample_n=n_c + n_t,
                trajectory=0.0,
                boundary=float("inf"),
                is_crossed=False,
                status=DecisionStatus.CONTINUE_,
            )

        # Baseline GAVI uses the arm-level sample size (number of pairs)
        # to drive the anytime-valid clock.
        n = (n_c + n_t) / 2.0

        # Calculate trajectory (Estimated effect size)
        if isinstance(metrics, BinomialScoreboard):
            val_c = metrics.arms[control_key].metrics.p_hat
            val_t = metrics.arms[treatment_key].metrics.p_hat
        else:
            val_c = metrics.arms[control_key].metrics.mean
            val_t = metrics.arms[treatment_key].metrics.mean

        estimate = val_t - val_c

        alpha = self._get_alpha_adjusted()
        phi = float(self.method.max_n)

        # Variance estimation
        sigma2 = self.method.variance
        if sigma2 is None:
            if isinstance(metrics, BinomialScoreboard):
                var_c = val_c * (1 - val_c)
                var_t = val_t * (1 - val_t)
            else:
                var_c = metrics.arms[control_key].metrics.variance
                var_t = metrics.arms[treatment_key].metrics.variance
            sigma2 = (var_c + var_t) / 2.0

        # Variance of the mean difference (two-sample)
        # V = sigma2/n_c + sigma2/n_t = 2*sigma2/n (for balanced)
        V = (sigma2 / n_c) + (sigma2 / n_t)

        # GAVI boundary formula from Waudby-Smith (2021)
        # for standard Brownian motion at time n
        denom = np.log(np.log(np.exp(1) * alpha ** (-2))) - 2 * np.log(alpha)
        rho = phi / denom
        term_log = np.log((n + rho) / (rho * alpha**2))
        uv = np.sqrt((n + rho) * term_log)

        # Boundary for mean difference: sigma * u(n) / n
        # Note: Baseline GAVI templates in this project use 1-arm equivalent variance scaling.
        ci = np.sqrt(V / 2.0) * uv / np.sqrt(n)

        # Boundary is 'ci'. Trajectory is 'estimate'.
        # If sides="two": check |estimate| > ci.
        # If sides="one": check estimate > ci.

        is_crossed = False
        if self.method.sides == "two":
            if abs(estimate) > ci:
                is_crossed = True
        else:
            if estimate > ci:
                is_crossed = True

        status = DecisionStatus.CONTINUE_
        if is_crossed:
            status = DecisionStatus.STOP_EFFICACY
        elif n >= self.method.max_n:
            status = DecisionStatus.STOP_PLAN_END_REACHED

        return LookResult(
            sample_n=int(n_c + n_t),
            trajectory=estimate,
            boundary=ci,
            is_crossed=is_crossed,
            status=status,
        )
