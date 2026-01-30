from typing import Any

import numpy as np

from earlysign.schema.ES3.AVI import GAVIMethodSpec, Protocol
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboard
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboard


class GAVIEngine:
    """
    Engine for Generalized Always Valid Inference (GAVI).
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
        arms = self.protocol.task.arms
        if len(arms) != 2:
            raise ValueError("AVI requires exactly 2 arms.")

        control_key = arms[0]
        treatment_key = arms[1]

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

        # Using average sample size per group for calculation.
        n = (n_c + n_t) / 2.0

        # Calculate trajectory (Estimated effect size)
        # Binary: p_hat diff. Continuous: mean diff.
        # Calculate trajectory (Estimated effect size)
        # Binary: p_hat diff. Continuous: mean diff.
        if isinstance(metrics, BinomialScoreboard):
            val_c = metrics.arms[control_key].metrics.p_hat
            val_t = metrics.arms[treatment_key].metrics.p_hat
        else:
            val_c = metrics.arms[control_key].metrics.mean
            val_t = metrics.arms[treatment_key].metrics.mean

        estimate = val_t - val_c

        alpha = self._get_alpha_adjusted()

        # phi (parameter ~ sample size)
        phi = float(self.method.max_n)

        # Variance estimation
        sigma2 = self.method.variance
        if sigma2 is None:
            # Estimate variance from data (Maharaj et al., 2023)
            # Use pooled variance estimate as effective variance for the difference
            if isinstance(metrics, BinomialScoreboard):
                # Binomial approximation: Var(X) ~ p(1-p)
                var_c = val_c * (1 - val_c)
                var_t = val_t * (1 - val_t)
            else:
                # Continuous sample variance
                var_c = metrics.arms[control_key].metrics.variance
                var_t = metrics.arms[treatment_key].metrics.variance

            sigma2_effective = (var_c + var_t) / 2.0
            sigma2 = sigma2_effective

        # variance of the difference in means
        # If variances were provided/estimated as 'common variance' sigma2:
        V = 2 * sigma2 / n

        # formula 21: normalized boundary minimisation using Lambert W_{-1} approximation
        # rho = phi / (np.log(np.log(np.exp(1) * alpha ** (-2))) - 2 * np.log(alpha))

        denom = np.log(np.log(np.exp(1) * alpha ** (-2))) - 2 * np.log(alpha)
        rho = phi / denom

        # formula 14: two-sided normal mixture boundary
        term_log = np.log((n + rho) / (rho * alpha**2))

        # (n + rho) / (rho * alpha^2) should be > 1.
        uv = np.sqrt((n + rho) * term_log)

        # ci = sqrt(2*sigma2/n) * uv / sqrt(n) = sqrt(2*sigma2) * uv / n
        ci = np.sqrt(V) * uv / np.sqrt(n)

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
