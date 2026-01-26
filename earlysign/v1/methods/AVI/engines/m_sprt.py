from typing import Any

import numpy as np

from earlysign.schema.ES3.AVI import MSPRTMethodSpec, Protocol
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboard
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboard


class mSPRTEngine:
    """
    Engine for Always Valid F-test (mSPRT).
    """

    def __init__(self, protocol: Protocol):
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
    ) -> LookResult:
        arms = self.protocol.task.arms
        if len(arms) != 2:
            raise ValueError("AVI requires exactly 2 arms.")

        control_key = arms[0]
        treatment_key = arms[1]

        s_c = metrics.arms.get(control_key)
        s_t = metrics.arms.get(treatment_key)

        if not s_c or not s_t:
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

        # Assuming equal sample sizes (or close to equal) for this logic.
        # We use the average sample size per group for calculation.

        n_group = (n_c + n_t) / 2.0
        n_total = n_c + n_t

        if isinstance(metrics, BinomialScoreboard):
            val_c = s_c.metrics.p_hat  # type: ignore[union-attr]
            val_t = s_t.metrics.p_hat  # type: ignore[union-attr]
        else:
            val_c = s_c.metrics.mean  # type: ignore[union-attr]
            val_t = s_t.metrics.mean  # type: ignore[union-attr]

        estimate = val_t - val_c

        alpha = self._get_alpha_adjusted()
        sigma2 = self.method.variance
        effect_size = self.method.mde

        # V = 2 * sigma2 / (n_group) = 4 * sigma2 / n_total
        # This matches the variance of difference in means for two groups of size n_group.
        rho_param = 0.5

        V = 2 * sigma2 / n_group

        # phi: Parameter ~ 1/relative_effect_size^2
        # We use the Target Effect Size (MDE) to calculate phi.

        phi = sigma2 / (effect_size**2 * rho_param * (1 - rho_param))

        # Calculate Confidence Interval (Boundary)

        term_log = np.log((phi + n_total) / (phi * alpha**2))
        ci = np.sqrt(V) * np.sqrt(term_log * (phi + n_total) / n_total)

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

        # Stop plan end reached check (if needed)

        return LookResult(
            sample_n=n_total,
            trajectory=estimate,
            boundary=ci,
            is_crossed=is_crossed,
            status=status,
        )
