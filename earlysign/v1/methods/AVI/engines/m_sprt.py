from typing import Any

import numpy as np

from earlysign.schema.ES3.AVI import MSPRTMethodSpec, Protocol
from earlysign.schema.ES3.AVI.Log import DecisionStatus, LookResult
from earlysign.schema.ES3.Binomial import Scoreboard as BinomialScoreboard
from earlysign.schema.ES3.Continuous import Scoreboard as ContinuousScoreboard


class mSPRTEngine:
    """Engine for Always Valid F-test (mSPRT).

    Implements the mixture Likelihood Ratio based confidence sequences
    described in Johari et al. (2019).
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
        """Runs the mSPRT engine to determine if a boundary is crossed.

        Args:
            metrics: The scoreboard containing metrics for control and treatment arms.
            **kwargs: Additional keyword arguments.

        Returns:
            A `LookResult` object indicating the current status of the experiment.
        """
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

        n_total = n_c + n_t

        if isinstance(metrics, BinomialScoreboard):
            val_c = metrics.arms[control_key].metrics.p_hat
            val_t = metrics.arms[treatment_key].metrics.p_hat
            # Variance estimation: p(1-p)
            var_c = val_c * (1 - val_c)
            var_t = val_t * (1 - val_t)
        else:
            val_c = metrics.arms[control_key].metrics.mean
            val_t = metrics.arms[treatment_key].metrics.mean
            # Continuous sample variance
            var_c = metrics.arms[control_key].metrics.variance or 1.0  # Fallback
            var_t = metrics.arms[treatment_key].metrics.variance or 1.0

        estimate = val_t - val_c
        alpha = self._get_alpha_adjusted()

        # Information-based calculation (Always-Valid CS)
        # Robustly handles unequal n_c and n_t.
        # Var(diff) = var_c/n_c + var_t/n_t
        var_diff = (
            (var_c / n_c) + (var_t / n_t) if (n_c > 0 and n_t > 0) else float("inf")
        )
        if var_diff == 0:  # Degenerate case with no variance
            var_diff = 1e-10

        info = 1.0 / var_diff

        # tau: Prior mixing standard deviation.
        # We use the Target Effect Size (MDE) as a proxy for the optimal tau.
        tau = self.method.mde
        if tau <= 0:
            tau = 0.05  # Sensible fallback for mSPRT

        phi = 1.0 / (tau**2)  # Prior precision (information)

        # Calculate Confidence Interval (Boundary)
        # CS Boundary derived from mixture Likelihood Ratio (Robbins 1970, Johari 2019)
        # B = sqrt( 2 * V_diff * (1 + phi/I) * log( sqrt((I+phi)/phi) / alpha ) )
        ratio = (info + phi) / phi
        ci = np.sqrt(2 * var_diff * (1 + phi / info) * np.log(np.sqrt(ratio) / alpha))

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

        return LookResult(
            sample_n=n_total,
            trajectory=estimate,
            boundary=ci,
            is_crossed=is_crossed,
            status=status,
        )
