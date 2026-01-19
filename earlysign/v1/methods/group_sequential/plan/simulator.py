"""Simulator for calculating operating characteristics of Group Sequential Tests."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess


@dataclass
class OperatingCharacteristic:
    """Operating characteristics of a group sequential design."""

    alpha: float
    power: float
    asn: float  # Expected Sample Size (Average Sample Number)
    stop_probs_efficacy: NDArray[np.float64]
    stop_probs_futility: NDArray[np.float64]
    drift: float


@runtime_checkable
class GSTFacade(Protocol):
    """Protocol for high-level trial templates like BinomialABTemplate."""

    def backtest(self, batches: Any) -> Dict[str, Any]: ...


class OperatingCharacteristicSimulator:
    """Simulator to evaluate Group Sequential designs."""

    def __init__(self, n_sims: int = 20000, rng_seed: Optional[int] = None):
        self.n_sims = n_sims
        self._rng = np.random.default_rng(rng_seed)

    def simulate_statistical(
        self,
        info_times: NDArray[np.float64],
        upper: NDArray[np.float64],
        lower: Optional[NDArray[np.float64]] = None,
        drift: float = 0.0,
        n_max: float = 1.0,
        samples: Optional[NDArray[np.float64]] = None,
    ) -> OperatingCharacteristic:
        """Evaluate operating characteristics using vectorized simulation.

        If 'samples' is provided, it uses those (useful for T-tests).
        Otherwise, it simulates a Canonical Gaussian Process.
        """
        if samples is None:
            gp = CanonicalGaussianProcess(drift=drift, rng=self._rng)
            samples = gp.sample(info_times, self.n_sims)  # (n_sims, k)

        k = len(info_times)
        stopped_eff = np.zeros(self.n_sims, dtype=bool)
        stopped_fut = np.zeros(self.n_sims, dtype=bool)
        stop_look = np.full(self.n_sims, k, dtype=int)

        stop_probs_eff = np.zeros(k)
        stop_probs_fut = np.zeros(k)

        for i in range(k):
            # Check for efficacy
            crossing_eff = (samples[:, i] > upper[i]) & ~(stopped_eff | stopped_fut)
            stop_probs_eff[i] = np.mean(crossing_eff)
            stopped_eff |= crossing_eff
            stop_look[crossing_eff] = i + 1

            # Check for futility
            if lower is not None:
                crossing_fut = (samples[:, i] < lower[i]) & ~(stopped_eff | stopped_fut)
                stop_probs_fut[i] = np.mean(crossing_fut)
                stopped_fut |= crossing_fut
                stop_look[crossing_fut] = i + 1

        alpha_power_upper = np.mean(stopped_eff)
        alpha_power_lower = np.mean(stopped_fut)

        # Expected Sample Size calculation
        asn = np.mean(info_times[stop_look - 1]) * n_max

        return OperatingCharacteristic(
            alpha=(
                alpha_power_upper + alpha_power_lower
                if drift == 0.0
                else alpha_power_upper + alpha_power_lower
            ),
            power=(
                alpha_power_upper
                if drift > 0.0
                else (
                    alpha_power_lower
                    if drift < 0.0
                    else alpha_power_upper + alpha_power_lower
                )
            ),
            asn=float(asn),
            stop_probs_efficacy=stop_probs_eff,
            stop_probs_futility=stop_probs_fut,
            drift=drift,
        )

    def simulate_template(
        self,
        template_factory: Any,  # Callable that returns a fresh template instance
        drift: float,
        stream_factory: Any,  # Callable that returns a fresh data stream for one trial
        n_trials: int = 100,
    ) -> OperatingCharacteristic:
        """Evaluate operating characteristics by running the actual template logic.

        This mode is slower but ensures that all ledger, projector, and framework
        overhead is accounted for.
        """
        stopped_eff = 0
        sum_info = 0.0

        # For simplicity in this implementation, we assume the template's backtest
        # returns a dictionary with 'is_rejected' and 'n_total' or similar.
        for _ in range(n_trials):
            template = template_factory()
            stream = stream_factory(drift)

            # The template.backtest should run the trial until completion
            result = template.backtest(stream)

            if result.get("is_rejected", False):
                stopped_eff += 1

            # Record stop time (as fraction of planned max)
            # This requires the result to contain enough info
            sum_info += result.get("info_frac", 1.0)

        return OperatingCharacteristic(
            alpha=stopped_eff / n_trials if drift == 0.0 else 0.0,
            power=stopped_eff / n_trials if drift > 0.0 else 0.0,
            asn=sum_info / n_trials,
            stop_probs_efficacy=np.zeros(
                1
            ),  # Not easily extracted from generic backtest
            stop_probs_futility=np.zeros(1),
            drift=drift,
        )
