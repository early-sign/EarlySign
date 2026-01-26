"""Simulator for calculating operating characteristics of Group Sequential Tests."""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)
from earlysign.v1.methods.group_sequential.shared.spending import SpendingFunction
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


@dataclass
class OptimizationConfig:
    """Configuration for optimization simulations."""

    n_global_samples: int = 100
    n_top_seeds: int = 5
    method: Literal["simulation", "numerical_integration"] = "simulation"
    n_sims: int = 50000
    rng_seed: Optional[int] = None
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.method == "numerical_integration" and self.n_sims > 0:
            pass


class SequentialASNEstimator:
    """Estimates ASN for a given schedule using CanonicalJointModel."""

    def __init__(
        self,
        k_looks: int,
        alpha: float,
        efficacy_spending: Optional[SpendingFunction],
        futility_spending: Optional[SpendingFunction],
        drift: float,
        prior: Optional[Any] = None,
        tails: int = 1,
        config: OptimizationConfig = OptimizationConfig(),
    ) -> None:
        self.k_looks = k_looks
        self.alpha = alpha
        self.efficacy_spending = efficacy_spending
        self.futility_spending = futility_spending
        self.drift = drift
        self.prior = prior
        self.tails = tails
        self.config = config
        self._rng_seed = config.rng_seed or np.random.randint(0, 10000)

    def calculate(self, x_increments: NDArray[np.float64]) -> float:
        """Calculate ASN (or E[ASN]) for a given increment vector."""
        t = np.cumsum(x_increments)
        t = np.clip(t, 1e-6, 1.0)
        t[-1] = 1.0

        model_config = Config(
            info_times=t,
            alpha=self.alpha,
            efficacy_spending=self.efficacy_spending,
            futility_spending=self.futility_spending,
            tails=self.tails,
            n_sims=self.config.n_sims,
            rng_seed=self._rng_seed,
        )
        model = CanonicalJointModel(model_config)

        try:
            a, b = model.solve_boundaries(drift=0.0, method=self.config.method)
        except Exception:
            return 1e6

        if self.prior is None:
            return self._compute_point_asn(model, t, a, b, self.drift)
        else:
            return self._compute_point_asn(model, t, a, b, self.drift)

    def _compute_point_asn(
        self,
        model: CanonicalJointModel,
        t: NDArray[np.float64],
        upper: Optional[NDArray[np.float64]],
        lower: Optional[NDArray[np.float64]],
        drift: float,
    ) -> float:
        """Compute ASN at a specific drift."""
        if self.config.method == "simulation":
            from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess

            gp_eval = CanonicalGaussianProcess(
                drift=drift, rng=np.random.default_rng(self._rng_seed)
            )
            samples = gp_eval.sample(t, self.config.n_sims)
            stopped, stop_looks = gp_eval.apply_stopping_rule(samples, upper, lower)
            stop_times = t[stop_looks - 1]
            return float(np.mean(stop_times))
        else:
            current_t = 0.0
            asn = 0.0
            for i in range(len(t)):
                dt = t[i] - current_t
                if i == 0:
                    prob_survive = 1.0
                else:
                    prob_stopped = model.compute_crossing_probability(
                        info_times=t[:i],
                        upper=upper[:i] if upper is not None else None,
                        lower=lower[:i] if lower is not None else None,
                        drift=drift,
                        method="numerical_integration",
                    )
                    prob_survive = 1.0 - prob_stopped
                asn += dt * prob_survive
                current_t = t[i]
            return asn

    def evaluate(self, x_increments: NDArray[np.float64]) -> Tuple[float, float]:
        """Calculate both ASN and Power for a given schedule.

        Returns:
            Tuple of (asn, power).
        """
        t = np.cumsum(x_increments)
        t = np.clip(t, 1e-6, 1.0)
        t[-1] = 1.0

        model_config = Config(
            info_times=t,
            alpha=self.alpha,
            efficacy_spending=self.efficacy_spending,
            futility_spending=self.futility_spending,
            tails=self.tails,
            n_sims=self.config.n_sims,
            rng_seed=self._rng_seed,
        )
        model = CanonicalJointModel(model_config)

        try:
            a, b = model.solve_boundaries(drift=0.0, method=self.config.method)
        except Exception:
            return 1e6, 0.0

        if a is None:
            return 1e6, 0.0

        # Calculate ASN
        if self.prior is None:
            asn = self._compute_point_asn(model, t, a, b, self.drift)
        else:
            asn = self._compute_point_asn(model, t, a, b, self.drift)

        # Calculate Power
        # Power = P(Reject H0) under H1 (drift=self.drift)
        # We use simulation for Power calculation as a robust fallback even for numerical_integration optimization
        # because the generic numerical integration for "Rejection excluding binding futility" is complex to handle generically here.

        gp_eval = CanonicalGaussianProcess(
            drift=self.drift, rng=np.random.default_rng(self._rng_seed)
        )
        samples = gp_eval.sample(t, self.config.n_sims)

        # Efficacy Rejection
        stopped_eff = np.zeros(self.config.n_sims, dtype=bool)
        stopped_any = np.zeros(self.config.n_sims, dtype=bool)

        for i in range(len(t)):
            # Upper crossing
            crossing_u = (samples[:, i] > a[i]) & ~stopped_any
            if self.tails == 2:
                crossing_l = (samples[:, i] < -a[i]) & ~stopped_any
                stopped_eff |= crossing_u | crossing_l
                stopped_any |= crossing_u | crossing_l
            else:
                stopped_eff |= crossing_u
                stopped_any |= crossing_u
                if b is not None:
                    crossing_fut = (samples[:, i] < b[i]) & ~stopped_any
                    stopped_any |= crossing_fut

        power = float(np.mean(stopped_eff))

        return asn, power
