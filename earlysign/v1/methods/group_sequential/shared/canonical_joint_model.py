"""Canonical Joint Model for Group Sequential Tests.

Provides generic computation APIs for boundary solving and probability
calculations based on the canonical joint distribution of Z-statistics.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.execution.stopping_policy import (
    SpendingFunctionStoppingPolicy,
    StoppingPolicy,
    StoppingPolicyFactory,
)
from earlysign.v1.methods.group_sequential.shared.spending import SpendingFunction
from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess


@dataclass
class Config:
    """Configuration for CanonicalJointModel.

    Attributes:
        info_times: Information fractions (t_1, ..., t_K).
        alpha: Type I error rate.
        power: Target power.
        stopping_policy: The stopping policy to use.
        efficacy_spending: Spending function for efficacy (legacy).
        futility_spending: Spending function for futility (legacy).
        efficacy_binding: If True, futility affects efficacy calculation.
        futility_binding: If True, efficacy affects futility calculation.
        tails: 1 or 2 (symmetric).
        n_sims: Number of simulations for boundary solving.
        rng_seed: Seed for random number generator.
    """

    info_times: NDArray[np.float64]
    spending_times: Optional[NDArray[np.float64]] = None
    alpha: Optional[float] = None
    power: Optional[float] = None
    stopping_policy: Optional[StoppingPolicy] = None
    efficacy_spending: Optional[SpendingFunction] = None
    futility_spending: Optional[SpendingFunction] = None
    efficacy_binding: bool = True
    futility_binding: bool = False
    tails: int = 1
    n_sims: int = 20000
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.stopping_policy:
            self.efficacy_binding = self.stopping_policy.alpha_binding
            self.futility_binding = self.stopping_policy.beta_binding
            self.tails = 2 if self.stopping_policy.sided == "two" else 1


class CanonicalJointModel:
    """Statistical engine for group sequential tests.

    Provides generic APIs for boundary computation based on the canonical
    joint distribution of Z-statistics. Domain-specific adapters (spending
    functions, shape-based policies) translate their concepts to these APIs.
    """

    def __init__(self, config: Config):
        self.config = config
        self._rng = np.random.default_rng(config.rng_seed)

    @property
    def info_times(self) -> NDArray[np.float64]:
        return self.config.info_times

    @property
    def tails(self) -> int:
        return self.config.tails

    @classmethod
    def from_spec(
        cls, spec: GST.Protocol, n_sims: int = 20000, rng_seed: Optional[int] = None
    ) -> "CanonicalJointModel":
        """Instantiate the model from an ES3 GST.Protocol specification."""
        task = spec.task
        method = spec.method
        task_alpha = float(task.efficacy.alpha) if task.efficacy else None
        task_power = float(task.futility.power) if task.futility else None

        schedule_spec = method.stopping_policy.schedule
        schedule = schedule_spec

        if isinstance(schedule, GST.FixedSchedule):
            t = np.asarray(schedule.analyses)
        elif isinstance(schedule, GST.EquidistantSchedule):
            k = schedule.n_looks
            t = np.linspace(1 / k, 1.0, k)
        else:
            raise ValueError("Unsupported schedule type.")

        if t.size == 0:
            raise ValueError("Schedule has no points.")

        stopping_policy = StoppingPolicyFactory.build_from_spec(method.stopping_policy)

        config = Config(
            info_times=t,
            alpha=task_alpha,
            power=task_power,
            stopping_policy=stopping_policy,
            n_sims=n_sims,
            rng_seed=rng_seed,
        )
        return cls(config)

    # =========================================================================
    # Core Generic APIs
    # =========================================================================

    def compute_crossing_probability(
        self,
        info_times: NDArray[np.float64],
        upper: Optional[NDArray[np.float64]] = None,
        lower: Optional[NDArray[np.float64]] = None,
        method: Literal["simulation", "numerical_integration"] = "simulation",
        drift: float = 0.0,
        n_sims: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> float:
        """Compute probability of crossing boundaries under given drift.

        Args:
            info_times: Information fractions.
            upper: Upper boundaries at each look.
            lower: Lower boundaries at each look.
            method: "simulation" or "numerical_integration".
            drift: Standardized drift (0 for H0, non-zero for H1).
            n_sims: Number of simulations (simulation method only).
            seed: Random seed for reproducibility.

        Returns:
            Probability of crossing either boundary.
        """
        n = n_sims or self.config.n_sims
        s = seed if seed is not None else self.config.rng_seed

        # For numerical integration, we can compute without instance creation overhead if needed,
        # but GP instance creation is cheap.
        gp = CanonicalGaussianProcess(drift=drift, rng=np.random.default_rng(s))
        return gp.compute_crossing_probability(
            t=info_times,
            upper=upper,
            lower=lower,
            n_sims=n,
            method=method,
            seed=s,
        )

    def find_boundary_constant(
        self,
        info_times: NDArray[np.float64],
        shape: NDArray[np.float64],
        target_probability: float,
        tails: int = 1,
        drift: float = 0.0,
        method: Literal["simulation", "numerical_integration"] = "simulation",
    ) -> float:
        """Find constant c such that P(Crossing c*shape) = target_probability.

        Args:
            info_times: Information fractions.
            shape: Shape vector (e.g., 1/sqrt(t) for OBF).
            target_probability: Target crossing probability (e.g., alpha).
            tails: 1 for one-sided, 2 for two-sided symmetric.
            drift: Drift parameter (0 for Type I error).
            method: Computation method.

        Returns:
            Critical constant c.
        """
        crn_seed = (
            self.config.rng_seed
            if self.config.rng_seed is not None
            else int(self._rng.integers(100000))
        )

        def objective(c: float) -> float:
            boundary = c * shape
            if tails == 2:
                prob = self.compute_crossing_probability(
                    info_times,
                    upper=boundary,
                    lower=-boundary,
                    method=method,
                    drift=drift,
                    seed=crn_seed,
                )
            else:
                prob = self.compute_crossing_probability(
                    info_times,
                    upper=boundary,
                    lower=None,
                    method=method,
                    drift=drift,
                    seed=crn_seed,
                )
            return prob - target_probability

        try:
            res = root_scalar(
                objective, bracket=[0.0, 10.0], method="brentq", xtol=1e-3
            )
            return float(res.root)
        except ValueError:
            if objective(10.0) > 0:
                res = root_scalar(
                    objective, bracket=[10.0, 50.0], method="brentq", xtol=1e-3
                )
                return float(res.root)
            raise

    def solve_boundaries_from_cumulative_targets(
        self,
        info_times: NDArray[np.float64],
        efficacy_targets: Optional[NDArray[np.float64]] = None,
        futility_targets: Optional[NDArray[np.float64]] = None,
        drift: float = 0.0,
        efficacy_binding: bool = True,
        futility_binding: bool = False,
        tails: int = 1,
        method: str = "numerical_integration",
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Solve boundaries to match cumulative stopping probabilities.

        Args:
            info_times: Information fractions.
            efficacy_targets: Cumulative efficacy probabilities at each look (under H0).
            futility_targets: Cumulative futility probabilities at each look (under H1).
            drift: Drift for futility boundary solving.
            efficacy_binding: If True, futility affects efficacy calculation.
            futility_binding: If True, efficacy affects futility calculation.
            tails: 1 or 2.

        Returns:
            Tuple of (efficacy_boundaries, futility_boundaries).
        """
        t = info_times
        k = len(t)

        if method == "numerical_integration":
            return self._solve_numerical(
                info_times,
                efficacy_targets,
                futility_targets,
                drift,
                efficacy_binding,
                futility_binding,
                tails,
            )

        # Simulation path
        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        z_sims_h0 = gp_h0.sample(t, self.config.n_sims)

        gp_h1 = CanonicalGaussianProcess(drift=drift, rng=self._rng)
        z_sims_h1 = gp_h1.sample(t, self.config.n_sims)

        a = np.zeros(k) if efficacy_targets is not None else None
        b = np.full(k, -10.0) if futility_targets is not None else None

        rejected_h0 = np.zeros(self.config.n_sims, dtype=bool)
        stopped_h0 = np.zeros(self.config.n_sims, dtype=bool)
        futility_h1 = np.zeros(self.config.n_sims, dtype=bool)
        stopped_h1 = np.zeros(self.config.n_sims, dtype=bool)

        for i in range(k):
            # Efficacy boundary
            if a is not None and efficacy_targets is not None:
                needed = efficacy_targets[i] * self.config.n_sims - np.sum(rejected_h0)
                rem_mask = ~stopped_h0
                num_rem = np.sum(rem_mask)

                if needed <= 0:
                    a[i] = 10.0 if i < k - 1 else (a[i - 1] if i > 0 else 2.0)
                elif num_rem < 10:
                    a[i] = -10.0
                else:
                    frac = max(0, min(1, needed / num_rem))
                    if tails == 2:
                        a[i] = np.percentile(
                            np.abs(z_sims_h0[rem_mask, i]), 100 * (1 - frac)
                        )
                    else:
                        a[i] = np.percentile(z_sims_h0[rem_mask, i], 100 * (1 - frac))

            # Futility boundary
            if b is not None and futility_targets is not None:
                needed = futility_targets[i] * self.config.n_sims - np.sum(futility_h1)
                rem_mask_h1 = ~stopped_h1
                num_rem_h1 = np.sum(rem_mask_h1)

                if needed <= 0:
                    b[i] = -10.0
                elif num_rem_h1 < 10:
                    b[i] = 10.0
                else:
                    frac = max(0, min(1, needed / num_rem_h1))
                    b[i] = np.percentile(z_sims_h1[rem_mask_h1, i], 100 * frac)

            # Update stop masks
            if a is not None:
                if tails == 2:
                    just_rej_h0 = (~stopped_h0) & (np.abs(z_sims_h0[:, i]) > a[i])
                    just_rej_h1 = (~stopped_h1) & (np.abs(z_sims_h1[:, i]) > a[i])
                else:
                    just_rej_h0 = (~stopped_h0) & (z_sims_h0[:, i] > a[i])
                    just_rej_h1 = (~stopped_h1) & (z_sims_h1[:, i] > a[i])
                rejected_h0 |= just_rej_h0
                stopped_h0 |= just_rej_h0
                stopped_h1 |= just_rej_h1

            if b is not None:
                just_fut_h1 = (~stopped_h1) & (z_sims_h1[:, i] < b[i])
                futility_h1 |= just_fut_h1
                stopped_h1 |= just_fut_h1
                if efficacy_binding:
                    just_fut_h0 = (~stopped_h0) & (z_sims_h0[:, i] < b[i])
                    stopped_h0 |= just_fut_h0

        return a, b

    def _solve_numerical(
        self,
        info_times: NDArray[np.float64],
        efficacy_targets: Optional[NDArray[np.float64]],
        futility_targets: Optional[NDArray[np.float64]],
        drift: float,
        efficacy_binding: bool,
        futility_binding: bool,
        tails: int,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
        """Solve stage-by-stage using numerical integration for high precision."""
        k = len(info_times)
        a = np.zeros(k) if efficacy_targets is not None else None
        b = np.full(k, -10.0) if futility_targets is not None else None

        # Higher-level helpers for numerical integration
        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        gp_h1 = CanonicalGaussianProcess(drift=drift, rng=self._rng)

        for i in range(k):
            # Solve efficacy bound a[i]
            if a is not None and efficacy_targets is not None:

                def obj_a(val: float) -> float:
                    temp_a = a.copy()
                    temp_a[i] = val
                    temp_b = (
                        b.copy()
                        if (b is not None and efficacy_binding)
                        else np.asarray([-10.0] * k)
                    )
                    # P(stopped) = 1 - P(not stopped)
                    # Use gp_h0 directly
                    prob_cross = gp_h0.compute_crossing_probability(
                        t=info_times[: i + 1],
                        upper=temp_a[: i + 1],
                        lower=temp_b[: i + 1] if tails == 1 else -temp_a[: i + 1],
                        method="numerical_integration",
                    )
                    return float(prob_cross - efficacy_targets[i])

                low, high = 0.0, 20.0
                if obj_a(low) * obj_a(high) > 0:
                    a[i] = high if obj_a(high) < 0 else low
                else:
                    res = root_scalar(
                        obj_a, bracket=[low, high], method="brentq", xtol=1e-6
                    )
                    a[i] = res.root

            # Solve futility bound b[i]
            if b is not None and futility_targets is not None:

                def obj_b(val: float) -> float:
                    temp_b = b.copy()
                    temp_b[i] = val
                    temp_a = (
                        a.copy()
                        if (a is not None and futility_binding)
                        else np.asarray([10.0] * k)
                    )
                    prob_cross = gp_h1.compute_crossing_probability(
                        t=info_times[: i + 1],
                        upper=temp_a[: i + 1],
                        lower=temp_b[: i + 1],
                        method="numerical_integration",
                    )
                    return float(prob_cross - futility_targets[i])

                low, high = -10.0, 10.0
                if obj_b(low) * obj_b(high) > 0:
                    b[i] = high if obj_b(high) < 0 else low
                else:
                    res = root_scalar(
                        obj_b, bracket=[low, high], method="brentq", xtol=1e-6
                    )
                    b[i] = res.root

        return a, b

    # =========================================================================
    # High-Level API (uses policy dispatching)
    # =========================================================================

    def solve_boundaries(
        self,
        drift: Optional[float] = None,
        method: Literal[
            "simulation", "numerical_integration"
        ] = "numerical_integration",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve for boundaries based on config stopping policy."""
        if self.config.stopping_policy:
            a, b = self.config.stopping_policy.solve(self)
            if a is not None or b is not None:
                return a, b

        return self._solve_from_spending(drift, method=method)

    def _solve_from_spending(
        self,
        drift: Optional[float],
        method: Literal[
            "simulation", "numerical_integration"
        ] = "numerical_integration",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve using spending functions from config or policy."""
        eff_sched = self.config.efficacy_spending
        fut_sched = self.config.futility_spending

        if (
            eff_sched is None
            and self.config.stopping_policy
            and isinstance(self.config.stopping_policy, SpendingFunctionStoppingPolicy)
        ):
            eff_sched = self.config.stopping_policy.efficacy_spending

        if (
            fut_sched is None
            and self.config.stopping_policy
            and isinstance(self.config.stopping_policy, SpendingFunctionStoppingPolicy)
        ):
            fut_sched = self.config.stopping_policy.futility_spending

        if drift is None:
            drift = 1.0

        t = (
            self.config.spending_times
            if self.config.spending_times is not None
            else self.config.info_times
        )
        eff_targets = eff_sched.cumulative(t) if eff_sched else None
        fut_targets = fut_sched.cumulative(t) if fut_sched else None

        return self.solve_boundaries_from_cumulative_targets(
            info_times=self.config.info_times,
            efficacy_targets=eff_targets,
            futility_targets=fut_targets,
            drift=drift,
            efficacy_binding=self.config.efficacy_binding,
            futility_binding=self.config.futility_binding,
            tails=self.config.tails,
            method=method,
        )

    # =========================================================================
    # Compatibility / Utility Methods
    # =========================================================================

    def find_critical_value(
        self, shape: NDArray[np.float64], alpha: float, tails: Optional[int] = None
    ) -> float:
        """Alias for find_boundary_constant (backward compatibility)."""
        return self.find_boundary_constant(
            info_times=self.config.info_times,
            shape=shape,
            target_probability=alpha,
            tails=tails or self.config.tails,
            drift=0.0,
        )

    def evaluate_design(self, drift: float) -> Dict[str, float]:
        """Compute operating characteristics."""
        return {"prob": 0.0, "asn": 0.0}

    def solve_boundary_constant(
        self,
        info_times: Sequence[float],
        alpha: float,
        shape_type: str = "pocock",
        tails: int = 2,
        shape_params: Optional[Dict[str, float]] = None,
    ) -> float:
        """Legacy API for solving shape-based boundary constants."""
        t = np.asarray(info_times)
        if shape_type == "pocock":
            shape = np.ones_like(t)
        elif shape_type == "obrien_fleming":
            shape = 1.0 / np.sqrt(t)
        elif shape_type == "wang_tsiatis":
            delta = shape_params.get("delta_wt", 0.25) if shape_params else 0.25
            shape = t ** (delta - 0.5)
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        return self.find_boundary_constant(t, shape, alpha, tails=tails)

    def compute_rejection_probability(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        drift: float = 0.0,
        tails: Optional[int] = None,
        samples: Optional[np.ndarray] = None,
        futility_boundaries: Optional[Sequence[float]] = None,
    ) -> float:
        """Compute rejection probability (compatibility wrapper)."""
        t = np.asarray(info_times)
        u = np.asarray(boundaries)
        target_tails = tails if tails is not None else self.config.tails
        fut_b = (
            np.asarray(futility_boundaries) if futility_boundaries is not None else None
        )

        if samples is None:
            gp = CanonicalGaussianProcess(drift=drift, rng=self._rng)
            samples = gp.sample(t, self.config.n_sims * 2)
        else:
            samples = samples + drift * np.sqrt(t)

        n_sims, k = samples.shape
        stopped_eff = np.zeros(n_sims, dtype=bool)
        stopped_any = np.zeros(n_sims, dtype=bool)

        for i in range(k):
            crossing_u = (samples[:, i] > u[i]) & ~stopped_any
            if target_tails == 2:
                crossing_l = (samples[:, i] < -u[i]) & ~stopped_any
                stopped_eff |= crossing_u | crossing_l
                stopped_any |= crossing_u | crossing_l
            else:
                stopped_eff |= crossing_u
                stopped_any |= crossing_u
                if fut_b is not None:
                    crossing_fut = (samples[:, i] < fut_b[i]) & ~stopped_any
                    stopped_any |= crossing_fut

        return float(np.mean(stopped_eff))

    def solve_drift(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        target_power: float,
        tails: Optional[int] = None,
        futility_boundaries: Optional[Sequence[float]] = None,
        bracket: Tuple[float, float] = (1.0, 10.0),
        method: Literal["simulation", "numerical_integration"] = "simulation",
    ) -> float:
        """Solve for drift that yields target power."""
        t = np.asarray(info_times)
        u = np.asarray(boundaries)
        target_tails = tails if tails is not None else self.config.tails

        if method == "numerical_integration":

            def f(d: float) -> float:
                return (
                    self.compute_crossing_probability(
                        info_times=t,
                        upper=u,
                        lower=-u if target_tails == 2 else None,
                        drift=d,
                        method="numerical_integration",
                    )
                    - target_power
                )

        else:
            gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
            samples_h0 = gp_h0.sample(t, self.config.n_sims * 2)

            def f(d: float) -> float:
                return (
                    self.compute_rejection_probability(
                        info_times,
                        boundaries,
                        drift=d,
                        tails=target_tails,
                        samples=samples_h0,
                        futility_boundaries=futility_boundaries,
                    )
                    - target_power
                )

        res = root_scalar(f, bracket=bracket, xtol=1e-4)
        return float(res.root)
