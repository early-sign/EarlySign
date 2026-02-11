"""Canonical Joint Model for Group Sequential Tests.

Provides generic computation APIs for boundary solving and probability
calculations based on the canonical joint distribution of Z-statistics.

Examples:
    >>> import numpy as np
    >>> import earlysign.schema.ES3.GST as GST
    >>> from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import CanonicalJointModel, Config
    >>> from earlysign.v1.methods.group_sequential.shared.spending import OBrienFlemingSpending
    >>>
    >>> # --- Test: Model from Spec Basic ---
    >>> info_times = [0.5, 1.0]
    >>> spec = GST.Protocol(
    ...     name="Test Protocol",
    ...     task=GST.TaskSpec(
    ...         kind="group_sequential",
    ...         arms=["C", "T"],
    ...         response_type=GST.ResponseType.BINARY,
    ...         efficacy=GST.EfficacyRequirement(alpha=0.025),
    ...         futility=GST.FutilityRequirement(power=0.9),
    ...         hypotheses=GST.HypothesisSpec(
    ...             h_null_description="H0", h_alt_description="H1",
    ...             test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
    ...             target_effect=GST.BinaryEffectSize(proportions={"C": 0.1, "T": 0.15})
    ...         )
    ...     ),
    ...     method=GST.MethodSpec(
    ...         kind="group_sequential",
    ...         stopping_policy=GST.StoppingPolicySpec(
    ...             statistic=GST.TwoArmBinomialZ(variance_estimation=GST.VarianceEstimation.POOLED),
    ...             strategy=GST.AlphaSpendingStrategy(
    ...                 spending_fn=GST.SpendingFunction(family="obrien_fleming"),
    ...                 budget=0.025,
    ...                 sided=GST.Sided.ONE,
    ...                 statistical_model=GST.CanonicalGaussianModel(),
    ...             ),
    ...             timer=GST.SampleSizeTimer(unit=GST.Unit.INDIVIDUALS, max_sample_size=100),
    ...             schedule=GST.FixedSchedule(analyses=info_times)
    ...         ),
    ...     )
    ... )
    >>> model = CanonicalJointModel.from_spec(spec, n_sims=5000)
    >>> model.config.alpha
    0.025
    >>> np.allclose(model.config.info_times, [0.5, 1.0])
    True
    >>>
    >>> # --- Test: Dual Boundary Solving (Binding) ---
    >>> info_times_arr = np.array([0.5, 1.0])
    >>> config = Config(
    ...     info_times=info_times_arr, alpha=0.025, power=0.9,
    ...     efficacy_spending=OBrienFlemingSpending(budget=0.025),
    ...     futility_spending=OBrienFlemingSpending(budget=0.1),
    ...     efficacy_binding=True, n_sims=5000, rng_seed=42, tails=1
    ... )
    >>> model = CanonicalJointModel(config)
    >>> a, b = model.solve_boundaries(drift=3.24)
    >>> len(a) == 2 and len(b) == 2
    True
    >>> bool(a[0] > a[1])  # OBF characteristic
    True
    >>> bool(b[0] < b[1])  # Futility characteristic
    True
"""

import warnings
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

# Internal numerical safety limits.
# These are kept local to avoid influencing general design logic.
_Z_SOLVER_LIMIT = 100.0
_SOLVER_BRACKET_HIGH = 100.0


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
        efficacy_binding: Is Efficacy boundary binding? (Affects futility calculation).
        futility_binding: Is Futility boundary binding? (Affects efficacy calculation).
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
        cls,
        spec: GST.Protocol,
        n_sims: int = 20000,
        rng_seed: Optional[int] = None,
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
                objective,
                bracket=[0.0, _Z_SOLVER_LIMIT],
                method="brentq",
                xtol=1e-3,
            )
            return float(res.root)
        except ValueError:
            if objective(_Z_SOLVER_LIMIT) > 0:
                warnings.warn(
                    f"Boundary constant exceeding standard search range ({_Z_SOLVER_LIMIT}). "
                    "A wider bracket will be used, but this may indicate an extreme alpha or design requirement.",
                    UserWarning,
                    stacklevel=2,
                )
                res = root_scalar(
                    objective,
                    bracket=[_Z_SOLVER_LIMIT, 5.0 * _Z_SOLVER_LIMIT],
                    method="brentq",
                    xtol=1e-3,
                )
                return float(res.root)
            raise

    def solve_boundaries_from_cumulative_targets(
        self,
        info_times: NDArray[np.float64],
        efficacy_targets: Optional[NDArray[np.float64]] = None,
        futility_targets: Optional[NDArray[np.float64]] = None,
        drift: Optional[float] = None,
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
            efficacy_binding: Is Efficacy boundary binding? (Affects futility calculation).
            futility_binding: Is Futility boundary binding? (Affects efficacy calculation).
            tails: 1 or 2.

        Returns:
            Tuple of (efficacy_boundaries, futility_boundaries).
        """
        t = info_times
        k = len(t)

        if futility_targets is not None and drift is None:
            raise ValueError(
                "Standardized drift must be provided to solve for futility boundaries."
            )

        # Fallback for efficacious-only simulation path if needed by underlying GP,
        # but drift is only used for H1 GP simulation.
        safe_drift = drift if drift is not None else 0.0

        if method == "numerical_integration":
            return self._solve_numerical(
                info_times,
                efficacy_targets,
                futility_targets,
                safe_drift,
                efficacy_binding,
                futility_binding,
                tails,
            )

        # Simulation path
        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        z_sims_h0 = gp_h0.sample(t, self.config.n_sims)

        gp_h1 = CanonicalGaussianProcess(drift=safe_drift, rng=self._rng)
        z_sims_h1 = gp_h1.sample(t, self.config.n_sims)

        a = np.zeros(k) if efficacy_targets is not None else None
        b = np.full(k, -np.inf) if futility_targets is not None else None

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
                    a[i] = np.inf if i < k - 1 else (a[i - 1] if i > 0 else 2.0)
                elif num_rem < 10:
                    a[i] = -np.inf
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
                    b[i] = -np.inf
                elif num_rem_h1 < 10:
                    b[i] = np.inf
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
                if futility_binding:
                    just_fut_h0 = (~stopped_h0) & (z_sims_h0[:, i] < b[i])
                    stopped_h0 |= just_fut_h0

        return a, b

    def solve_next_boundary(
        self,
        previous_times: Sequence[float],
        current_t: float,
        target_cumulative_prob: float,
        previous_efficacy: Optional[Sequence[float]] = None,
        previous_futility: Optional[Sequence[float]] = None,
        rule_type: str = "efficacy",
        drift: float = 0.0,
    ) -> float:
        """
        Solve for the next boundary point given history and a cumulative probability target.
        This enables true information-driven spending at arbitrary information times.
        """
        times = np.concatenate([np.asarray(previous_times), [current_t]])
        k = len(times)
        tails = self.config.tails

        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        gp_h1 = CanonicalGaussianProcess(drift=drift, rng=self._rng)

        # Prepare boundary masks
        eff = (
            np.asarray(list(previous_efficacy) + [np.inf])
            if previous_efficacy is not None
            else None
        )
        fut = (
            np.asarray(list(previous_futility) + [-np.inf])
            if previous_futility is not None
            else None
        )

        if rule_type == "efficacy":
            if eff is None:
                eff = np.full(k, np.inf)

            # Efficacy binding logic (usually true)
            # We want P(Cross eff or Cross fut_binding) = target_cumulative_prob
            binding_fut = (
                fut
                if (fut is not None and self.config.futility_binding)
                else np.full(k, -np.inf)
            )

            def obj_a(val: float) -> float:
                temp_eff = eff.copy()
                temp_eff[-1] = val
                prob = gp_h0.compute_crossing_probability(
                    t=times,
                    upper=temp_eff,
                    lower=binding_fut if tails == 1 else -temp_eff,
                    method="numerical_integration",
                )
                return float(prob - target_cumulative_prob)

            # Robust check to avoid BrentQ failure on extremely small alpha spent or large B
            f_0 = obj_a(0.0)
            f_lim = obj_a(_SOLVER_BRACKET_HIGH)
            if f_0 * f_lim > 0:
                # If both are same sign, the root is likely outside [0, _SOLVER_BRACKET_HIGH]
                # Since f_0 (at val=0) is usually 0.5 - target (> 0),
                # if f_lim is also positive, the boundary is > _SOLVER_BRACKET_HIGH.
                warnings.warn(
                    f"Efficacy boundary solver reached technical limit ({_SOLVER_BRACKET_HIGH}). "
                    "This usually happens when using extremely aggressive spending functions (like OBF) at early looks.",
                    UserWarning,
                    stacklevel=2,
                )
                return _SOLVER_BRACKET_HIGH if f_0 > 0 else 0.0

            res = root_scalar(
                obj_a, bracket=[0.0, _SOLVER_BRACKET_HIGH], method="brentq", xtol=1e-6
            )
            return float(res.root)

        elif rule_type == "futility":
            if fut is None:
                fut = np.full(k, -np.inf)

            binding_eff = (
                eff
                if (eff is not None and self.config.efficacy_binding)
                else np.full(k, np.inf)
            )

            def obj_b(val: float) -> float:
                temp_fut = fut.copy()
                temp_fut[-1] = val
                prob = gp_h1.compute_crossing_probability(
                    t=times,
                    upper=binding_eff,
                    lower=temp_fut,
                    method="numerical_integration",
                )
                return float(prob - target_cumulative_prob)

            # Robust check for futility
            f_low = obj_b(-_Z_SOLVER_LIMIT)
            f_high = obj_b(_Z_SOLVER_LIMIT)
            if f_low * f_high > 0:
                warnings.warn(
                    f"Futility boundary solver reached technical limit (plus/minus {_Z_SOLVER_LIMIT}).",
                    UserWarning,
                    stacklevel=2,
                )
                return -np.inf if f_high < 0 else np.inf

            res = root_scalar(
                obj_b,
                bracket=[-_Z_SOLVER_LIMIT, _Z_SOLVER_LIMIT],
                method="brentq",
                xtol=1e-6,
            )
            return float(res.root)

        return 0.0

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
        b = np.full(k, -np.inf) if futility_targets is not None else None

        # Higher-level helpers for numerical integration
        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        gp_h1 = CanonicalGaussianProcess(drift=drift, rng=self._rng)

        for i in range(k):
            # Solve efficacy bound a[i]
            if a is not None and efficacy_targets is not None:
                temp_b = (
                    b.copy()
                    if (b is not None and futility_binding)
                    else np.asarray([-np.inf] * k)
                )

                # Calculate base probability (excluding efficacy stop at current step i)
                temp_a_base = a.copy()
                temp_a_base[i] = np.inf  # Approx +inf
                prob_base = gp_h0.compute_crossing_probability(
                    t=info_times[: i + 1],
                    upper=temp_a_base[: i + 1],
                    lower=temp_b[: i + 1] if tails == 1 else -temp_a_base[: i + 1],
                    method="numerical_integration",
                )

                # Target probability for this step's efficacy is the marginal increase
                margin = efficacy_targets[i] - (
                    efficacy_targets[i - 1] if i > 0 else 0.0
                )
                target_prob = prob_base + margin

                def obj_a(val: float) -> float:
                    temp_a = a.copy()
                    temp_a[i] = val
                    # temp_b defined above
                    prob_cross = gp_h0.compute_crossing_probability(
                        t=info_times[: i + 1],
                        upper=temp_a[: i + 1],
                        lower=temp_b[: i + 1] if tails == 1 else -temp_a[: i + 1],
                        method="numerical_integration",
                    )
                    return float(prob_cross - target_prob)

                low, high = 0.0, _SOLVER_BRACKET_HIGH
                if obj_a(low) * obj_a(high) > 0:
                    warnings.warn(
                        f"Numerical efficacy boundary i={i} reached technical limit ({high}).",
                        UserWarning,
                        stacklevel=2,
                    )
                    a[i] = high if obj_a(high) < 0 else low
                else:
                    res = root_scalar(
                        obj_a, bracket=[low, high], method="brentq", xtol=1e-6
                    )
                    a[i] = res.root

            # Solve futility bound b[i]
            if b is not None and futility_targets is not None:
                temp_a = (
                    a.copy()
                    if (a is not None and efficacy_binding)
                    else np.asarray([np.inf] * k)
                )

                # Calculate base probability (excluding futility stop at current step i)
                # We use np.inf for numerical stability in existing routines
                temp_b_base = b.copy()
                temp_b_base[i] = -np.inf
                prob_base = gp_h1.compute_crossing_probability(
                    t=info_times[: i + 1],
                    upper=temp_a[: i + 1],
                    lower=temp_b_base[: i + 1],
                    method="numerical_integration",
                )

                # Target probability for this step's futility is the marginal increase
                margin = futility_targets[i] - (
                    futility_targets[i - 1] if i > 0 else 0.0
                )
                target_prob = prob_base + margin

                def obj_b(val: float) -> float:
                    temp_b = b.copy()
                    temp_b[i] = val
                    prob_cross = gp_h1.compute_crossing_probability(
                        t=info_times[: i + 1],
                        upper=temp_a[: i + 1],
                        lower=temp_b[: i + 1],
                        method="numerical_integration",
                    )
                    return float(prob_cross - target_prob)

                low, high = -_Z_SOLVER_LIMIT, _Z_SOLVER_LIMIT
                if obj_b(low) * obj_b(high) > 0:
                    warnings.warn(
                        f"Numerical futility boundary i={i} reached technical limit ({high}).",
                        UserWarning,
                        stacklevel=2,
                    )
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
        efficacy_spending: Optional[SpendingFunction] = None,
        futility_spending: Optional[SpendingFunction] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve for boundaries based on config stopping policy or provided spending.

        Args:
            drift: Drift parameter (optional).
            method: Computation method.
            efficacy_spending: Optional override for efficacy spending function.
            futility_spending: Optional override for futility spending function.
        """
        if (
            self.config.stopping_policy
            and efficacy_spending is None
            and futility_spending is None
        ):
            a, b = self.config.stopping_policy.solve(self)
            if a is not None or b is not None:
                return a, b

        return self._solve_from_spending(
            drift,
            method=method,
            efficacy_spending=efficacy_spending,
            futility_spending=futility_spending,
        )

    def solve_boundaries_from_policy(
        self, policy: StoppingPolicy
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve for boundaries using a specific StoppingPolicy."""
        return policy.solve(self)

    def _solve_from_spending(
        self,
        drift: Optional[float],
        method: Literal[
            "simulation", "numerical_integration"
        ] = "numerical_integration",
        efficacy_spending: Optional[SpendingFunction] = None,
        futility_spending: Optional[SpendingFunction] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve using spending functions from config, policy, or arguments."""
        eff_sched = efficacy_spending or self.config.efficacy_spending
        fut_sched = futility_spending or self.config.futility_spending

        if (
            eff_sched is None
            and self.config.stopping_policy
            and isinstance(self.config.stopping_policy, SpendingFunctionStoppingPolicy)
            and efficacy_spending is None
        ):
            eff_sched = self.config.stopping_policy.efficacy_spending

        if (
            fut_sched is None
            and self.config.stopping_policy
            and isinstance(self.config.stopping_policy, SpendingFunctionStoppingPolicy)
            and futility_spending is None
        ):
            fut_sched = self.config.stopping_policy.futility_spending

        if fut_sched and drift is None:
            raise ValueError(
                "Standardized drift must be provided to solve for futility boundaries "
                "from a spending function."
            )

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
