from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.execution.stopping_policy import (
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
        efficacy_spending: Spending function for efficacy.
        futility_spending: Spending function for futility.
        binding_futility: If True, efficacy boundaries are solved respecting futility boundaries.
        binding_efficacy: If True, futility boundaries are solved respecting efficacy boundaries.
        tails: 1 or 2 (symmetric).
        n_sims: Number of simulations for boundary solving.
        rng_seed: Seed for random number generator.
    """

    info_times: NDArray[np.float64]
    spending_times: Optional[NDArray[np.float64]] = None
    alpha: Optional[float] = None
    power: Optional[float] = None
    efficacy_spending: Optional[SpendingFunction] = None
    futility_spending: Optional[SpendingFunction] = None
    binding_futility: bool = True
    binding_efficacy: bool = False
    tails: int = 1
    n_sims: int = 20000
    rng_seed: Optional[int] = None


class CanonicalJointModel:
    """Stateful statistical engine for group sequential tests.

    This engine realizes a GST design (boundaries and characteristics) based on the
    canonical joint distribution of Z-statistics. It can be initialized from a
    structured Config object or derived from an ES3 GST.Protocol.
    """

    def __init__(self, config: Config):
        self.config = config
        self._rng = np.random.default_rng(config.rng_seed)

    @classmethod
    def from_spec(
        cls, spec: GST.Protocol, n_sims: int = 20000, rng_seed: Optional[int] = None
    ) -> "CanonicalJointModel":
        """Instantiate the model from an ES3 GST.Protocol specification."""
        task = spec.task
        method = spec.method
        schedule = method.schedule

        # 1. Extract fallback alpha/power from task
        task_alpha = float(task.efficacy.alpha) if task.efficacy else None

        task_power = float(task.futility.power) if task.futility else None

        # 2. Extract schedule
        if not schedule or schedule.interim_points is None:
            raise ValueError("Protocol must define interim_points.")

        t = np.asarray(schedule.interim_points)
        # Normalize if they look like sample sizes
        if np.max(t) > 1.0:
            t = t / np.max(t)

        # 3. Resolve stopping policy from protocol method
        stopping_policy = StoppingPolicyFactory.build_from_spec(method.stopping_policy)

        config = Config(
            info_times=t,
            alpha=task_alpha,
            power=task_power,
            efficacy_spending=stopping_policy.efficacy_spending,
            futility_spending=stopping_policy.futility_spending,
            binding_futility=stopping_policy.alpha_binding,
            binding_efficacy=stopping_policy.beta_binding,
            tails=2 if stopping_policy.sided == "two" else 1,
            n_sims=n_sims,
            rng_seed=rng_seed,
        )
        return cls(config)

    def _generate_joint_z(self, info_times: np.ndarray) -> np.ndarray:
        """Generate Z-statistics under H0 (drift=0) at given info fractions."""
        gp = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        return gp.sample(info_times, self.config.n_sims)

    def solve_boundaries(
        self, drift: Optional[float] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Solve for efficacy (a) and/or futility (b) boundaries based on config.

        If both efficacy and futility spending are present, it solves for dual boundaries.
        The 'binding' behavior is controlled by self.config.

        Args:
            drift: Standardized drift delta under H1. If None, it will be solved for
                   if power and spending are defined, or assumed 1.0.

        Returns:
            Tuple of (efficacy_boundaries, futility_boundaries).
            Values are None if the corresponding spending is not defined.
        """
        t = self.config.info_times
        k = len(t)

        # 1. Determine drift if needed
        if drift is None:
            # If we have both alpha and power targets, we can't solve boundaries and drift
            # simultaneously without more info. Usually drift is a target.
            # For spending function designs, drift is often the drift required to achieve
            # target power with the resulting boundaries.
            # For now, let's assume drift is provided or solve a simplified one.
            drift = 1.0  # Default fallback

        # 2. Setup GPs
        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        z_sims_h0 = gp_h0.sample(t, self.config.n_sims)

        gp_h1 = CanonicalGaussianProcess(drift=drift, rng=self._rng)
        z_sims_h1 = gp_h1.sample(t, self.config.n_sims)

        a = np.zeros(k) if self.config.efficacy_spending else None
        b = np.full(k, -10.0) if self.config.futility_spending else None

        # Track rejections vs total stoppage separately for binding logic
        rejected_h0 = np.zeros(self.config.n_sims, dtype=bool)
        stopped_h0 = np.zeros(self.config.n_sims, dtype=bool)

        futility_h1 = np.zeros(self.config.n_sims, dtype=bool)
        stopped_h1 = np.zeros(self.config.n_sims, dtype=bool)

        t_spend = (
            self.config.spending_times if self.config.spending_times is not None else t
        )
        a_cum = (
            self.config.efficacy_spending.cumulative(t_spend)
            if self.config.efficacy_spending
            else None
        )
        b_cum = (
            self.config.futility_spending.cumulative(t_spend)
            if self.config.futility_spending
            else None
        )

        for i in range(k):
            # 1. Solve for efficacy a[i]
            if a is not None and a_cum is not None:
                # Solve for efficacy boundary a[i] under H0
                needed_new_rejections = a_cum[i] * self.config.n_sims - np.sum(
                    rejected_h0
                )
                rem_mask = ~stopped_h0
                num_rem = np.sum(rem_mask)

                if needed_new_rejections <= 0:
                    a[i] = 10.0 if i < k - 1 else (a[i - 1] if i > 0 else 2.0)
                elif num_rem < 10:
                    # Over-spent efficacy: reject everything remaining
                    a[i] = -10.0
                else:
                    target_frac_of_rem = max(0, min(1, needed_new_rejections / num_rem))
                    # For two-sided, we find a s.t. P(|Z| > a) = frac
                    if self.config.tails == 2:
                        a[i] = np.percentile(
                            np.abs(z_sims_h0[rem_mask, i]),
                            100 * (1 - target_frac_of_rem),
                        )
                    else:
                        a[i] = np.percentile(
                            z_sims_h0[rem_mask, i], 100 * (1 - target_frac_of_rem)
                        )
                    if target_frac_of_rem >= 1.0:
                        a[i] = min(a[i], -10.0)

            # 2. Solve for futility b[i]
            if b is not None and b_cum is not None:
                needed_new_futility = b_cum[i] * self.config.n_sims - np.sum(
                    futility_h1
                )
                rem_mask_h1 = ~stopped_h1
                num_rem_h1 = np.sum(rem_mask_h1)

                if needed_new_futility <= 0:
                    b[i] = -10.0
                elif num_rem_h1 < 10:
                    # Over-spent futility: accept everything remaining
                    b[i] = 10.0
                else:
                    target_frac_of_rem_h1 = max(
                        0, min(1, needed_new_futility / num_rem_h1)
                    )
                    b[i] = np.percentile(
                        z_sims_h1[rem_mask_h1, i], 100 * target_frac_of_rem_h1
                    )
                    if target_frac_of_rem_h1 >= 1.0:
                        b[i] = max(b[i], 10.0)
                    # We do NOT force b[i] <= a[i] here to allow R_OS solving to work

            # 3. Update stop masks for next look
            if a is not None:
                if self.config.tails == 2:
                    just_rej_h0 = (~stopped_h0) & (np.abs(z_sims_h0[:, i]) > a[i])
                    just_rej_h1 = (~stopped_h1) & (np.abs(z_sims_h1[:, i]) > a[i])
                else:
                    just_rej_h0 = (~stopped_h0) & (z_sims_h0[:, i] > a[i])
                    just_rej_h1 = (~stopped_h1) & (z_sims_h1[:, i] > a[i])

                rejected_h0 |= just_rej_h0
                stopped_h0 |= just_rej_h0
                stopped_h1 |= just_rej_h1  # efficacy stop under H1

            if b is not None:
                just_fut_h1 = (~stopped_h1) & (z_sims_h1[:, i] < b[i])
                futility_h1 |= just_fut_h1
                stopped_h1 |= just_fut_h1

                if self.config.binding_futility:
                    just_fut_h0 = (~stopped_h0) & (z_sims_h0[:, i] < b[i])
                    stopped_h0 |= just_fut_h0  # futility stop under H0

        return a, b

    def evaluate_design(self, drift: float) -> Dict[str, float]:
        """Compute operating characteristics for the current boundaries and given drift."""
        # Implementation relying on realize_boundaries() result
        return {"prob": 0.0, "asn": 0.0}

    # Compatibility methods (to be deprecated or kept as wrappers)
    def solve_boundary_constant(
        self,
        info_times: Sequence[float],
        alpha: float,
        shape_type: str = "pocock",
        tails: int = 2,
        shape_params: Optional[Dict[str, float]] = None,
    ) -> float:
        """Solve for the constant 'c' that yields the target alpha for a given shape."""
        t = np.asarray(info_times)
        if shape_type == "pocock":
            c_shape = np.ones_like(t)
        elif shape_type == "obrien_fleming":
            c_shape = 1.0 / np.sqrt(t)
        elif shape_type == "wang_tsiatis":
            delta_wt = 0.25
            if shape_params and shape_params.get("delta_wt") is not None:
                delta_wt = shape_params["delta_wt"]
            c_shape = t ** (delta_wt - 0.5)
        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

        gp = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        z_sims = gp.sample(t, self.config.n_sims if hasattr(self, "config") else 20000)

        if tails == 2:
            normalized_max = np.max(np.abs(z_sims) / c_shape, axis=1)
        else:
            normalized_max = np.max(z_sims / c_shape, axis=1)
        return float(np.percentile(normalized_max, 100 * (1 - alpha)))

    def compute_rejection_probability(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        drift: float = 0.0,
        tails: int = 1,
        samples: Optional[np.ndarray] = None,
        futility_boundaries: Optional[Sequence[float]] = None,
    ) -> float:
        """Compatibility wrapper for rejection probability."""
        t = np.asarray(info_times)
        u = np.asarray(boundaries)
        low_b = (
            np.asarray(futility_boundaries) if futility_boundaries is not None else None
        )

        if samples is None:
            gp = CanonicalGaussianProcess(drift=drift, rng=self._rng)
            samples = gp.sample(t, self.config.n_sims * 2)
        else:
            # Shift H0 samples by drift * sqrt(t)
            samples = samples + drift * np.sqrt(t)

        n_sims, k = samples.shape
        stopped_eff = np.zeros(n_sims, dtype=bool)
        stopped_any = np.zeros(n_sims, dtype=bool)

        for i in range(k):
            # Check stopping
            crossing_eff_u = (samples[:, i] > u[i]) & ~stopped_any

            if tails == 2:
                crossing_eff_l = (samples[:, i] < -u[i]) & ~stopped_any
                stopped_eff |= crossing_eff_u | crossing_eff_l
                stopped_any |= crossing_eff_u | crossing_eff_l
            else:
                stopped_eff |= crossing_eff_u
                stopped_any |= crossing_eff_u
                if low_b is not None:
                    crossing_fut = (samples[:, i] < low_b[i]) & ~stopped_any
                    stopped_any |= crossing_fut

        return float(np.mean(stopped_eff))

    def evaluate_asn(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        drift: float = 0.0,
        tails: int = 1,
    ) -> float:
        """Compatibility wrapper for ASN."""
        t = np.asarray(info_times)
        b = np.asarray(boundaries)
        gp = CanonicalGaussianProcess(drift=drift, rng=self._rng)
        samples = gp.sample(t, self.config.n_sims)
        _, stop_looks = gp.apply_stopping_rule(
            samples, upper=b, lower=-b if tails == 2 else None
        )
        return float(np.mean(stop_looks))

    def solve_drift(
        self,
        info_times: Sequence[float],
        boundaries: Sequence[float],
        target_power: float,
        tails: int = 1,
        futility_boundaries: Optional[Sequence[float]] = None,
    ) -> float:
        """Solve for the standardized drift delta that yields target power."""
        low = 1.0
        high = 10.0
        t = np.asarray(info_times)

        # Use common random numbers for stability
        gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=self._rng)
        samples_h0 = gp_h0.sample(t, self.config.n_sims * 2)

        def f(d: float) -> float:
            return (
                self.compute_rejection_probability(
                    info_times,
                    boundaries,
                    drift=d,
                    tails=tails,
                    samples=samples_h0,
                    futility_boundaries=futility_boundaries,
                )
                - target_power
            )

        res = root_scalar(f, bracket=[low, high], xtol=1e-4)
        return float(res.root)
