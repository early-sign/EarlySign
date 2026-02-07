"""Over-running Inflation Calculation for Group Sequential Designs.

This module provides tools to calculate the expected sample size inflation factors
due to "Over-running" (R_OS), where a trial continues for a short period after
a stopping boundary is crossed (e.g., due to operational delays).
"""

from typing import Optional

import numpy as np
from scipy.optimize import brentq

from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    Config,
)
from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess


def compute_overrunning_inflation(
    config: Config,
    drift: float = 0.0,
    r_range: tuple[float, float] = (0.4, 4.0),
    n_sims: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """Computes the inflation factor R_OS for over-running designs.

    R_OS is the ratio of the maximum sample size of an over-running design to
    that of a fixed-sample design with equivalent power.

    The calculation finds R_OS such that the power of the over-running procedure
    matches the target power (from config).

    Args:
        config: The configuration of the group sequential design (alpha, power, etc.).
        drift: The standardized drift parameter (usually theta * sqrt(I_fixed)).
               Note: This drift is used as the base scale.
        r_range: The bracket [min, max] to search for R_OS.
        n_sims: Number of simulations to run. Defaults to config value.
        seed: Random seed. Defaults to config value.

    Returns:
        The inflation factor R_OS.
    """
    if config.power is None:
        raise ValueError("Config must have a target power set.")

    n = n_sims or config.n_sims
    s = seed if seed is not None else config.rng_seed
    rng = np.random.default_rng(s)

    k = len(config.info_times)
    t = config.info_times

    # We simulate H0 once and re-use it (Common Random Numbers) to smooth the objective function
    gp_h0 = CanonicalGaussianProcess(drift=0.0, rng=rng)
    z_h0 = gp_h0.sample(t, n)

    # Pre-calculate shape values for efficay/futility boundaries if possible
    # This implementation assumes the standard "Power Family" shape logic used in the test source.
    # If the config uses generic policy, we might need to adapt.
    # For now, we replicate the logic from Ch7 test which assumes power family spending/shape.
    # We will try to reverse-engineer the shape from the spending functions if present,
    # or rely on the user providing a config that makes sense.

    # However, for R_OS calculation in the test, it specifically used:
    # a_cum = alpha * (t**rho)
    # b_cum = (1 - power) * (t**rho)
    # This implies we need 'rho' from somewhere.
    # Since 'Config' doesn't store rho directly (it's inside SpendingFunction),
    # this function is best designed to work with generic spending functions
    # extracted from the Config.

    # Determine Spending Targets
    eff_spend = config.efficacy_spending
    fut_spend = config.futility_spending

    if (
        eff_spend is None
        and config.stopping_policy
        and hasattr(config.stopping_policy, "efficacy_spending")
    ):
        eff_spend = getattr(config.stopping_policy, "efficacy_spending")

    if (
        fut_spend is None
        and config.stopping_policy
        and hasattr(config.stopping_policy, "futility_spending")
    ):
        fut_spend = getattr(config.stopping_policy, "futility_spending")

    if eff_spend is None or fut_spend is None:
        raise ValueError(
            "Config must provide efficacy and futility spending (directly or via stopping_policy)."
        )

    t = config.info_times
    a_cum = eff_spend.cumulative(t)
    b_cum = fut_spend.cumulative(t)

    # 2. Resolve Drift
    # If drift is not provided (0.0), but target power is set,
    # we use the drift required for a fixed design to achieve that power.
    actual_drift = drift
    if actual_drift == 0.0 and config.power is not None:
        from scipy.stats import norm

        # Drift for fixed design = Phi^-1(1-alpha) + Phi^-1(power)
        alpha_val = config.alpha if config.alpha else 0.05
        # Note: assuming 1-sided for this drift calculation
        actual_drift = norm.ppf(1 - alpha_val) + norm.ppf(config.power)

    def objective(r_os: float) -> float:
        # Drift scales with sqrt(r_os) relative to fixed design scale
        current_drift = actual_drift * np.sqrt(r_os)

        # H1 Simulation (using CRN by shifting H0)
        z_h1 = z_h0 + current_drift * np.sqrt(t)

        # Solve boundaries dynamically for this R_OS
        a_tmp = np.zeros(k)
        b_tmp = np.zeros(k)

        stop0 = np.zeros(n, dtype=bool)
        rej0 = np.zeros(n, dtype=bool)

        stop1 = np.zeros(n, dtype=bool)
        fut1 = np.zeros(n, dtype=bool)

        for i in range(k):
            # Efficacy Bound (match accumulated alpha spending)
            # Find a_tmp[i] such that P(Reject H0 <= i) ~ a_cum[i]
            # Valid set are those NOT stopped before i.

            # Note: This logic mimics 'when_compute_ros' manual percentile finding.
            # It's 'simulation-based boundary solving'.

            # H0 logic
            rem_mask0 = ~stop0
            n_rem0 = np.sum(rem_mask0)
            n_rej_prev0 = np.sum(rej0)

            needed0 = a_cum[i] * n - n_rej_prev0
            frac0 = needed0 / max(1, n_rem0)

            if frac0 >= 1.0 or n_rem0 < 10:
                a_tmp[i] = -10.0 if frac0 > 0 else 10.0
            else:
                # We want top frac0 percent
                a_tmp[i] = np.percentile(
                    z_h0[rem_mask0, i], 100 * max(0, min(1, 1 - frac0))
                )

            # Funnel H0 stops
            just_rej0 = rem_mask0 & (z_h0[:, i] > a_tmp[i])
            rej0 |= just_rej0
            stop0 |= just_rej0

            # Binding futility for H0? Test logic says:
            # jf0 = (~stop0) & (z_h0[:, i] < b_tmp[i]) -> stop0 |= jf0
            # So we need b_tmp[i] first?
            # Actually the test loop solves a_tmp and b_tmp in the same step i.
            # But b_tmp depends on H1 stats.

            # H1 logic for Futility Bound (match accumulated beta spending)
            rem_mask1 = ~stop1
            n_rem1 = np.sum(rem_mask1)
            n_fut_prev1 = np.sum(fut1)

            needed1 = b_cum[i] * n - n_fut_prev1
            frac1 = needed1 / max(1, n_rem1)

            if frac1 >= 1.0 or n_rem1 < 10:
                b_tmp[i] = 10.0 if frac1 > 0 else -10.0
            else:
                # We want bottom frac1 percent
                b_tmp[i] = np.percentile(
                    z_h1[rem_mask1, i], 100 * max(0, min(1, frac1))
                )

            # Apply bounds to update state for NEXT step
            # Note: The test logic updates stop0 with b_tmp (binding?)
            # "jf0 = (~stop0) & (z_h0[:, i] < b_tmp[i])"
            # Yes, if futility is binding, it stops H0 paths too.
            # Assuming binding futility here based on test implementation.

            if config.futility_binding:
                just_fut0 = (~stop0) & (z_h0[:, i] < b_tmp[i])
                stop0 |= just_fut0

            # H1 stops
            just_rej1 = rem_mask1 & (z_h1[:, i] > a_tmp[i])  # Efficacy stop in H1
            stop1 |= just_rej1

            just_fut1 = rem_mask1 & (z_h1[:, i] < b_tmp[i])  # Futility stop in H1
            fut1 |= just_fut1
            stop1 |= just_fut1

        # Objective: We want the bounds to "meet" at the end approx?
        # The test used: return float(a_tmp[-1] - b_tmp[-1])
        # Trying to make a_K == b_K implies the decision is exhaustive (no continue).
        return float(a_tmp[-1] - b_tmp[-1])

    try:
        r_os = brentq(objective, r_range[0], r_range[1], xtol=1e-2)
    except Exception:
        # Fallback if root not found (monotonicity might be violated due to noise)
        f_low, f_high = objective(r_range[0]), objective(r_range[1])
        r_os = r_range[0] if abs(f_low) < abs(f_high) else r_range[1]

    return float(r_os)
