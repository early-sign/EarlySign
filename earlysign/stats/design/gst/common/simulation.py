"""Simulation engine for power and operating characteristics estimation."""

from typing import Any, Dict

import numpy as np

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.essentials.schemes.protocols import EffectSizeCalculator


class SimulationEngine:
    """Monte Carlo simulation engine for group sequential trial operating characteristics.

    This class estimates the statistical properties of a group sequential design
    through simulation under specified hypotheses. It generates thousands of
    synthetic trial trajectories following Brownian motion with drift, applies
    the sequential decision rules at each analysis, and aggregates outcomes.

    Key operating characteristics computed:
    - **Power**: Probability of rejecting H0 when alternative is true
    - **Type I error**: Probability of false positive (when simulating under H0)
    - **Expected sample size (ASN)**: Average sample size across trials
    - **Stopping probabilities**: Distribution of when trials stop
    - **Rejection patterns**: Whether stopped for efficacy, futility, or completed

    The simulation uses the canonical joint distribution of Z-statistics at
    sequential analyses, which follow multivariate normal with known covariance
    structure determined by information times.

    Methods
    -------
    simulate_trial(spec, boundaries, effect_calc, rng) -> Dict[str, Any]
        Simulate one complete trial trajectory. Returns dictionary with
        Z statistics, stopping time, reason, and rejection decision.

    run_simulations(spec, boundaries, effect_calc) -> Dict[str, Any]
        Run n_sims independent trials and aggregate results. Returns
        comprehensive operating characteristics including power, ASN,
        stopping probabilities, and empirical distributions.

    Examples
    --------
    >>> from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    >>> from earlysign.stats.design.gst.common.adapter import BoundaryCalculator
    >>> from earlysign.stats.essentials.schemes.two_proportions.effect_size import TwoProportionsEffectSizeCalculator
    >>>
    >>> spec = ProportionsDesignSpec()
    >>> spec.simulation.n_sims = 1000  # Use more for production
    >>> boundaries = BoundaryCalculator.critical_values(spec)
    >>> calc = TwoProportionsEffectSizeCalculator()
    >>> results = SimulationEngine.run_simulations(spec, boundaries, calc)
    >>>
    >>> # Check power
    >>> 0 <= results['power'] <= 1
    True
    >>>
    >>> # Expected sample size
    >>> bool(results['expected_sample_size'] > 0)
    True
    >>>
    >>> # Stopping distribution sums to n_sims
    >>> sum(results['stop_distribution'].values()) == spec.simulation.n_sims
    True

    Notes
    -----
    The simulation assumes:
    1. Independent increments of the Z-process (valid under large samples)
    2. Known information times (deterministic accrual)
    3. Continuous monitoring approximation (analysis times are fixed)

    For two-proportion tests, the standardized effect is:
        θ = (p_treatment - p_control) / SE(p_treatment - p_control)

    The Z-statistic at information time t follows:
        Z_t ~ N(θ * sqrt(I_t), 1) under H1
    where I_t is information accrued by time t.

    See Also
    --------
    BoundaryCalculator : Computes critical values for boundaries
    EffectSizeCalculator : Computes standardized effects for test types
    DesignLab : Orchestrator combining boundaries and simulation
    DesignSpec : Specifies simulation parameters (n_sims, seed, etc.)

    References
    ----------
    .. [1] Jennison, C., & Turnbull, B. W. (1999). Group Sequential Methods
           with Applications to Clinical Trials. Chapman and Hall/CRC.
    .. [2] Proschan, M. A., Lan, K. K. G., & Wittes, J. T. (2006).
           Statistical Monitoring of Clinical Trials: A Unified Approach.
           Springer.
    """

    @staticmethod
    def simulate_trial(
        spec: DesignSpec,
        boundaries: Dict[str, Any],
        effect_calc: EffectSizeCalculator,
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Simulate a single trial.

        Args:
            spec: Design specification
            boundaries: Boundary values from BoundaryCalculator
            effect_calc: Effect calculator for the test type
            rng: Random number generator

        Returns:
            Dictionary with Z trajectory, stopping info, and rejection decision
        """
        t = boundaries["info_times"]
        z_upper = boundaries["z_efficacy"]
        z_lower = boundaries["z_futility"]

        k = len(t)
        stopped = False
        stop_analysis = None
        stop_reason = None
        allocation_ratio = spec.allocation.alloc_ratio
        info_schedule = spec.resolved_info_times()

        # Generate Z-statistics trajectory
        Z = np.zeros(k)
        for i in range(k):
            if stopped:
                Z[i] = Z[i - 1]  # Carry forward
            else:
                # CORRECTED: standardized_effect(t) returns E[Z(t)] = theta * sqrt(t)
                # For Brownian motion with drift: Z(t) ~ N(theta * sqrt(t), sqrt(t))
                # So we use standardized_effect directly as the mean
                mean_at_t = effect_calc.standardized_effect(
                    spec.effect,
                    t[i],
                    sample_size=spec.sample_size,
                    allocation_ratio=allocation_ratio,
                    info_times=info_schedule,
                )

                # Generate from joint distribution using incremental form
                if i == 0:
                    # Z(t[0]) ~ N(theta * sqrt(t[0]), sqrt(t[0]))
                    Z[i] = rng.normal(mean_at_t, np.sqrt(t[i]))
                else:
                    # Z(t[i]) | Z(t[i-1]) has conditional mean and variance
                    # E[Z(t[i]) | Z(t[i-1])] = Z(t[i-1]) + theta * sqrt(t[i] - t[i-1])
                    # Var[Z(t[i]) | Z(t[i-1])] = t[i] - t[i-1]
                    mean_at_prev = effect_calc.standardized_effect(
                        spec.effect,
                        t[i - 1],
                        sample_size=spec.sample_size,
                        allocation_ratio=allocation_ratio,
                        info_times=info_schedule,
                    )
                    conditional_mean = Z[i - 1] + (mean_at_t - mean_at_prev)
                    dt = t[i] - t[i - 1]
                    Z[i] = rng.normal(conditional_mean, np.sqrt(dt))

                # Check boundaries
                if Z[i] >= z_upper[i]:
                    stopped = True
                    stop_analysis = i + 1
                    stop_reason = "efficacy"
                elif z_lower is not None and Z[i] <= z_lower[i]:
                    stopped = True
                    stop_analysis = i + 1
                    stop_reason = "futility"

        if not stopped:
            stop_analysis = k
            stop_reason = "final"

        return {
            "Z": Z,
            "stopped_at": stop_analysis,
            "reason": stop_reason,
            "reject_h0": stop_reason == "efficacy",
        }

    @staticmethod
    def run_simulations(
        spec: DesignSpec, boundaries: Dict[str, Any], effect_calc: EffectSizeCalculator
    ) -> Dict[str, Any]:
        """Run full simulation study.

        Args:
            spec: Design specification
            boundaries: Boundary values
            effect_calc: Effect calculator

        Returns:
            Dictionary with power, expected sample size, and stopping distribution
        """
        rng = np.random.default_rng(spec.simulation.seed)

        results = []
        for _ in range(spec.simulation.n_sims):
            results.append(
                SimulationEngine.simulate_trial(spec, boundaries, effect_calc, rng)
            )

        # Aggregate results
        rejections = sum(r["reject_h0"] for r in results)
        power = rejections / spec.simulation.n_sims

        # Stop distribution
        stop_dist: Dict[int, int] = {}
        for r in results:
            a = r["stopped_at"]
            stop_dist[a] = stop_dist.get(a, 0) + 1

        # Expected sample size
        sample_info = effect_calc.sample_sizes(
            spec.sample_size,
            allocation_ratio=spec.allocation.alloc_ratio,
            info_times=spec.resolved_info_times(),
        )
        if "n_total" in sample_info:
            n_per_analysis = sample_info["n_total"]
        elif "events" in sample_info:
            n_per_analysis = sample_info["events"]
        else:
            # Fallback
            n_per_analysis = np.zeros(len(boundaries["info_times"]))

        ess = sum(
            stop_dist.get(i + 1, 0) * n_per_analysis[i]
            for i in range(len(boundaries["info_times"]))
        )
        ess /= spec.simulation.n_sims

        return {
            "power": power,
            "expected_sample_size": ess,
            "stop_distribution": stop_dist,
            "max_sample_size": int(n_per_analysis[-1]),
            "results": results,
        }
