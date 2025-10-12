"""Simulation engine for power and operating characteristics estimation."""

from typing import Any, Dict

import numpy as np

from earlysign.stats.design.config import DesignSpec
from earlysign.stats.design.effects import EffectCalculator


class SimulationEngine:
    """Run simulations to estimate power and operating characteristics.

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> from earlysign.stats.design.boundaries import BoundaryCalculator
    >>> from earlysign.stats.design.effects import ProportionsEffectCalculator
    >>> spec = ProportionsDesignSpec()
    >>> spec.simulation.n_sims = 100  # Small for testing
    >>> boundaries = BoundaryCalculator.critical_values(spec)
    >>> calc = ProportionsEffectCalculator()
    >>> results = SimulationEngine.run_simulations(spec, boundaries, calc)
    >>> 'power' in results
    True
    >>> 0 <= results['power'] <= 1
    True
    """

    @staticmethod
    def simulate_trial(
        spec: DesignSpec,
        boundaries: Dict[str, Any],
        effect_calc: EffectCalculator,
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

        # Generate Z-statistics trajectory
        Z = np.zeros(k)
        for i in range(k):
            if stopped:
                Z[i] = Z[i - 1]  # Carry forward
            else:
                # Drift under alternative
                drift = effect_calc.standardized_effect(spec, t[i])
                # Brownian motion increment
                if i == 0:
                    Z[i] = rng.normal(drift * np.sqrt(t[i]), np.sqrt(t[i]))
                else:
                    dt = t[i] - t[i - 1]
                    Z[i] = Z[i - 1] + rng.normal(drift * np.sqrt(dt), np.sqrt(dt))

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
        spec: DesignSpec, boundaries: Dict[str, Any], effect_calc: EffectCalculator
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
        sample_info = effect_calc.sample_sizes(spec)
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
