"""
Design-spec evaluation helpers for legacy GST UI/API flows.

This module replaces the old ``DesignLab``/``SimulationEngine`` helpers so
callers no longer need to reach into ``earlysign.stats.design.gst.common``.
It keeps a compatible surface (compute boundaries, run simulations, plot
summaries) while delegating to the canonical essentials-layer components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import earlysign.stats.essentials.methods.group_sequential.boundary as boundary_mod
from earlysign.integration.design.group_sequential.initial_design.schema import (
    DesignSpec,
    MeansDesignSpec,
    ProportionsDesignSpec,
    SpendingFunction,
    TimeToEventDesignSpec,
)
from earlysign.stats.essentials.methods.group_sequential import spending as spending_mod
from earlysign.stats.essentials.methods.group_sequential.boundary import (
    BoundaryCalculatorSpec,
    EfficacySpec,
    FutilitySpec,
)
from earlysign.stats.essentials.schemes.protocols import EffectSizeCalculator
from earlysign.stats.essentials.schemes.survival.effect_size import (
    TimeToEventEffectSizeCalculator,
)
from earlysign.stats.essentials.schemes.two_means.effect_size import (
    MeansEffectSizeCalculator,
)
from earlysign.stats.essentials.schemes.two_proportions.effect_size import (
    TwoProportionsEffectSizeCalculator,
)


def _resolve_effect_calculator(spec: DesignSpec) -> EffectSizeCalculator:
    if isinstance(spec, ProportionsDesignSpec):
        return TwoProportionsEffectSizeCalculator()
    if isinstance(spec, TimeToEventDesignSpec):
        return TimeToEventEffectSizeCalculator()
    if isinstance(spec, MeansDesignSpec):
        return MeansEffectSizeCalculator()
    raise ValueError(f"Unsupported design spec type: {type(spec)!r}")


def _build_spending(spec: DesignSpec) -> Tuple[spending_mod.SpendingFunction, int]:
    """Return (spending_function, tails) for the given spec."""

    sided = getattr(spec.test, "sided", "one")
    tails = 1 if sided == "one" else 2
    spending_kind = getattr(
        spec.boundary, "spending_function", SpendingFunction.OBRIEN_FLEMING
    )
    alpha = float(spec.test.alpha)

    spending: spending_mod.SpendingFunction
    if spending_kind == SpendingFunction.POCOCK:
        spending = spending_mod.PocockSpending(alpha=alpha)
    elif spending_kind == SpendingFunction.HSD:
        spending = spending_mod.HSDSpending(
            alpha=alpha,
            gamma=float(getattr(spec.boundary, "hsd_gamma", -4.0)),
        )
    else:
        spending = spending_mod.OBFSpending(alpha=alpha, sided=tails)
    return spending, tails


def _compute_boundaries_dict(spec: DesignSpec) -> Dict[str, Any]:
    """Compute z-boundaries and cumulative alpha for the given spec."""

    info_times = np.asarray(spec.resolved_info_times(), dtype=float)
    spending, tails = _build_spending(spec)
    cumulative_alpha = np.asarray(spending.cumulative(info_times), dtype=float)
    hsd_gamma = float(getattr(spec.boundary, "hsd_gamma", -4.0))

    eff = EfficacySpec(
        style="alpha_spending",
        family=str(getattr(spending, "name", "obrien_fleming")),
        gamma=hsd_gamma,
    )

    fut = FutilitySpec(mode="none")
    if getattr(spec.boundary, "futility_enabled", False):
        z_val = getattr(spec.boundary, "futility_z", None)
        if z_val is not None:
            fut = FutilitySpec(mode="fixed_threshold", z=float(z_val))
        elif getattr(spec.boundary, "futility_spending", None) is not None:
            fut = FutilitySpec(
                mode="beta_spending",
                family=str(
                    getattr(
                        spec.boundary.futility_spending,
                        "value",
                        "obrien_fleming",
                    )
                ),
                gamma=hsd_gamma,
                beta=float(1.0 - float(spec.test.power)),
            )
        else:
            fut = FutilitySpec(mode="symmetric")

    calc_spec = BoundaryCalculatorSpec(
        alpha=float(spec.test.alpha),
        tails=tails,
        scale="z",
        efficacy=eff,
        futility=fut,
        process=None,
    )

    calculator = boundary_mod.BoundaryCalculator(spec=calc_spec, process=None)
    boundaries = calculator.compute_boundaries(info_times=info_times)
    z_upper = np.asarray(boundaries["upper"], dtype=float)
    z_lower = (
        np.asarray(boundaries["lower"], dtype=float)
        if boundaries.get("lower") is not None
        else None
    )
    return {
        "info_times": info_times,
        "cumulative_alpha": cumulative_alpha,
        "z_efficacy": z_upper,
        "z_futility": z_lower,
    }


def _simulate_trial(
    spec: DesignSpec,
    boundaries: Dict[str, Any],
    effect_calc: EffectSizeCalculator,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Simulate a single trajectory for the specification."""

    info_times = boundaries["info_times"]
    z_upper = boundaries["z_efficacy"]
    z_lower = boundaries["z_futility"]
    k = len(info_times)
    allocation_ratio = spec.allocation.alloc_ratio
    schedule = spec.resolved_info_times()

    z_path = np.zeros(k)
    stopped = False
    stop_idx = k
    stop_reason = "final"

    for idx in range(k):
        if stopped:
            z_path[idx] = z_path[idx - 1]
            continue

        mean_at_t = effect_calc.standardized_effect(
            spec.effect,
            info_times[idx],
            sample_size=spec.sample_size,
            allocation_ratio=allocation_ratio,
            info_times=schedule,
        )

        if idx == 0:
            z_path[idx] = rng.normal(mean_at_t, np.sqrt(info_times[idx]))
        else:
            prev_mean = effect_calc.standardized_effect(
                spec.effect,
                info_times[idx - 1],
                sample_size=spec.sample_size,
                allocation_ratio=allocation_ratio,
                info_times=schedule,
            )
            drift = mean_at_t - prev_mean
            dt = info_times[idx] - info_times[idx - 1]
            z_path[idx] = rng.normal(z_path[idx - 1] + drift, np.sqrt(max(dt, 1e-9)))

        if z_path[idx] >= z_upper[idx]:
            stopped = True
            stop_idx = idx + 1
            stop_reason = "efficacy"
        elif z_lower is not None and z_path[idx] <= z_lower[idx]:
            stopped = True
            stop_idx = idx + 1
            stop_reason = "futility"

    return {
        "Z": z_path,
        "stopped_at": stop_idx,
        "reason": stop_reason,
        "reject_h0": stop_reason == "efficacy",
    }


def _aggregate_simulations(
    spec: DesignSpec,
    boundaries: Dict[str, Any],
    effect_calc: EffectSizeCalculator,
) -> Dict[str, Any]:
    rng = np.random.default_rng(spec.simulation.seed)
    results = [
        _simulate_trial(spec, boundaries, effect_calc, rng)
        for _ in range(spec.simulation.n_sims)
    ]

    rejections = sum(int(r["reject_h0"]) for r in results)
    power = rejections / spec.simulation.n_sims

    stop_dist: Dict[int, int] = {}
    for r in results:
        idx = int(r["stopped_at"])
        stop_dist[idx] = stop_dist.get(idx, 0) + 1

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
        n_per_analysis = np.zeros(len(boundaries["info_times"]))

    ess = sum(
        stop_dist.get(i + 1, 0) * float(n_per_analysis[i])
        for i in range(len(boundaries["info_times"]))
    ) / float(spec.simulation.n_sims)

    return {
        "power": power,
        "expected_sample_size": ess,
        "stop_distribution": stop_dist,
        "max_sample_size": int(float(n_per_analysis[-1])),
        "results": results,
    }


@dataclass
class DesignSpecAnalyzer:
    """Evaluate a ``DesignSpec`` by computing boundaries and simulations."""

    spec: DesignSpec

    def __post_init__(self) -> None:
        self.boundaries: Optional[Dict[str, Any]] = None
        self.simulation_results: Optional[Dict[str, Any]] = None
        self._effect_calculator = _resolve_effect_calculator(self.spec)

    def compute_boundaries(self) -> "DesignSpecAnalyzer":
        self.boundaries = _compute_boundaries_dict(self.spec)
        return self

    def run_simulations(self) -> "DesignSpecAnalyzer":
        if self.boundaries is None:
            self.compute_boundaries()
        assert self.boundaries is not None
        self.simulation_results = _aggregate_simulations(
            self.spec, self.boundaries, self._effect_calculator
        )
        return self

    def get_summary(self) -> pd.DataFrame:
        if self.boundaries is None:
            self.compute_boundaries()
        assert self.boundaries is not None
        info_times = self.boundaries["info_times"]
        z_upper = self.boundaries["z_efficacy"]
        z_lower = self.boundaries["z_futility"]
        alpha_cum = self.boundaries["cumulative_alpha"]

        sample_info = self._effect_calculator.sample_sizes(
            self.spec.sample_size,
            allocation_ratio=self.spec.allocation.alloc_ratio,
            info_times=self.spec.resolved_info_times(),
        )

        data: Dict[str, Any] = {
            "Analysis": np.arange(1, len(info_times) + 1, dtype=int),
            "Info Fraction": info_times,
            "Z Efficacy": z_upper,
            "Cumulative α": alpha_cum,
        }

        if "n_control" in sample_info:
            data["N Control"] = sample_info["n_control"]
            data["N Treatment"] = sample_info["n_treatment"]
            data["N Total"] = sample_info["n_total"]
        elif "events" in sample_info:
            data["Events"] = sample_info["events"]
            data["N Control"] = sample_info["n_control"]
            data["N Treatment"] = sample_info["n_treatment"]

        if z_lower is not None:
            data["Z Futility"] = z_lower

        df = pd.DataFrame(data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if "Info" in col or "α" in col:
                df[col] = df[col].round(self.spec.display.ddigits + 2)
            elif col.startswith("Z"):
                df[col] = df[col].round(self.spec.display.ddigits)
            else:
                df[col] = df[col].round(0).astype(int)
        return df

    def get_power_summary(self) -> Dict[str, Any]:
        if self.simulation_results is None:
            self.run_simulations()
        assert self.simulation_results is not None
        ess = self.simulation_results["expected_sample_size"]
        max_n = self.simulation_results["max_sample_size"]
        power = self.simulation_results["power"]
        return {
            "Estimated Power": f"{power:.1%}",
            "Expected Sample Size": f"{ess:.0f}",
            "Maximum Sample Size": f"{max_n:.0f}",
            "Efficiency (ESS/Max)": f"{ess / max_n:.1%}",
        }

    def plot_boundaries(
        self, show_trajectories: bool = False, n_trajectories: int = 20
    ) -> Any:
        import matplotlib.pyplot as plt

        if self.boundaries is None:
            self.compute_boundaries()
        assert self.boundaries is not None
        info_times = self.boundaries["info_times"]
        z_upper = self.boundaries["z_efficacy"]
        z_lower = self.boundaries["z_futility"]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(info_times, z_upper, "r-", linewidth=2, label="Efficacy", marker="o")
        if z_lower is not None:
            ax.plot(
                info_times, z_lower, "b-", linewidth=2, label="Futility", marker="s"
            )

        if show_trajectories and self.simulation_results is not None:
            for i in range(
                min(n_trajectories, len(self.simulation_results["results"]))
            ):
                ax.plot(info_times, self.simulation_results["results"][i]["Z"], "gray")

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Information Fraction", fontsize=12)
        ax.set_ylabel("Z-statistic", fontsize=12)
        ax.set_title("Sequential Boundaries", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "specification": self.spec.to_dict(),
        }
        if self.boundaries is not None:
            payload["boundaries"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.boundaries.items()
            }
        if self.simulation_results is not None:
            payload["simulation_results"] = dict(self.simulation_results)
        return payload
