"""Design Lab: Main orchestrator for sequential design planning."""

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from earlysign.stats.design.boundaries import BoundaryCalculator
from earlysign.stats.design.config import (
    DesignSpec,
    MeansDesignSpec,
    ProportionsDesignSpec,
    TimeToEventDesignSpec,
)
from earlysign.stats.design.effects import (
    EffectCalculator,
    MeansEffectCalculator,
    ProportionsEffectCalculator,
    TimeToEventEffectCalculator,
)
from earlysign.stats.design.simulation import SimulationEngine


class DesignLab:
    """Main orchestrator for sequential design planning.

    Provides a unified interface for:
    - Configuring sequential designs
    - Computing boundaries
    - Running simulations
    - Generating reports

    >>> from earlysign.stats.design.config import ProportionsDesignSpec
    >>> spec = ProportionsDesignSpec()
    >>> lab = DesignLab(spec)
    >>> _ = lab.compute_boundaries()
    >>> summary = lab.get_summary()
    >>> 'Analysis' in summary.columns
    True
    """

    def __init__(self, spec: DesignSpec):
        self.spec = spec
        self.boundaries: Optional[Dict[str, Any]] = None
        self.simulation_results: Optional[Dict[str, Any]] = None
        self._effect_calculator = self._create_effect_calculator()

    def _create_effect_calculator(self) -> EffectCalculator:
        """Factory method to create appropriate effect calculator."""
        if isinstance(self.spec, ProportionsDesignSpec):
            return ProportionsEffectCalculator()
        elif isinstance(self.spec, TimeToEventDesignSpec):
            return TimeToEventEffectCalculator()
        elif isinstance(self.spec, MeansDesignSpec):
            return MeansEffectCalculator()
        else:
            raise ValueError(f"Unsupported design spec type: {type(self.spec)}")

    def compute_boundaries(self) -> "DesignLab":
        """Compute critical boundaries."""
        self.boundaries = BoundaryCalculator.critical_values(self.spec)
        return self

    def run_simulations(self) -> "DesignLab":
        """Run simulation study."""
        if self.boundaries is None:
            self.compute_boundaries()

        assert self.boundaries is not None
        self.simulation_results = SimulationEngine.run_simulations(
            self.spec, self.boundaries, self._effect_calculator
        )
        return self

    def get_summary(self) -> pd.DataFrame:
        """Get summary table of design characteristics."""
        if self.boundaries is None:
            self.compute_boundaries()

        assert self.boundaries is not None
        t = self.boundaries["info_times"]
        z_upper = self.boundaries["z_efficacy"]
        z_lower = self.boundaries["z_futility"]
        alpha_cum = self.boundaries["cumulative_alpha"]

        # Sample size information
        sample_info = self._effect_calculator.sample_sizes(self.spec)

        data: Dict[str, Any] = {
            "Analysis": np.arange(1, len(t) + 1),
            "Info Fraction": t,
            "Z Efficacy": z_upper,
            "Cumulative α": alpha_cum,
        }

        # Add sample size columns
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

        # Format numeric columns
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
        """Get power and operating characteristics summary."""
        if self.simulation_results is None:
            self.run_simulations()

        assert self.simulation_results is not None
        return {
            "Estimated Power": f"{self.simulation_results['power']:.1%}",
            "Expected Sample Size": f"{self.simulation_results['expected_sample_size']:.0f}",
            "Maximum Sample Size": f"{self.simulation_results['max_sample_size']:.0f}",
            "Efficiency (ESS/Max)": f"{self.simulation_results['expected_sample_size'] / self.simulation_results['max_sample_size']:.1%}",
        }

    def plot_boundaries(
        self, show_trajectories: bool = False, n_trajectories: int = 20
    ) -> Any:
        """Plot critical boundaries and optionally sample trajectories.

        Args:
            show_trajectories: Whether to show sample Z-statistic trajectories
            n_trajectories: Number of trajectories to plot

        Returns:
            Matplotlib figure object
        """
        import matplotlib.pyplot as plt

        if self.boundaries is None:
            self.compute_boundaries()

        assert self.boundaries is not None
        t = self.boundaries["info_times"]
        z_upper = self.boundaries["z_efficacy"]
        z_lower = self.boundaries["z_futility"]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Efficacy boundary
        ax.plot(t, z_upper, "r-", linewidth=2, label="Efficacy Boundary", marker="o")

        # Futility boundary
        if z_lower is not None:
            ax.plot(
                t, z_lower, "b-", linewidth=2, label="Futility Boundary", marker="s"
            )

        # Sample trajectories
        if show_trajectories and self.simulation_results is not None:
            for i in range(
                min(n_trajectories, len(self.simulation_results["results"]))
            ):
                Z = self.simulation_results["results"][i]["Z"]
                ax.plot(t, Z, "gray", alpha=0.3, linewidth=0.5)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Information Fraction", fontsize=12)
        ax.set_ylabel("Z-statistic", fontsize=12)
        ax.set_title(
            f'{self.spec.boundary.spending_function.value.replace("_", " ").title()} Boundaries',
            fontsize=14,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def to_dict(self) -> Dict[str, Any]:
        """Export full design information as dictionary."""
        result: Dict[str, Any] = {
            "specification": self.spec.to_dict(),
        }

        if self.boundaries is not None:
            result["boundaries"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.boundaries.items()
            }

        if self.simulation_results is not None:
            sim_copy = self.simulation_results.copy()
            sim_copy.pop("results", None)  # Don't include all individual results
            result["simulation_summary"] = sim_copy

        return result

    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export design to JSON file or return JSON string.

        Args:
            filepath: Optional path to save JSON file

        Returns:
            JSON string or confirmation message
        """
        json_str = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
            return f"Design exported to {filepath}"
        return json_str

    def __repr__(self) -> str:
        lines = [
            f"DesignLab({self.spec.test.test_type.value})",
            f"  Analyses: {self.spec.sequential.n_analyses}",
            f"  Alpha: {self.spec.test.alpha}",
            f"  Spending: {self.spec.boundary.spending_function.value}",
        ]

        if self.simulation_results:
            lines.append(f"  Power: {self.simulation_results['power']:.1%}")

        return "\n".join(lines)
