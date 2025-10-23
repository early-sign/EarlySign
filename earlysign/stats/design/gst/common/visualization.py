"""Visualization tools for GST design evaluation results."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from earlysign.stats.design.gst.essentials.operating_characteristics import (
    OCCurveResult,
)
from earlysign.stats.design.gst.scenarios import (
    ScenarioAResult,
    ScenarioBResult,
)


class OCCurvePlotter:
    """Plotter for operating characteristics curves.

    Attributes:
        figsize: Figure size (width, height)
        colors: List of colors for different designs
        linestyles: List of line styles for different designs
    """

    def __init__(
        self,
        figsize: tuple[int, int] = (16, 6),
        colors: List[str] | None = None,
        linestyles: List[str] | None = None,
    ):
        self.figsize = figsize
        self.colors = colors or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        self.linestyles = linestyles or ["-", "--", "-.", ":"]

    def plot_scenario_a(
        self,
        result: ScenarioAResult,
        target_effect: float,
        null_value: float,
        effect_label: str = "Effect Size",
        save_path: str | None = None,
    ) -> None:
        """Plot operating characteristics for Scenario A (Fixed Max N).

        Args:
            result: ScenarioAResult with OC curves
            target_effect: Target effect size (for vertical line)
            null_value: Null hypothesis value (for x-axis offset)
            effect_label: Label for effect size axis
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0] // 2, self.figsize[1]))

        ax.set_title("Scenario A: Fixed Max N", fontsize=14, fontweight="bold")
        ax.set_xlabel(effect_label, fontsize=12)
        ax.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot fixed design point at target
        ax.scatter(
            [target_effect + null_value],
            [result.fixed_n],
            color=self.colors[0],
            s=150,
            marker="o",
            label=f"Fixed (0 interim), N={result.fixed_n}",
            zorder=5,
            edgecolors="black",
            linewidths=2,
        )

        # Plot GST curves
        for idx, (n_interim, oc_result) in enumerate(
            sorted(result.gst_results.items())
        ):
            # Find power at target (effect_sizes are differences, not absolute values)
            target_idx = np.argmin(np.abs(oc_result.effect_sizes - target_effect))
            power_at_target = oc_result.power_values[target_idx]

            # Plot ESS curve (convert effect sizes to absolute scale by adding null_value)
            ax.plot(
                oc_result.effect_sizes + null_value,
                oc_result.ess_values,
                color=self.colors[idx + 1],
                linestyle=self.linestyles[idx + 1],
                linewidth=2.5,
                label=f"{n_interim} interim ({oc_result.n_looks} looks), "
                f"Power@target: {power_at_target:.3f}",
                alpha=0.9,
            )

            # Overlay stopping distribution
            self._plot_stop_distribution(
                ax,
                oc_result,
                null_value,
                self.colors[idx + 1],
            )

        # Mark target effect
        ax.axvline(
            x=target_effect + null_value,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Target: {target_effect + null_value:.4f}",
        )

        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"   Plot saved to: {save_path}")

        plt.show()

    def plot_scenario_b(
        self,
        result: ScenarioBResult,
        target_effect: float,
        null_value: float,
        effect_label: str = "Effect Size",
        save_path: str | None = None,
    ) -> None:
        """Plot operating characteristics for Scenario B (Fixed Power).

        Args:
            result: ScenarioBResult with OC curves
            target_effect: Target effect size (for vertical line)
            null_value: Null hypothesis value (for x-axis offset)
            effect_label: Label for effect size axis
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0] // 2, self.figsize[1]))

        ax.set_title(
            "Scenario B: Fixed Power at Target δ", fontsize=14, fontweight="bold"
        )
        ax.set_xlabel(effect_label, fontsize=12)
        ax.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot fixed design as horizontal line
        ax.axhline(
            y=result.fixed_n,
            color=self.colors[0],
            linestyle=self.linestyles[0],
            linewidth=2,
            label=f"Fixed (0 interim), N={result.fixed_n}",
            alpha=0.6,
        )

        # Plot fixed design point at target
        ax.scatter(
            [target_effect + null_value],
            [result.fixed_n],
            color=self.colors[0],
            s=150,
            marker="o",
            zorder=5,
            edgecolors="black",
            linewidths=2,
        )

        # Plot GST curves
        for idx, (n_interim, oc_result) in enumerate(
            sorted(result.gst_results.items())
        ):
            # Plot ESS curve (convert effect sizes to absolute scale by adding null_value)
            ax.plot(
                oc_result.effect_sizes + null_value,
                oc_result.ess_values,
                color=self.colors[idx + 1],
                linestyle=self.linestyles[idx + 1],
                linewidth=2.5,
                label=f"{n_interim} interim ({oc_result.n_looks} looks)",
                alpha=0.9,
            )

            # Overlay stopping distribution
            self._plot_stop_distribution(
                ax,
                oc_result,
                null_value,
                self.colors[idx + 1],
            )

        # Mark target effect
        ax.axvline(
            x=target_effect + null_value,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Target: {target_effect + null_value:.4f}",
        )

        ax.legend(loc="best", fontsize=10)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"   Plot saved to: {save_path}")

        plt.show()

    def plot_comparison(
        self,
        result_a: ScenarioAResult,
        result_b: ScenarioBResult,
        target_effect: float,
        null_value: float,
        effect_label: str = "Effect Size",
        save_path: str | None = None,
    ) -> None:
        """Plot both scenarios side-by-side for comparison.

        Args:
            result_a: ScenarioAResult with OC curves
            result_b: ScenarioBResult with OC curves
            target_effect: Target effect size (for vertical line)
            null_value: Null hypothesis value (for x-axis offset)
            effect_label: Label for effect size axis
            save_path: Path to save figure (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Plot Scenario A
        ax1.set_title("Scenario A: Fixed Max N", fontsize=14, fontweight="bold")
        ax1.set_xlabel(effect_label, fontsize=12)
        ax1.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Fixed design point
        ax1.scatter(
            [target_effect + null_value],
            [result_a.fixed_n],
            color=self.colors[0],
            s=150,
            marker="o",
            label=f"Fixed (0 interim), N={result_a.fixed_n}",
            zorder=5,
            edgecolors="black",
            linewidths=2,
        )

        # GST curves for Scenario A
        for idx, (n_interim, oc_result) in enumerate(
            sorted(result_a.gst_results.items())
        ):
            # Find power at target (effect_sizes are differences, not absolute values)
            target_idx = np.argmin(np.abs(oc_result.effect_sizes - target_effect))
            power_at_target = oc_result.power_values[target_idx]

            # Convert effect sizes to absolute scale by adding null_value
            ax1.plot(
                oc_result.effect_sizes + null_value,
                oc_result.ess_values,
                color=self.colors[idx + 1],
                linestyle=self.linestyles[idx + 1],
                linewidth=2.5,
                label=f"{n_interim} interim, Power@target: {power_at_target:.3f}",
                alpha=0.9,
            )

            self._plot_stop_distribution(
                ax1, oc_result, null_value, self.colors[idx + 1]
            )

        ax1.axvline(
            x=target_effect + null_value,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Target: {target_effect + null_value:.4f}",
        )
        ax1.legend(loc="best", fontsize=10)

        # Plot Scenario B
        ax2.set_title(
            "Scenario B: Fixed Power at Target δ", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel(effect_label, fontsize=12)
        ax2.set_ylabel("Expected Sample Size (ESS)", fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Fixed design horizontal line
        ax2.axhline(
            y=result_b.fixed_n,
            color=self.colors[0],
            linestyle=self.linestyles[0],
            linewidth=2,
            label=f"Fixed (0 interim), N={result_b.fixed_n}",
            alpha=0.6,
        )

        # Fixed design point
        ax2.scatter(
            [target_effect + null_value],
            [result_b.fixed_n],
            color=self.colors[0],
            s=150,
            marker="o",
            zorder=5,
            edgecolors="black",
            linewidths=2,
        )

        # GST curves for Scenario B
        for idx, (n_interim, oc_result) in enumerate(
            sorted(result_b.gst_results.items())
        ):
            # Convert effect sizes to absolute scale by adding null_value
            ax2.plot(
                oc_result.effect_sizes + null_value,
                oc_result.ess_values,
                color=self.colors[idx + 1],
                linestyle=self.linestyles[idx + 1],
                linewidth=2.5,
                label=f"{n_interim} interim ({oc_result.n_looks} looks)",
                alpha=0.9,
            )

            self._plot_stop_distribution(
                ax2, oc_result, null_value, self.colors[idx + 1]
            )

        ax2.axvline(
            x=target_effect + null_value,
            color="gray",
            linestyle=":",
            alpha=0.5,
            label=f"Target: {target_effect + null_value:.4f}",
        )
        ax2.legend(loc="best", fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"   Plot saved to: {save_path}")

        plt.show()

    def _plot_stop_distribution(
        self,
        ax: Axes,
        oc_result: OCCurveResult,
        null_value: float,
        color: str,
    ) -> None:
        """Plot stopping distribution as scatter with transparency.

        Args:
            ax: Matplotlib axes
            oc_result: OC curve result
            null_value: Null hypothesis value (for x-axis offset)
            color: Color for scatter points
        """
        # Sample sizes at each analysis (total N)
        sample_sizes_at_analysis = np.array(
            [oc_result.n_per_analysis * (i + 1) * 2 for i in range(oc_result.n_looks)]
        )

        # For each effect size, plot points at each analysis
        for idx, result in enumerate(oc_result.results):
            # Convert effect size to absolute scale by adding null_value
            effect_val = result.effect_size + null_value
            stop_dist = result.stop_distribution

            if not stop_dist:
                continue

            # Total number of simulations
            total_sims = sum(stop_dist.values())

            # Plot a point for each analysis where stopping occurred
            for analysis_idx, count in stop_dist.items():
                n_total = sample_sizes_at_analysis[analysis_idx - 1]
                prob = count / total_sims

                # Use probability as alpha (transparency)
                alpha_val = prob * 0.5

                ax.scatter(
                    [effect_val],
                    [n_total],
                    color=color,
                    s=150,
                    alpha=alpha_val,
                    edgecolors="none",
                    zorder=3,
                )


def print_scenario_summary(
    result: ScenarioAResult | ScenarioBResult,
    target_effect: float,
    null_value: float,
    scenario_name: str = "A",
) -> None:
    """Print summary statistics for a scenario.

    Args:
        result: Scenario result (A or B)
        target_effect: Target effect size
        null_value: Null hypothesis value
        scenario_name: Name of scenario (for display)
    """
    print("\n" + "=" * 60)
    print(f"SCENARIO {scenario_name} SUMMARY")
    print("=" * 60)
    print(f"Fixed design total N: {result.fixed_n}")
    print()

    for n_interim, oc_result in sorted(result.gst_results.items()):
        # Find metrics at target effect (effect_sizes are differences, not absolute values)
        target_idx = np.argmin(np.abs(oc_result.effect_sizes - target_effect))
        ess_at_target = oc_result.ess_values[target_idx]
        power_at_target = oc_result.power_values[target_idx]
        max_n = oc_result.n_per_analysis * oc_result.n_looks * 2

        print(f"{n_interim} interim analysis ({oc_result.n_looks} looks):")
        print(f"  Max N:                {max_n}")
        print(f"  Power at target:      {power_at_target:.3f}")
        print(f"  ESS at target:        {ess_at_target:.0f}")
        print(
            f"  ESS savings:          {result.fixed_n - ess_at_target:.0f} "
            f"({(1 - ess_at_target/result.fixed_n)*100:.1f}%)"
        )
        print()

    print("=" * 60)
