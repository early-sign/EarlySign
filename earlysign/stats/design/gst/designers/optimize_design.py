"""
Optimize Design (Multi-objective) Mode UI Module.

This module provides an interactive Jupyter widget-based UI for the Optimize Design
mode in group sequential trials, which explores optimal combinations of design parameters.
"""

from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, clear_output, display

from earlysign.stats.design.gst.common.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.optimization import (
    BalancedDesign,
    DesignObjective,
    DesignOptimizer,
    MaximizePower,
    MinimizeASN,
)
from earlysign.stats.design.gst.common.types import DesignMode, SpendingFunction


class OptimizeDesignDesigner:
    """Mode designer: Flexible Parameters → Optimize Design."""

    mode = DesignMode.OPTIMIZE_DESIGN

    def __init__(self) -> None:
        # Common parameter widgets (owned by this mode)
        self.w_alpha = widgets.FloatText(
            value=0.025,
            min=0.001,
            max=0.5,
            step=0.001,
            description="Alpha (α):",
            style={"description_width": "120px"},
        )
        self.w_power = widgets.FloatText(
            value=0.90,
            min=0.50,
            max=0.99,
            step=0.01,
            description="Power (1-β):",
            style={"description_width": "120px"},
        )
        self.w_n_analyses = widgets.IntText(
            value=3,
            min=2,
            max=20,
            step=1,
            description="# Analyses:",
            style={"description_width": "120px"},
        )
        self.w_spending_func = widgets.Dropdown(
            options=[
                ("O'Brien-Fleming", "obrien_fleming"),
                ("Pocock", "pocock"),
                ("HSD", "hsd"),
            ],
            value="obrien_fleming",
            description="Spending Fn:",
            style={"description_width": "120px"},
        )
        self.w_p_control = widgets.FloatText(
            value=0.10,
            min=0.001,
            max=0.999,
            step=0.001,
            description="p (Control):",
            style={"description_width": "120px"},
        )
        self.w_effect_size = widgets.FloatText(
            value=0.02,
            min=0.001,
            max=1.0,
            step=0.001,
            description="Effect Size:",
            style={"description_width": "120px"},
        )
        self.w_alpha_range_min = widgets.FloatText(
            value=0.01,
            min=0.001,
            max=0.5,
            step=0.001,
            description="Alpha Min:",
            style={"description_width": "120px"},
        )
        self.w_alpha_range_max = widgets.FloatText(
            value=0.05,
            min=0.001,
            max=0.5,
            step=0.001,
            description="Alpha Max:",
            style={"description_width": "120px"},
        )
        self.w_k_range_min = widgets.IntText(
            value=2,
            min=1,
            max=20,
            step=1,
            description="K Min:",
            style={"description_width": "120px"},
        )
        self.w_k_range_max = widgets.IntText(
            value=6,
            min=1,
            max=20,
            step=1,
            description="K Max:",
            style={"description_width": "120px"},
        )
        self.w_optimize_criterion = widgets.Dropdown(
            options=[
                ("Minimize ASN", "minimize_asn"),
                ("Maximize Power", "maximize_power"),
                ("Balanced", "balanced"),
            ],
            value="minimize_asn",
            description="Criterion:",
            style={"description_width": "120px"},
        )

        # Single output widget for all results (no tabs or buttons)
        self.out_results = widgets.Output()

        # Flag to prevent recursive updates
        self._updating = False

    def build_panel(self, ui: Any) -> widgets.Widget:
        # Auto-update handler for parameter changes
        def on_parameter_change(_change: Any) -> None:
            if not self._updating:
                self._auto_update_results(ui)

        # Attach observers to all parameter widgets
        self.w_alpha.observe(on_parameter_change, names="value")
        self.w_power.observe(on_parameter_change, names="value")
        self.w_n_analyses.observe(on_parameter_change, names="value")
        self.w_spending_func.observe(on_parameter_change, names="value")
        self.w_p_control.observe(on_parameter_change, names="value")
        self.w_effect_size.observe(on_parameter_change, names="value")
        self.w_alpha_range_min.observe(on_parameter_change, names="value")
        self.w_alpha_range_max.observe(on_parameter_change, names="value")
        self.w_k_range_min.observe(on_parameter_change, names="value")
        self.w_k_range_max.observe(on_parameter_change, names="value")
        self.w_optimize_criterion.observe(on_parameter_change, names="value")

        common_box = widgets.VBox(
            [
                widgets.HTML("<h4>Common Parameters</h4>"),
                self.w_alpha,
                self.w_power,
                self.w_n_analyses,
                self.w_spending_func,
                self.w_p_control,
                self.w_effect_size,
            ]
        )

        opt_box = widgets.VBox(
            [
                widgets.HTML("<h4>Design Optimization Parameters</h4>"),
                widgets.HBox([self.w_alpha_range_min, self.w_alpha_range_max]),
                widgets.HBox([self.w_k_range_min, self.w_k_range_max]),
                self.w_optimize_criterion,
                widgets.HTML("<h4>Results</h4>"),
                self.out_results,
            ]
        )

        panel = widgets.VBox([common_box, widgets.HTML("<hr>"), opt_box])

        # Trigger initial optimization
        self._auto_update_results(ui)

        return panel

    def _auto_update_results(self, ui: Any) -> None:
        """Automatically update results when parameters change."""
        if self._updating:
            return

        self._updating = True
        try:
            with self.out_results:
                clear_output(wait=True)
                display(
                    HTML("<h3>🎯 Running design optimization and simulation...</h3>")
                )

            # Run optimization
            self.update_spec(ui.spec, ui)
            self.run_optimization(ui)

        except Exception as e:
            with self.out_results:
                clear_output(wait=True)
                display(HTML(f"<p style='color: red;'><b>Error:</b> {str(e)}</p>"))
        finally:
            self._updating = False

    def get_description_widget(self) -> widgets.Widget:
        return widgets.HTML(
            """
            <div>
              <b>Flexible Parameters → Optimize Design</b><br>
              Explore optimal combinations of α, number of analyses (K), and information spacing
              based on a chosen criterion (minimize ASN, maximize power, or a balanced objective).
            </div>
            """
        )

    def save_common_from(self, ui: Any) -> None:
        # No-op; common parameters are owned by this designer's widgets
        return None

    def load_common_into(self, ui: Any) -> None:
        # No-op; there are no global common widgets to sync into
        return None

    def update_spec(self, spec: DesignSpec, ui: Any) -> None:
        # Use designer-owned common values
        spec.test.alpha = self.w_alpha.value
        spec.test.power = self.w_power.value
        spec.sequential.n_analyses = self.w_n_analyses.value
        spec.boundary.spending_function = SpendingFunction(self.w_spending_func.value)

        if isinstance(spec, ProportionsDesignSpec):
            spec.effect.p_control = self.w_p_control.value
            spec.effect.effect_size = self.w_effect_size.value

        # Clear info_times so the optimizer can choose
        spec.sequential.info_times = None

    def run_optimization(self, ui: Any) -> None:
        # Choose objective
        criterion = self.w_optimize_criterion.value
        if criterion == "minimize_asn":
            objective_inst: DesignObjective = MinimizeASN(
                planned_max_n=10000, target_power=self.w_power.value
            )
        elif criterion == "maximize_power":
            objective_inst = MaximizePower(planned_max_n=10000)
        else:
            objective_inst = BalancedDesign()

        ui.optimizer = DesignOptimizer(ui.spec, objective_inst)
        optimized_spec = ui.optimizer.optimize_comprehensive()

        ui.spec = optimized_spec
        ui.lab = DesignLab(ui.spec)
        ui.lab.compute_boundaries()
        ui.lab.run_simulations()

        if ui.spec.sequential.info_times is not None:
            self._display_optimization_results(np.array(ui.spec.sequential.info_times))
        self._display_results(ui)

    def _display_results(self, ui: Any) -> None:
        """Display optimization results, design summary, boundaries plot, and power results inline."""
        with self.out_results:
            clear_output(wait=True)

            # Optimization Results
            display(HTML("<h3>🎯 Optimization Results</h3>"))
            if ui.spec.sequential.info_times is not None:
                optimal_times = np.array(ui.spec.sequential.info_times)
                display(
                    HTML(
                        f"<p><b>Optimal Information Times:</b> {', '.join([f'{t:.3f}' for t in optimal_times])}</p>"
                    )
                )

            # Design Summary
            display(HTML("<h3 style='margin-top: 20px;'>📊 Design Summary</h3>"))
            display(ui.lab.get_summary())

            # Boundaries Plot
            display(HTML("<h3 style='margin-top: 20px;'>📈 Boundaries</h3>"))
            fig = ui.lab.plot_boundaries()
            display(fig)
            plt.close(fig)

            # Power Analysis Results
            display(HTML("<h3 style='margin-top: 20px;'>⚡ Power Analysis</h3>"))
            power_summary = ui.lab.get_power_summary()
            html = "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
            for key, value in power_summary.items():
                html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; background-color: #f0f0f0;'><b>{key}</b></td>"
                html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{value}</td></tr>"
            html += "</table>"
            display(HTML(html))

    def _display_optimization_results(self, optimal_times: np.ndarray) -> None:
        """Display optimization results - now integrated into _display_results."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize widget values to dictionary."""
        return {
            "alpha": self.w_alpha.value,
            "power": self.w_power.value,
            "n_analyses": self.w_n_analyses.value,
            "spending_func": self.w_spending_func.value,
            "p_control": self.w_p_control.value,
            "effect_size": self.w_effect_size.value,
            "alpha_range_min": self.w_alpha_range_min.value,
            "alpha_range_max": self.w_alpha_range_max.value,
            "k_range_min": self.w_k_range_min.value,
            "k_range_max": self.w_k_range_max.value,
            "optimize_criterion": self.w_optimize_criterion.value,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore widget values from dictionary."""
        if "alpha" in data:
            self.w_alpha.value = data["alpha"]
        if "power" in data:
            self.w_power.value = data["power"]
        if "n_analyses" in data:
            self.w_n_analyses.value = data["n_analyses"]
        if "spending_func" in data:
            self.w_spending_func.value = data["spending_func"]
        if "p_control" in data:
            self.w_p_control.value = data["p_control"]
        if "effect_size" in data:
            self.w_effect_size.value = data["effect_size"]
        if "alpha_range_min" in data:
            self.w_alpha_range_min.value = data["alpha_range_min"]
        if "alpha_range_max" in data:
            self.w_alpha_range_max.value = data["alpha_range_max"]
        if "k_range_min" in data:
            self.w_k_range_min.value = data["k_range_min"]
        if "k_range_max" in data:
            self.w_k_range_max.value = data["k_range_max"]
        if "optimize_criterion" in data:
            self.w_optimize_criterion.value = data["optimize_criterion"]
