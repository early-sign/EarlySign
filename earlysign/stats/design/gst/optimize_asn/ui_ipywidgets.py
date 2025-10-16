"""
Optimize ASN Design Mode UI Module.

This module provides an interactive Jupyter widget-based UI for the Optimize ASN
(Average Sample Number) design mode in group sequential trials.
"""

from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, clear_output, display

from earlysign.stats.design.gst.common.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.optimization import (
    DesignObjective,
    DesignOptimizer,
    MinimizeASN,
)
from earlysign.stats.design.gst.common.types import (
    DesignMode,
    InformationSpacing,
    SpendingFunction,
)


class OptimizeASNDesigner:
    """Mode designer: Fixed Max N → Optimize ASN."""

    mode = DesignMode.OPTIMIZE_ASN

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
        self.w_max_n_total = widgets.IntText(
            value=3000,
            min=100,
            max=20000,
            description="Max N Total:",
            style={"description_width": "120px"},
        )
        self.w_target_power = widgets.FloatText(
            value=0.90,
            min=0.50,
            max=0.99,
            step=0.01,
            description="Target Power:",
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
        self.w_n_analyses.observe(on_parameter_change, names="value")
        self.w_spending_func.observe(on_parameter_change, names="value")
        self.w_p_control.observe(on_parameter_change, names="value")
        self.w_effect_size.observe(on_parameter_change, names="value")
        self.w_max_n_total.observe(on_parameter_change, names="value")
        self.w_target_power.observe(on_parameter_change, names="value")

        common_box = widgets.VBox(
            [
                widgets.HTML("<h4>Common Parameters</h4>"),
                self.w_alpha,
                self.w_n_analyses,
                self.w_spending_func,
                self.w_p_control,
                self.w_effect_size,
            ]
        )

        opt_box = widgets.VBox(
            [
                widgets.HTML("<h4>ASN Optimization Parameters</h4>"),
                self.w_max_n_total,
                self.w_target_power,
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
                display(HTML("<h3>🎯 Running optimization and simulation...</h3>"))

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
              <b>Fixed Max N → Optimize ASN</b><br>
              Specify a maximum total sample size and a target power, then optimize
              the information times to minimize the expected sample number (ASN).
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
        # For this mode, the power target is the mode-specific target
        spec.test.power = self.w_target_power.value
        spec.sequential.n_analyses = self.w_n_analyses.value
        spec.boundary.spending_function = SpendingFunction(self.w_spending_func.value)

        if isinstance(spec, ProportionsDesignSpec):
            spec.effect.p_control = self.w_p_control.value
            spec.effect.effect_size = self.w_effect_size.value

        # info_times will be set by the optimizer; clear to CUSTOM later
        spec.sequential.info_times = None

    def run_optimization(self, ui: Any) -> None:
        # Set up optimizer and run information time optimization
        objective: DesignObjective = MinimizeASN(
            planned_max_n=self.w_max_n_total.value,
            target_power=self.w_target_power.value,
        )
        ui.optimizer = DesignOptimizer(ui.spec, objective)

        optimal_times = ui.optimizer.optimize_info_times(ui.spec.sequential.n_analyses)

        # Update spec and recompute
        ui.spec.sequential.info_times = optimal_times.tolist()
        ui.spec.sequential.info_spacing = InformationSpacing.CUSTOM

        ui.lab = DesignLab(ui.spec)
        ui.lab.compute_boundaries()
        ui.lab.run_simulations()

        self._display_optimization_results(optimal_times)
        self._display_results(ui)

    def _display_results(self, ui: Any) -> None:
        """Display optimization results, design summary, boundaries plot, and power results inline."""
        with self.out_results:
            clear_output(wait=True)

            # Optimization Results
            display(HTML("<h3>🎯 Optimization Results</h3>"))
            optimal_times = ui.spec.sequential.info_times
            if optimal_times:
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
            "n_analyses": self.w_n_analyses.value,
            "spending_func": self.w_spending_func.value,
            "p_control": self.w_p_control.value,
            "effect_size": self.w_effect_size.value,
            "max_n_total": self.w_max_n_total.value,
            "target_power": self.w_target_power.value,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore widget values from dictionary."""
        if "alpha" in data:
            self.w_alpha.value = data["alpha"]
        if "n_analyses" in data:
            self.w_n_analyses.value = data["n_analyses"]
        if "spending_func" in data:
            self.w_spending_func.value = data["spending_func"]
        if "p_control" in data:
            self.w_p_control.value = data["p_control"]
        if "effect_size" in data:
            self.w_effect_size.value = data["effect_size"]
        if "max_n_total" in data:
            self.w_max_n_total.value = data["max_n_total"]
        if "target_power" in data:
            self.w_target_power.value = data["target_power"]
