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

        # Own action buttons
        self.btn_optimize = widgets.Button(
            description="🎯 Run Optimization",
            button_style="success",
            tooltip="Run design optimization",
        )
        self.btn_simulate = widgets.Button(
            description="🎲 Run Simulation",
            button_style="info",
            tooltip="Run power simulation",
        )

        # Mode-owned result areas and tabs
        self.out_summary = widgets.Output()
        self.out_plot = widgets.Output()
        self.out_power = widgets.Output()
        self.out_opt = widgets.Output()
        self.tabs = widgets.Tab()
        self.tabs.children = [
            self.out_summary,
            self.out_plot,
            self.out_power,
            self.out_opt,
        ]
        self.tabs.set_title(0, "📊 Summary")
        self.tabs.set_title(1, "📈 Boundaries")
        self.tabs.set_title(2, "⚡ Power")
        self.tabs.set_title(3, "🎯 Optimization")

    def build_panel(self, ui: Any) -> widgets.Widget:
        def on_optimize(_btn: Any) -> None:
            self.update_spec(ui.spec, ui)
            with self.out_opt:
                clear_output(wait=True)
                display(HTML("<h3>🎯 Running Optimization...</h3>"))
            try:
                self.run_optimization(ui)
            except Exception as e:
                with self.out_opt:
                    clear_output(wait=True)
                    display(
                        HTML(
                            f"<p style='color: red;'><b>Optimization Error:</b> {str(e)}</p>"
                        )
                    )

        def on_simulate(_btn: Any) -> None:
            try:
                if getattr(ui.lab, "boundaries", None) is None:
                    ui.lab.compute_boundaries()
                with self.out_power:
                    clear_output(wait=True)
                    display(HTML("<h3>🎲 Running Simulation...</h3>"))
                ui.lab.run_simulations()
                self._display_power_results(ui)
            except Exception as e:
                with self.out_power:
                    clear_output(wait=True)
                    display(
                        HTML(
                            f"<p style='color: red;'><b>Simulation Error:</b> {str(e)}</p>"
                        )
                    )

        self.btn_optimize.on_click(on_optimize)
        self.btn_simulate.on_click(on_simulate)

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
                widgets.HBox(
                    [self.btn_optimize, self.btn_simulate],
                    layout=widgets.Layout(justify_content="flex-start"),
                ),
                widgets.HTML("<h4>Results</h4>"),
                self.tabs,
            ]
        )

        return widgets.VBox([common_box, widgets.HTML("<hr>"), opt_box])

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
        with self.out_summary:
            clear_output(wait=True)
            display(HTML("<h3>Design Summary</h3>"))
            display(ui.lab.get_summary())
        with self.out_plot:
            clear_output(wait=True)
            fig = ui.lab.plot_boundaries()
            display(fig)
            plt.close(fig)

    def _display_power_results(self, ui: Any) -> None:
        with self.out_power:
            clear_output(wait=True)
            display(HTML("<h3>Power Analysis Results</h3>"))
            power_summary = ui.lab.get_power_summary()
            html = "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
            for key, value in power_summary.items():
                html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; background-color: #f0f0f0;'><b>{key}</b></td>"
                html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{value}</td></tr>"
            html += "</table>"
            display(HTML(html))

    def _display_optimization_results(self, optimal_times: np.ndarray) -> None:
        with self.out_opt:
            clear_output(wait=True)
            display(HTML("<h3>Optimization Results</h3>"))
            display(
                HTML(
                    f"<p><b>Optimal Information Times:</b> {', '.join([f'{t:.3f}' for t in optimal_times])}</p>"
                )
            )
