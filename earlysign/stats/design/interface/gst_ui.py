"""
Group Sequential Trial Design UI Module.

This module provides an interactive Jupyter widget-based UI for exploring
and optimizing group sequential trial designs.
"""

from typing import Any, Optional, Protocol

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, clear_output, display

from earlysign.stats.design.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.design.lab import DesignLab
from earlysign.stats.design.optimization import (
    BalancedDesign,
    DesignObjective,
    DesignOptimizer,
    MaximizePower,
    MinimizeASN,
)
from earlysign.stats.design.types import (
    DesignMode,
    InformationSpacing,
    SpendingFunction,
)


class ModeController(Protocol):
    """Protocol for per-mode modular controllers.

    Each controller owns its mode-specific widgets and a copy of common values
    so switching modes preserves per-mode configurations.
    """

    mode: DesignMode

    # ---- Widget lifecycle ----
    def build_panel(self, ui: "GSTDesignUI") -> widgets.Widget:
        """Return a container widget that renders full mode UI (controls+actions).

        Widgets should be created in the controller's constructor so their
        values persist across mode switches. Action buttons must be owned by
        this controller and bind their handlers internally.
        """

        ...

    def get_description_widget(self) -> widgets.Widget:
        """Return a rich description widget to render under the mode selector."""

        ...

    # ---- State sync for common values ----
    def save_common_from(self, ui: "GSTDesignUI") -> None:
        """Save current common widget values from the main UI into this controller."""

        ...

    def load_common_into(self, ui: "GSTDesignUI") -> None:
        """Load this controller's saved common values into the main UI widgets."""

        ...

    # ---- Spec update hooks ----
    def update_spec(self, spec: DesignSpec, ui: "GSTDesignUI") -> None:
        """Write both common and mode-specific values into the given DesignSpec."""

        ...

    # ---- Optional actions (owned internally by panel) ----
    def run_optimization(self, ui: "GSTDesignUI") -> None:  # optional
        """Execute mode-specific optimization flow if the panel has such action."""

        ...


class FixedTimingController:
    """Mode controller: Fixed Timing → Compute Boundaries."""

    mode = DesignMode.FIXED_TIMING

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
        # Mode-specific widgets
        self.w_info_times_text = widgets.Textarea(
            value="0.33, 0.67, 1.0",
            description="Info Times:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="400px", height="60px"),
        )
        self.w_n_per_analysis = widgets.IntText(
            value=500,
            min=10,
            max=10000,
            description="N per Analysis:",
            style={"description_width": "120px"},
        )

        # Own action buttons
        self.btn_compute = widgets.Button(
            description="🔧 Compute Design",
            button_style="primary",
            tooltip="Compute boundaries",
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

    # ---- ModeController protocol ----
    def build_panel(self, ui: "GSTDesignUI") -> widgets.Widget:
        # Bind handlers once per panel build
        def on_compute(_btn: Any) -> None:
            self.update_spec(ui.spec, ui)
            with self.out_summary:
                clear_output(wait=True)
                display(HTML("<h3>⏳ Computing...</h3>"))
            ui.lab = DesignLab(ui.spec)
            ui.lab.compute_boundaries()
            self._display_results(ui)

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

        self.btn_compute.on_click(on_compute)
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

        fixed_box = widgets.VBox(
            [
                widgets.HTML("<h4>Fixed Timing Parameters</h4>"),
                self.w_info_times_text,
                self.w_n_per_analysis,
                widgets.HBox(
                    [self.btn_compute, self.btn_simulate],
                    layout=widgets.Layout(justify_content="flex-start"),
                ),
                widgets.HTML("<h4>Results</h4>"),
                self.tabs,
            ]
        )

        return widgets.VBox([common_box, widgets.HTML("<hr>"), fixed_box])

    def get_description_widget(self) -> widgets.Widget:
        return widgets.HTML(
            """
            <div>
              <b>Fixed Timing → Compute Boundaries</b><br>
              Provide the information times (e.g., 0.33, 0.67, 1.0) and per-analysis sample size.
              This mode computes group sequential boundaries for the given α, power and spending function.
            </div>
            """
        )

    def _display_results(self, ui: "GSTDesignUI") -> None:
        with self.out_summary:
            clear_output(wait=True)
            display(HTML("<h3>Design Summary</h3>"))
            display(ui.lab.get_summary())
        with self.out_plot:
            clear_output(wait=True)
            fig = ui.lab.plot_boundaries()
            display(fig)
            plt.close(fig)

    def _display_power_results(self, ui: "GSTDesignUI") -> None:
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

    def save_common_from(self, ui: "GSTDesignUI") -> None:
        # No-op; common parameters are owned by this controller's widgets
        return None

    def load_common_into(self, ui: "GSTDesignUI") -> None:
        # No-op; there are no global common widgets to sync into
        return None

    def update_spec(self, spec: DesignSpec, ui: "GSTDesignUI") -> None:
        # Common parameters from this controller's widgets
        spec.test.alpha = self.w_alpha.value
        spec.test.power = self.w_power.value
        spec.sequential.n_analyses = self.w_n_analyses.value
        spec.boundary.spending_function = SpendingFunction(self.w_spending_func.value)

        if isinstance(spec, ProportionsDesignSpec):
            spec.effect.p_control = self.w_p_control.value
            spec.effect.effect_size = self.w_effect_size.value

        # Mode-specific
        try:
            times_str = self.w_info_times_text.value
            times = [float(t.strip()) for t in times_str.split(",")]
            spec.sequential.info_times = times
            spec.sequential.info_spacing = InformationSpacing.CUSTOM
        except Exception:
            spec.sequential.info_spacing = InformationSpacing.EQUAL

        if isinstance(spec, ProportionsDesignSpec):
            spec.sample_size.n_per_analysis = self.w_n_per_analysis.value

    def run_optimization(self, ui: "GSTDesignUI") -> None:
        # Not applicable for this mode
        return


class OptimizeASNController:
    """Mode controller: Fixed Max N → Optimize ASN."""

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

        # Own action buttons
        self.btn_optimize = widgets.Button(
            description="🎯 Run Optimization",
            button_style="success",
            tooltip="Run ASN optimization",
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

    def build_panel(self, ui: "GSTDesignUI") -> widgets.Widget:
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
              <b>Fixed Max N → Optimize ASN</b><br>
              Specify a maximum total sample size and a target power, then optimize
              the information times to minimize the expected sample number (ASN).
            </div>
            """
        )

    def save_common_from(self, ui: "GSTDesignUI") -> None:
        # No-op; common parameters are owned by this controller's widgets
        return None

    def load_common_into(self, ui: "GSTDesignUI") -> None:
        # No-op; there are no global common widgets to sync into
        return None

    def update_spec(self, spec: DesignSpec, ui: "GSTDesignUI") -> None:
        # Use controller-owned common values
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

    def run_optimization(self, ui: "GSTDesignUI") -> None:
        # Set up optimizer and run information time optimization
        objective: DesignObjective = MinimizeASN(
            max_n=self.w_max_n_total.value, target_power=self.w_target_power.value
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

    def _display_results(self, ui: "GSTDesignUI") -> None:
        with self.out_summary:
            clear_output(wait=True)
            display(HTML("<h3>Design Summary</h3>"))
            display(ui.lab.get_summary())
        with self.out_plot:
            clear_output(wait=True)
            fig = ui.lab.plot_boundaries()
            display(fig)
            plt.close(fig)

    def _display_power_results(self, ui: "GSTDesignUI") -> None:
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


class OptimizeDesignController:
    """Mode controller: Flexible Parameters → Optimize Design."""

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

    def build_panel(self, ui: "GSTDesignUI") -> widgets.Widget:
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

    def save_common_from(self, ui: "GSTDesignUI") -> None:
        # No-op; common parameters are owned by this controller's widgets
        return None

    def load_common_into(self, ui: "GSTDesignUI") -> None:
        # No-op; there are no global common widgets to sync into
        return None

    def update_spec(self, spec: DesignSpec, ui: "GSTDesignUI") -> None:
        # Use controller-owned common values
        spec.test.alpha = self.w_alpha.value
        spec.test.power = self.w_power.value
        spec.sequential.n_analyses = self.w_n_analyses.value
        spec.boundary.spending_function = SpendingFunction(self.w_spending_func.value)

        if isinstance(spec, ProportionsDesignSpec):
            spec.effect.p_control = self.w_p_control.value
            spec.effect.effect_size = self.w_effect_size.value

        # Clear info_times so the optimizer can choose
        spec.sequential.info_times = None

    def run_optimization(self, ui: "GSTDesignUI") -> None:
        # Choose objective
        criterion = self.w_optimize_criterion.value
        if criterion == "minimize_asn":
            objective_inst: DesignObjective = MinimizeASN(
                max_n=10000, target_power=self.w_power.value
            )
        elif criterion == "maximize_power":
            objective_inst = MaximizePower(max_n=10000)
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

    def _display_results(self, ui: "GSTDesignUI") -> None:
        with self.out_summary:
            clear_output(wait=True)
            display(HTML("<h3>Design Summary</h3>"))
            display(ui.lab.get_summary())
        with self.out_plot:
            clear_output(wait=True)
            fig = ui.lab.plot_boundaries()
            display(fig)
            plt.close(fig)

    def _display_power_results(self, ui: "GSTDesignUI") -> None:
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


class FixedPowerController:
    """Mode controller: Fixed Power → Compute Required N."""

    mode = DesignMode.FIXED_POWER

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
        self.w_target_power = widgets.FloatText(
            value=0.90,
            min=0.50,
            max=0.99,
            step=0.01,
            description="Target Power:",
            style={"description_width": "120px"},
        )

        # Own action buttons
        self.btn_compute = widgets.Button(
            description="🔧 Compute Design",
            button_style="primary",
            tooltip="Compute required N",
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

    def build_panel(self, ui: "GSTDesignUI") -> widgets.Widget:
        def on_compute(_btn: Any) -> None:
            self.update_spec(ui.spec, ui)
            with self.out_summary:
                clear_output(wait=True)
                display(HTML("<h3>⏳ Computing...</h3>"))
            ui.lab = DesignLab(ui.spec)
            ui.lab.compute_boundaries()
            self._display_results(ui)

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

        self.btn_compute.on_click(on_compute)
        self.btn_simulate.on_click(on_simulate)

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

        fixed_box = widgets.VBox(
            [
                widgets.HTML("<h4>Fixed Power Parameters</h4>"),
                self.w_target_power,
                widgets.HBox(
                    [self.btn_compute, self.btn_simulate],
                    layout=widgets.Layout(justify_content="flex-start"),
                ),
                widgets.HTML("<h4>Results</h4>"),
                self.tabs,
            ]
        )

        return widgets.VBox([common_box, widgets.HTML("<hr>"), fixed_box])

    def get_description_widget(self) -> widgets.Widget:
        return widgets.HTML(
            """
            <div>
              <b>Fixed Power → Compute Required N</b><br>
              Set a target power and compute the sample size required under the chosen spending function
              and effect assumptions.
            </div>
            """
        )

    def save_common_from(self, ui: "GSTDesignUI") -> None:
        # No-op; common parameters are owned by this controller's widgets
        return None

    def load_common_into(self, ui: "GSTDesignUI") -> None:
        # No-op; there are no global common widgets to sync into
        return None

    def update_spec(self, spec: DesignSpec, ui: "GSTDesignUI") -> None:
        # Use target power for spec
        spec.test.alpha = self.w_alpha.value
        spec.test.power = self.w_target_power.value
        spec.sequential.n_analyses = self.w_n_analyses.value
        spec.boundary.spending_function = SpendingFunction(self.w_spending_func.value)

        if isinstance(spec, ProportionsDesignSpec):
            spec.effect.p_control = self.w_p_control.value
            spec.effect.effect_size = self.w_effect_size.value

        # Leave info_times as-is; compute_boundaries will derive N
        # based on fixed power settings in the lab.

    def run_optimization(self, ui: "GSTDesignUI") -> None:
        # Not applicable for this mode
        return

    # Results helpers for FixedPower
    def _display_results(self, ui: "GSTDesignUI") -> None:
        with self.out_summary:
            clear_output(wait=True)
            display(HTML("<h3>Design Summary</h3>"))
            display(ui.lab.get_summary())
        with self.out_plot:
            clear_output(wait=True)
            fig = ui.lab.plot_boundaries()
            display(fig)
            plt.close(fig)

    def _display_power_results(self, ui: "GSTDesignUI") -> None:
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


class GSTDesignUI:
    """
    Modular, mode-based design planner UI for Group Sequential Trials.

    Supports multiple use case patterns:
    - Fixed Timing: timing is fixed, compute boundaries
    - Optimize ASN: max N is fixed, optimize ASN
    - Optimize Design: search for optimal combination of α, power, effect size, etc.
    - Fixed Power: power is fixed, compute required sample size

    Examples
    --------
    >>> ui = GSTDesignUI()  # doctest: +SKIP
    >>> ui.display()  # doctest: +SKIP
    """

    _singleton_instance: Optional["GSTDesignUI"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "GSTDesignUI":
        """Ensure only one instance exists in the notebook."""
        if cls._singleton_instance is not None:
            return cls._singleton_instance
        instance = super().__new__(cls)
        cls._singleton_instance = instance
        return instance

    def __init__(self, initial_spec: Optional[DesignSpec] = None):
        """
        Initialize the UI.

        Parameters
        ----------
        initial_spec : DesignSpec, optional
            Initial design specification. Defaults to ProportionsDesignSpec.
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized: bool = True

        self.spec = initial_spec or ProportionsDesignSpec()
        self.lab = DesignLab(self.spec)
        self.optimizer: Optional[DesignOptimizer] = None
        self.current_mode = DesignMode.FIXED_TIMING

        # Initialize widgets and controllers
        self._create_widgets()
        self._init_mode_controllers()
        self._create_layout()
        self._attach_observers()

        # Initial calculation
        self._update_design()

    def _create_widgets(self) -> None:
        """Create widgets."""

        # ========== Mode Selection ==========
        self.w_mode = widgets.Dropdown(
            options=[
                ("Fixed Timing → Compute Boundaries", DesignMode.FIXED_TIMING.value),
                ("Fixed Max N → Optimize ASN", DesignMode.OPTIMIZE_ASN.value),
                (
                    "Flexible Parameters → Optimize Design",
                    DesignMode.OPTIMIZE_DESIGN.value,
                ),
                ("Fixed Power → Compute Required N", DesignMode.FIXED_POWER.value),
            ],
            value=DesignMode.FIXED_TIMING.value,
            description="Design Mode:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="600px"),
        )

        # ========== Mode-Specific Parameters moved to controllers ==========

        # Event handlers
        self.w_mode.observe(self._on_mode_change, "value")
        # Action buttons are owned by controllers; no global bindings

    def _init_mode_controllers(self) -> None:
        """Create per-mode controllers and initialize their saved common state."""
        self._controllers: dict[DesignMode, ModeController] = {
            DesignMode.FIXED_TIMING: FixedTimingController(),
            DesignMode.OPTIMIZE_ASN: OptimizeASNController(),
            DesignMode.OPTIMIZE_DESIGN: OptimizeDesignController(),
            DesignMode.FIXED_POWER: FixedPowerController(),
        }

        # Nothing to sync; common parameters are now controller-owned
        pass

    def _create_layout(self) -> None:
        """Create layout (top: config, bottom: results)."""

        # === Top: Design Configuration ===

        # Mode selection (always shown)
        self.description_box = widgets.VBox([])
        self.mode_specific_box = widgets.VBox([])

        controller = self._controllers[self.current_mode]
        self._render_mode_specific(controller)

        config_panel = widgets.VBox(
            [
                widgets.HTML("<h3>📋 Design Mode</h3>"),
                self.w_mode,
                self.description_box,
                widgets.HTML("<hr>"),
                self.mode_specific_box,
            ],
            layout=widgets.Layout(
                padding="15px",
                border="2px solid #ddd",
                border_radius="10px",
                background_color="#f9f9f9",
            ),
        )

        # Main layout: selector + description + per-mode panel
        self.main_layout = widgets.VBox(
            [widgets.HTML("<h1>🎛️ Design Configuration</h1>"), config_panel]
        )

    def _attach_observers(self) -> None:
        """Set event handlers (lightweight)."""
        # Only mode switching triggers update; others are manual
        pass

    def _on_mode_change(self, change: dict[str, Any]) -> None:
        """Handler for mode switching."""
        mode = DesignMode(change["new"])
        # Switch mode
        self.current_mode = mode
        controller = self._controllers[self.current_mode]
        self._render_mode_specific(controller)

    def _render_mode_specific(self, controller: ModeController) -> None:
        """Render mode-specific controls and set action enable flags."""
        # Update description below the selector
        self.description_box.children = [controller.get_description_widget()]
        # Build the panel (contains controller-owned common params + actions + results)
        panel = controller.build_panel(self)
        self.mode_specific_box.children = [panel]

    def _update_spec_from_widgets(self) -> None:
        """Update settings from widgets via the current mode controller."""
        controller = self._controllers[self.current_mode]
        controller.update_spec(self.spec, self)

    def _update_design(self) -> None:
        """Initial design calculation."""
        # Initialize using current mode's controller by computing boundaries once
        controller = self._controllers[self.current_mode]
        controller.update_spec(self.spec, self)
        try:
            self.lab = DesignLab(self.spec)
            self.lab.compute_boundaries()
            # Initial display is handled by per-mode actions; we avoid global outputs.
        except Exception:
            # On init, do not write to outputs; just raise for visibility in notebook logs
            raise

    def display(self) -> None:
        """
        Display UI, ensuring only one instance is shown and previous widgets are closed.

        Examples
        --------
        >>> ui = GSTDesignUI()  # doctest: +SKIP
        >>> ui.display()  # doctest: +SKIP
        """
        if hasattr(self, "_last_displayed_widget"):
            last_widget: Any = self._last_displayed_widget
            if last_widget is not self.main_layout:
                try:
                    last_widget.close()
                except Exception:
                    pass
        self._last_displayed_widget: Any = self.main_layout
        clear_output(wait=True)
        display(self.main_layout)
