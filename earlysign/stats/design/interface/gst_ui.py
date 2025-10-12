"""
Group Sequential Trial Design UI Module.

This module provides an interactive Jupyter widget-based UI for exploring
and optimizing group sequential trial designs.
"""

from typing import Any, Optional

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

        # Initialize widgets
        self._create_widgets()
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

        # ========== Common Parameters ==========
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

        # ========== Effect Parameters ==========
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

        # ========== Mode-Specific Parameters ==========

        # Fixed Timing mode
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

        # Optimize ASN mode
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

        # Optimize Design mode
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

        # Action Buttons
        self.w_compute_btn = widgets.Button(
            description="🔧 Compute Design",
            button_style="primary",
            tooltip="Compute boundaries for fixed timing",
        )

        self.w_optimize_btn = widgets.Button(
            description="🎯 Run Optimization",
            button_style="success",
            tooltip="Run design optimization",
        )

        self.w_simulate_btn = widgets.Button(
            description="🎲 Run Simulation",
            button_style="info",
            tooltip="Run power simulation",
        )

        # Output areas
        self.output_config = widgets.Output()
        self.output_summary = widgets.Output()
        self.output_plot = widgets.Output()
        self.output_power = widgets.Output()
        self.output_optimization = widgets.Output()

        # Event handlers
        self.w_mode.observe(self._on_mode_change, "value")
        self.w_compute_btn.on_click(self._on_compute_click)
        self.w_optimize_btn.on_click(self._on_optimize_click)
        self.w_simulate_btn.on_click(self._on_simulate_click)

    def _create_layout(self) -> None:
        """Create layout (top: config, bottom: results)."""

        # === Top: Design Configuration ===

        # Mode selection (always shown)
        mode_box = widgets.VBox(
            [widgets.HTML("<h3>📋 Design Mode</h3>"), self.w_mode, widgets.HTML("<hr>")]
        )

        # Common parameters
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

        # Mode-specific parameters (switch dynamically)
        self.mode_specific_box = widgets.VBox(
            [
                widgets.HTML("<h4>Mode-Specific Parameters</h4>"),
                self.w_info_times_text,
                self.w_n_per_analysis,
            ]
        )

        # Action buttons
        actions_box = widgets.HBox(
            [self.w_compute_btn, self.w_optimize_btn, self.w_simulate_btn],
            layout=widgets.Layout(justify_content="space-around"),
        )

        # Configuration panel (upper part)
        config_panel = widgets.VBox(
            [
                mode_box,
                widgets.HBox(
                    [common_box, self.mode_specific_box],
                    layout=widgets.Layout(justify_content="space-between"),
                ),
                widgets.HTML("<hr>"),
                actions_box,
            ],
            layout=widgets.Layout(
                padding="15px",
                border="2px solid #ddd",
                border_radius="10px",
                background_color="#f9f9f9",
            ),
        )

        # === Bottom: Design Results ===

        # Results tabs (singleton)
        if not hasattr(self, "results_tabs"):
            self.results_tabs = widgets.Tab()
            self.results_tabs.children = [
                self.output_summary,
                self.output_plot,
                self.output_power,
                self.output_optimization,
            ]
            self.results_tabs.set_title(0, "📊 Summary")
            self.results_tabs.set_title(1, "📈 Boundaries")
            self.results_tabs.set_title(2, "⚡ Power")
            self.results_tabs.set_title(3, "🎯 Optimization")
        else:
            # Always ensure only one set of children
            self.results_tabs.children = [
                self.output_summary,
                self.output_plot,
                self.output_power,
                self.output_optimization,
            ]

        results_panel = widgets.VBox(
            [widgets.HTML("<h2>📊 Design Results</h2>"), self.results_tabs],
            layout=widgets.Layout(
                padding="15px",
                border="2px solid #ddd",
                border_radius="10px",
                margin_top="20px",
            ),
        )

        # Main layout (vertical: config on top, results on bottom)
        self.main_layout = widgets.VBox(
            [
                widgets.HTML("<h1>🎛️ Design Configuration</h1>"),
                config_panel,
                results_panel,
            ]
        )

    def _attach_observers(self) -> None:
        """Set event handlers (lightweight)."""
        # Only mode switching triggers update; others are manual
        pass

    def _on_mode_change(self, change: dict[str, Any]) -> None:
        """Handler for mode switching."""
        mode = DesignMode(change["new"])
        self.current_mode = mode

        # Update mode-specific UI
        if mode == DesignMode.FIXED_TIMING:
            self.mode_specific_box.children = [
                widgets.HTML("<h4>Fixed Timing Parameters</h4>"),
                self.w_info_times_text,
                self.w_n_per_analysis,
                widgets.HTML(
                    "<p><i>Specify information times and compute boundaries</i></p>"
                ),
            ]
            self.w_compute_btn.disabled = False
            self.w_optimize_btn.disabled = True

        elif mode == DesignMode.OPTIMIZE_ASN:
            self.mode_specific_box.children = [
                widgets.HTML("<h4>ASN Optimization Parameters</h4>"),
                self.w_max_n_total,
                self.w_target_power,
                widgets.HTML(
                    "<p><i>Fix max N and optimize information times for min ASN</i></p>"
                ),
            ]
            self.w_compute_btn.disabled = True
            self.w_optimize_btn.disabled = False

        elif mode == DesignMode.OPTIMIZE_DESIGN:
            self.mode_specific_box.children = [
                widgets.HTML("<h4>Design Optimization Parameters</h4>"),
                widgets.HBox([self.w_alpha_range_min, self.w_alpha_range_max]),
                widgets.HBox([self.w_k_range_min, self.w_k_range_max]),
                self.w_optimize_criterion,
                widgets.HTML(
                    "<p><i>Find optimal combination of alpha, K, and timing</i></p>"
                ),
            ]
            self.w_compute_btn.disabled = True
            self.w_optimize_btn.disabled = False

        elif mode == DesignMode.FIXED_POWER:
            self.mode_specific_box.children = [
                widgets.HTML("<h4>Fixed Power Parameters</h4>"),
                self.w_target_power,
                widgets.HTML(
                    "<p><i>Fix power and compute required sample size</i></p>"
                ),
            ]
            self.w_compute_btn.disabled = False
            self.w_optimize_btn.disabled = True

    def _on_compute_click(self, button: Any) -> None:
        """Handler for Compute button click."""
        self._update_spec_from_widgets()

        with self.output_summary:
            clear_output(wait=True)
            display(HTML("<h3>⏳ Computing...</h3>"))

        try:
            self.lab = DesignLab(self.spec)
            self.lab.compute_boundaries()
            self._display_results()
        except Exception as e:
            with self.output_summary:
                clear_output(wait=True)
                display(HTML(f"<p style='color: red;'><b>Error:</b> {str(e)}</p>"))

    def _on_optimize_click(self, button: Any) -> None:
        """Handler for Optimize button click."""
        self._update_spec_from_widgets()

        with self.output_optimization:
            clear_output(wait=True)
            display(HTML("<h3>🎯 Running Optimization...</h3>"))

        try:
            if self.current_mode == DesignMode.OPTIMIZE_ASN:
                objective: DesignObjective = MinimizeASN(
                    max_n=self.w_max_n_total.value,
                    target_power=self.w_target_power.value,
                )
                self.optimizer = DesignOptimizer(self.spec, objective)

                # Optimize information times
                optimal_times = self.optimizer.optimize_info_times(
                    self.spec.sequential.n_analyses
                )

                # Update settings
                self.spec.sequential.info_times = optimal_times.tolist()
                self.spec.sequential.info_spacing = InformationSpacing.CUSTOM

                # Calculate results
                self.lab = DesignLab(self.spec)
                self.lab.compute_boundaries()
                self.lab.run_simulations()

                self._display_optimization_results(optimal_times)
                self._display_results()

            elif self.current_mode == DesignMode.OPTIMIZE_DESIGN:
                # Comprehensive optimization
                criterion = self.w_optimize_criterion.value

                objective_inst: DesignObjective
                if criterion == "minimize_asn":
                    objective_inst = MinimizeASN(
                        max_n=10000, target_power=self.w_power.value
                    )
                elif criterion == "maximize_power":
                    objective_inst = MaximizePower(max_n=10000)
                else:
                    objective_inst = BalancedDesign()

                self.optimizer = DesignOptimizer(self.spec, objective_inst)
                optimized_spec = self.optimizer.optimize_comprehensive()

                self.spec = optimized_spec
                self.lab = DesignLab(self.spec)
                self.lab.compute_boundaries()
                self.lab.run_simulations()

                if self.spec.sequential.info_times is not None:
                    self._display_optimization_results(
                        np.array(self.spec.sequential.info_times)
                    )
                self._display_results()

        except Exception as e:
            with self.output_optimization:
                clear_output(wait=True)
                display(
                    HTML(
                        f"<p style='color: red;'><b>Optimization Error:</b> {str(e)}</p>"
                    )
                )

    def _on_simulate_click(self, button: Any) -> None:
        """Handler for Simulate button click."""
        try:
            if self.lab.boundaries is None:
                self.lab.compute_boundaries()

            with self.output_power:
                clear_output(wait=True)
                display(HTML("<h3>🎲 Running Simulation...</h3>"))

            self.lab.run_simulations()
            self._display_power_results()

        except Exception as e:
            with self.output_power:
                clear_output(wait=True)
                display(
                    HTML(
                        f"<p style='color: red;'><b>Simulation Error:</b> {str(e)}</p>"
                    )
                )

    def _update_spec_from_widgets(self) -> None:
        """Update settings from widgets."""
        # Common parameters
        self.spec.test.alpha = self.w_alpha.value
        self.spec.test.power = self.w_power.value
        self.spec.sequential.n_analyses = self.w_n_analyses.value
        self.spec.boundary.spending_function = SpendingFunction(
            self.w_spending_func.value
        )

        # Effect parameters
        if isinstance(self.spec, ProportionsDesignSpec):
            self.spec.effect.p_control = self.w_p_control.value
            self.spec.effect.effect_size = self.w_effect_size.value

        # Mode-specific
        if self.current_mode == DesignMode.FIXED_TIMING:
            # Parse info times
            try:
                times_str = self.w_info_times_text.value
                times = [float(t.strip()) for t in times_str.split(",")]
                self.spec.sequential.info_times = times
                self.spec.sequential.info_spacing = InformationSpacing.CUSTOM
            except Exception:
                self.spec.sequential.info_spacing = InformationSpacing.EQUAL

            if isinstance(self.spec, ProportionsDesignSpec):
                self.spec.sample_size.n_per_analysis = self.w_n_per_analysis.value

    def _display_results(self) -> None:
        """Display results."""
        # Summary
        with self.output_summary:
            clear_output(wait=True)
            display(HTML("<h3>Design Summary</h3>"))
            display(self.lab.get_summary())

        # Plot
        with self.output_plot:
            clear_output(wait=True)
            fig = self.lab.plot_boundaries()
            # Display in notebook output widget, not as popup
            display(fig)
            plt.close(fig)

    def _display_power_results(self) -> None:
        """Display power analysis results."""
        with self.output_power:
            clear_output(wait=True)
            display(HTML("<h3>Power Analysis Results</h3>"))
            power_summary = self.lab.get_power_summary()

            html = "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
            for key, value in power_summary.items():
                html += f"<tr><td style='padding: 8px; border: 1px solid #ddd; background-color: #f0f0f0;'><b>{key}</b></td>"
                html += f"<td style='padding: 8px; border: 1px solid #ddd;'>{value}</td></tr>"
            html += "</table>"
            display(HTML(html))

    def _display_optimization_results(self, optimal_times: np.ndarray) -> None:
        """Display optimization results."""
        with self.output_optimization:
            clear_output(wait=True)
            display(HTML("<h3>Optimization Results</h3>"))

            display(
                HTML(
                    f"<p><b>Optimal Information Times:</b> {', '.join([f'{t:.3f}' for t in optimal_times])}</p>"
                )
            )

            if self.optimizer and self.optimizer.optimization_history:
                history_df = self.optimizer.get_optimization_summary()
                display(HTML("<h4>Optimization History (Last 10 iterations)</h4>"))
                display(history_df.tail(10))

    def _update_design(self) -> None:
        """Initial design calculation."""
        self._on_compute_click(None)

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
