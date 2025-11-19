"""
Fixed Timing Design Mode UI Module.

This module provides an interactive Jupyter widget-based UI for the Fixed Timing
design mode in group sequential trials.
"""

from typing import Any

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display

from earlysign.integration.design.group_sequential.initial_design.schema import (
    DesignMode,
    DesignSpec,
    InformationSpacing,
    ProportionsDesignSpec,
    SpendingFunction,
)
from earlysign.integration.design.group_sequential.initial_design.workflows.spec_analysis import (
    DesignSpecAnalyzer,
)


class FixedTimingDesigner:
    """Mode designer: Fixed Timing → Compute Boundaries."""

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

        # Single output widget for all results (no tabs)
        self.out_results = widgets.Output()

        # Flag to prevent recursive updates
        self._updating = False

    # ---- ModeDesigner protocol ----
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
        self.w_info_times_text.observe(on_parameter_change, names="value")
        self.w_n_per_analysis.observe(on_parameter_change, names="value")

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
                widgets.HTML("<h4>Results</h4>"),
                self.out_results,
            ]
        )

        panel = widgets.VBox([common_box, widgets.HTML("<hr>"), fixed_box])

        # Trigger initial computation
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
                display(HTML("<h3>⏳ Computing design and running simulation...</h3>"))

            # Update spec and compute boundaries
            self.update_spec(ui.spec, ui)
            ui.lab = DesignSpecAnalyzer(ui.spec)
            ui.lab.compute_boundaries()

            # Run simulation
            ui.lab.run_simulations()

            # Display results
            self._display_results(ui)

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
              <b>Fixed Timing → Compute Boundaries</b><br>
              Provide the information times (e.g., 0.33, 0.67, 1.0) and per-analysis sample size.
              This mode computes group sequential boundaries for the given α, power and spending function.
            </div>
            """
        )

    def _display_results(self, ui: Any) -> None:
        """Display design summary, boundaries plot, and power results inline."""
        with self.out_results:
            clear_output(wait=True)

            # Design Summary
            display(HTML("<h3>📊 Design Summary</h3>"))
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

    def save_common_from(self, ui: Any) -> None:
        # No-op; common parameters are owned by this designer's widgets
        return None

    def load_common_into(self, ui: Any) -> None:
        # No-op; there are no global common widgets to sync into
        return None

    def update_spec(self, spec: DesignSpec, ui: Any) -> None:
        # Common parameters from this designer's widgets
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

    def run_optimization(self, ui: Any) -> None:
        # Not applicable for this mode
        return

    def to_dict(self) -> dict[str, Any]:
        """Serialize widget values to dictionary."""
        return {
            "alpha": self.w_alpha.value,
            "power": self.w_power.value,
            "n_analyses": self.w_n_analyses.value,
            "spending_func": self.w_spending_func.value,
            "p_control": self.w_p_control.value,
            "effect_size": self.w_effect_size.value,
            "info_times_text": self.w_info_times_text.value,
            "n_per_analysis": self.w_n_per_analysis.value,
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
        if "info_times_text" in data:
            self.w_info_times_text.value = data["info_times_text"]
        if "n_per_analysis" in data:
            self.w_n_per_analysis.value = data["n_per_analysis"]
