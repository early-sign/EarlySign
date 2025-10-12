"""
Fixed Timing Design Mode UI Module.

This module provides an interactive Jupyter widget-based UI for the Fixed Timing
design mode in group sequential trials.
"""

from typing import Any, Protocol

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import HTML, clear_output, display

from earlysign.stats.design.gst.common.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.types import (
    DesignMode,
    InformationSpacing,
    SpendingFunction,
)


class ModeDesigner(Protocol):
    """Protocol for per-mode modular designers.

    Each designer owns its mode-specific widgets and a copy of common values
    so switching modes preserves per-mode configurations.
    """

    mode: DesignMode

    # ---- Widget lifecycle ----
    def build_panel(self, ui: Any) -> widgets.Widget:
        """Return a container widget that renders full mode UI (controls+actions).

        Widgets should be created in the designer's constructor so their
        values persist across mode switches. Action buttons must be owned by
        this designer and bind their handlers internally.
        """
        ...

    def get_description_widget(self) -> widgets.Widget:
        """Return a rich description widget to render under the mode selector."""
        ...

    # ---- State sync for common values ----
    def save_common_from(self, ui: Any) -> None:
        """Save current common widget values from the main UI into this designer."""
        ...

    def load_common_into(self, ui: Any) -> None:
        """Load this designer's saved common values into the main UI widgets."""
        ...

    # ---- Spec update hooks ----
    def update_spec(self, spec: DesignSpec, ui: Any) -> None:
        """Write both common and mode-specific values into the given DesignSpec."""
        ...

    # ---- Optional actions (owned internally by panel) ----
    def run_optimization(self, ui: Any) -> None:  # optional
        """Execute mode-specific optimization flow if the panel has such action."""
        ...


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

    # ---- ModeDesigner protocol ----
    def build_panel(self, ui: Any) -> widgets.Widget:
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
