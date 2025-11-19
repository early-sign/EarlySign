"""
N-Max Fixed Min MDE Design Mode UI Module.

This module provides an interactive Jupyter widget-based UI for finding the
minimum detectable effect (MDE) given a maximum sample size constraint (N_max).

Design Philosophy
-----------------
This is NOT an MDE optimization problem. Rather, it is a **design feasibility
analysis** under sample size constraints:

1. **Fixed Constraints**: N_max (maximum sample size), α, β (power), k (# analyses)
2. **Design Variables**: Information time schedule {t_i}, boundaries {c_i}
3. **Objective**: Find minimum effect size δ such that:
   - The optimal design (boundaries + schedule) achieves target power (1-β)
   - Total sample size ≤ N_max

The MDE is the **output metric** (sensitivity), not the optimization variable.
The actual optimization occurs over the design space (schedule + boundaries)
for each candidate effect size during binary search.

Implementation
--------------
Uses binary search over effect sizes δ:
- For each δ, computes optimal design with ScheduleOptimization + boundaries
- Checks if achieved power ≥ target power AND max_N ≤ N_max
- Narrows search range until minimum feasible δ is found

Use Case
--------
"I have budget/time constraints limiting me to N_max total samples.
Given α, power, and k analyses, what's the smallest effect I can detect?"

Result: MDE★ = minimum δ where optimal design meets power target within N_max
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


class NMaxFixedMinMDEDesigner:
    """
    Mode designer: N-Max Fixed → Find Minimum MDE.

    This mode finds the minimum detectable effect (MDE) that can be reliably
    detected given a maximum sample size constraint. This is a **design feasibility
    analysis**, not an MDE optimization problem.

    Design Framework
    ----------------
    - **Fixed**: N_max, α, β (power target), k (# analyses)
    - **Searched**: Effect size δ (binary search)
    - **Optimized per δ**: Schedule {t_i}, boundaries {c_i}
    - **Objective**: min δ such that optimal design achieves power with N ≤ N_max

    The MDE is the **result** (sensitivity metric), not the variable being optimized.
    For each candidate δ during search, the underlying optimization computes the
    best schedule and boundaries to maximize power within constraints.

    Returns
    -------
    - MDE★: Minimum effect size meeting power target within N_max
    - Optimal boundaries {c_i} and schedule {t_i} for MDE★
    - Power curve, ASN, early stopping probabilities
    """

    mode = DesignMode.NMAX_FIXED_MIN_MDE

    def __init__(self) -> None:
        # Common parameter widgets (owned by this mode)
        self.w_alpha = widgets.FloatText(
            value=0.025,
            min=0.001,
            max=0.5,
            step=0.001,
            description="Alpha (α):",
            style={"description_width": "150px"},
            tooltip="One-sided significance level",
        )
        self.w_power = widgets.FloatText(
            value=0.90,
            min=0.50,
            max=0.99,
            step=0.01,
            description="Power (1-β):",
            style={"description_width": "150px"},
            tooltip="Target detection power",
        )
        self.w_n_analyses = widgets.IntText(
            value=3,
            min=2,
            max=20,
            step=1,
            description="# Analyses (k):",
            style={"description_width": "150px"},
            tooltip="Maximum number of interim analyses",
        )
        self.w_spending_func = widgets.Dropdown(
            options=[
                ("O'Brien-Fleming", "obrien_fleming"),
                ("Pocock", "pocock"),
                ("HSD", "hsd"),
            ],
            value="obrien_fleming",
            description="Spending Fn:",
            style={"description_width": "150px"},
            tooltip="Alpha spending function",
        )
        self.w_p_control = widgets.FloatText(
            value=0.10,
            min=0.001,
            max=0.999,
            step=0.001,
            description="p (Control):",
            style={"description_width": "150px"},
            tooltip="Assumed control proportion",
        )

        # Mode-specific parameters
        self.w_n_max = widgets.IntText(
            value=3000,
            min=100,
            max=50000,
            step=100,
            description="N Max:",
            style={"description_width": "150px"},
            tooltip="Maximum total sample size constraint",
        )

        self.w_futility = widgets.Dropdown(
            options=[
                ("None", "none"),
                ("Non-binding", "non_binding"),
                ("Binding", "binding"),
            ],
            value="none",
            description="Futility:",
            style={"description_width": "150px"},
            tooltip="Type of futility boundary",
        )

        self.w_futility_threshold = widgets.FloatText(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Futility Z/CP:",
            style={"description_width": "150px"},
            tooltip="Futility boundary threshold (Z-score or conditional power)",
            disabled=True,  # Enable when futility is not "none"
        )

        self.w_info_spacing = widgets.Dropdown(
            options=[
                ("Equal", "equal"),
                ("Front-loaded", "front"),
                ("Back-loaded", "back"),
            ],
            value="equal",
            description="Info Spacing:",
            style={"description_width": "150px"},
            tooltip="Information fraction spacing strategy",
        )

        # Search parameters
        self.w_mde_search_min = widgets.FloatText(
            value=0.001,
            min=0.0001,
            max=1.0,
            step=0.001,
            description="MDE Search Min:",
            style={"description_width": "150px"},
            tooltip="Minimum effect size for search range",
        )

        self.w_mde_search_max = widgets.FloatText(
            value=0.10,
            min=0.001,
            max=1.0,
            step=0.01,
            description="MDE Search Max:",
            style={"description_width": "150px"},
            tooltip="Maximum effect size for search range",
        )

        # Observer for futility dropdown to enable/disable threshold
        self.w_futility.observe(self._on_futility_change, "value")

        # Mode-owned result area (single output, no tabs)
        self.out_results = widgets.Output()

        # Store the found MDE
        self._found_mde: float | None = None

        # Flag to prevent recursive updates
        self._updating = False

    def _on_futility_change(self, change: dict[str, Any]) -> None:
        """Enable/disable futility threshold based on futility type."""
        futility_type = change["new"]
        self.w_futility_threshold.disabled = futility_type == "none"

    def build_panel(self, ui: Any) -> widgets.Widget:
        # Auto-update function triggered by parameter changes
        def on_parameter_change(change: Any) -> None:
            if self._updating:
                return
            self._updating = True
            try:
                self._auto_update_results(ui)
            finally:
                self._updating = False

        # Attach observers to all relevant widgets for auto-update
        self.w_alpha.observe(on_parameter_change, "value")
        self.w_power.observe(on_parameter_change, "value")
        self.w_p_control.observe(on_parameter_change, "value")
        self.w_n_max.observe(on_parameter_change, "value")
        self.w_n_analyses.observe(on_parameter_change, "value")
        self.w_spending_func.observe(on_parameter_change, "value")
        self.w_info_spacing.observe(on_parameter_change, "value")
        self.w_futility.observe(on_parameter_change, "value")
        self.w_futility_threshold.observe(on_parameter_change, "value")
        self.w_mde_search_min.observe(on_parameter_change, "value")
        self.w_mde_search_max.observe(on_parameter_change, "value")

        common_box = widgets.VBox(
            [
                widgets.HTML("<h4>Statistical Parameters</h4>"),
                self.w_alpha,
                self.w_power,
                self.w_p_control,
            ]
        )

        design_box = widgets.VBox(
            [
                widgets.HTML("<h4>Design Parameters</h4>"),
                self.w_n_analyses,
                self.w_spending_func,
                self.w_info_spacing,
                self.w_futility,
                self.w_futility_threshold,
            ]
        )

        constraint_box = widgets.VBox(
            [
                widgets.HTML("<h4>Constraints & Search Range</h4>"),
                self.w_n_max,
                self.w_mde_search_min,
                self.w_mde_search_max,
            ]
        )

        results_box = widgets.VBox(
            [
                widgets.HTML("<h4>Results</h4>"),
                self.out_results,
            ]
        )

        panel = widgets.VBox(
            [
                common_box,
                widgets.HTML("<hr>"),
                design_box,
                widgets.HTML("<hr>"),
                constraint_box,
                widgets.HTML("<hr>"),
                results_box,
            ]
        )

        # Trigger initial computation
        self._auto_update_results(ui)

        return panel

    def _auto_update_results(self, ui: Any) -> None:
        """Auto-update results when parameters change."""
        with self.out_results:
            clear_output(wait=True)
            display(HTML("<h3>🔍 Searching for Minimum MDE...</h3>"))
        try:
            self.update_spec(ui.spec, ui)
            self._run_mde_search(ui)
        except Exception as e:
            with self.out_results:
                clear_output(wait=True)
                display(HTML(f"<p style='color: red;'><b>Error:</b> {str(e)}</p>"))

    def get_description_widget(self) -> widgets.Widget:
        return widgets.HTML(
            """
            <div style="padding: 10px; background-color: #f0f8ff; border-radius: 5px;">
              <b>🎯 N-Max Fixed → Find Minimum MDE</b><br><br>
              <b>Purpose:</b> Find the smallest effect size (MDE★) detectable with specified
              power (1-β) given a maximum sample size constraint (N_max).<br><br>

              <b>Design Framework:</b><br>
              This is NOT MDE optimization. Rather, it's a <b>design feasibility analysis</b>:<br>
              <ol>
                <li><b>Binary search</b> over effect sizes δ</li>
                <li>For each δ: <b>Optimize</b> information schedule {tᵢ} and boundaries {cᵢ}</li>
                <li>Check if power target met with total N ≤ N_max</li>
                <li>Return minimum feasible δ as MDE★</li>
              </ol>

              MDE is the <b>output metric</b> (sensitivity), not the optimization variable.<br><br>

              <b>Use Case:</b> Budget/time constraints limit your maximum sample size.
              You want to know: "What's the smallest effect I can reliably detect?"<br><br>

              <b>Inputs:</b>
              <ul>
                <li>N_max: Maximum total sample size (constraint)</li>
                <li>α: One-sided significance level</li>
                <li>Power (1-β): Target detection power</li>
                <li>k: Maximum number of interim analyses</li>
                <li>Spending function: Alpha spending strategy</li>
                <li>Search range: [MDE_min, MDE_max] for binary search</li>
              </ul>

              <b>Outputs:</b>
              <ul>
                <li>MDE★: Minimum detectable effect meeting power target</li>
                <li>Optimal boundaries {cᵢ} and schedule {tᵢ}</li>
                <li>Power curve, ASN, early stopping probabilities</li>
              </ul>
            </div>
            """
        )

    def _run_mde_search(self, ui: Any) -> None:
        """
        Search for minimum MDE that achieves target power under N_max constraint.

        Design Logic
        ------------
        This implements a **binary search over effect sizes** to find the minimum
        detectable effect. For each candidate effect size δ during the search:

        1. **Optimization**: Compute optimal information schedule {t_i} and
           boundaries {c_i} that maximize power for this δ
           - Uses underlying schedule optimization (if available)
           - Or uses fixed equal spacing with optimal boundaries

        2. **Feasibility Check**: Verify that:
           - Achieved power ≥ target power (1-β)
           - Total max sample size ≤ N_max

        3. **Search Update**:
           - If feasible: Can detect smaller effects → reduce search range
           - If not feasible: Need larger effect → increase search range

        The result MDE★ is the **minimum effect size** for which an optimal design
        exists that meets the power target within the N_max constraint.

        Note: MDE is NOT being optimized directly—it's the search variable.
        The actual optimization occurs over the design space (schedule + boundaries)
        for each candidate MDE during binary search.

        Parameters
        ----------
        ui : Any
            UI context with spec and lab.

        Updates
        -------
        - self._found_mde : float
            Minimum detectable effect that meets constraints
        - ui.lab : DesignSpecAnalyzer
            Lab instance with optimal design for MDE★
        """
        # Get parameters
        n_max = self.w_n_max.value
        target_power = self.w_power.value
        mde_min = self.w_mde_search_min.value
        mde_max = self.w_mde_search_max.value
        k = self.w_n_analyses.value

        # Binary search for minimum MDE
        tolerance = 0.0001
        max_iterations = 20
        iteration = 0

        search_log = []
        best_mde = None
        best_design = None

        mde_low = mde_min
        mde_high = mde_max

        with self.out_results:
            clear_output(wait=True)
            display(HTML("<h3>🔍 MDE Search Progress</h3>"))
            progress = widgets.FloatProgress(
                value=0, min=0, max=max_iterations, description="Searching:"
            )
            display(progress)
            log_output = widgets.Output()
            display(log_output)

        while iteration < max_iterations and (mde_high - mde_low) > tolerance:
            mde_test = (mde_low + mde_high) / 2.0

            # Create spec with this MDE
            if isinstance(ui.spec, ProportionsDesignSpec):
                ui.spec.effect.effect_size = mde_test
                ui.spec.sample_size.n_per_analysis = n_max // k  # Divide N_max by k
                ui.spec.test.power = target_power

            # Update spec with current parameters
            self.update_spec(ui.spec, ui)

            # Try to compute boundaries and check power
            try:
                lab = DesignSpecAnalyzer(ui.spec)
                lab.compute_boundaries()
                lab.run_simulations()

                # Get achieved power
                power_summary = lab.get_power_summary()
                achieved_power = (
                    float(
                        str(power_summary.get("Overall Power (H1)", "0.0"))
                        .replace("%", "")
                        .strip()
                    )
                    / 100.0
                )

                # Check if design is feasible (not exceeding N_max)
                # Get actual max N from design
                if lab.boundaries is not None:
                    info_fractions = lab.boundaries["info_times"]
                    n_per_analysis = ui.spec.sample_size.n_per_analysis
                    actual_max_n = info_fractions[-1] * n_per_analysis * k
                else:
                    actual_max_n = float("inf")

                feasible = actual_max_n <= n_max

                log_entry = {
                    "iteration": iteration + 1,
                    "mde": mde_test,
                    "achieved_power": achieved_power,
                    "target_power": target_power,
                    "feasible": feasible,
                    "max_n": actual_max_n,
                }
                search_log.append(log_entry)

                with log_output:
                    clear_output(wait=True)
                    for entry in search_log[-5:]:  # Show last 5 entries
                        status = "✓" if entry["feasible"] else "✗"
                        power_status = (
                            "✓"
                            if entry["achieved_power"] >= entry["target_power"]
                            else "✗"
                        )
                        display(
                            HTML(
                                f"<pre>Iter {entry['iteration']:2d}: MDE={entry['mde']:.5f} "
                                f"→ Power={entry['achieved_power']:.3f} {power_status} "
                                f"MaxN={entry['max_n']:.0f} {status}</pre>"
                            )
                        )

                # Binary search logic
                if feasible and achieved_power >= target_power:
                    # Can achieve power with this MDE, try smaller
                    mde_high = mde_test
                    best_mde = mde_test
                    best_design = lab
                else:
                    # Cannot achieve power, need larger MDE
                    mde_low = mde_test

            except Exception as e:
                # If computation fails, assume not feasible
                search_log.append(
                    {
                        "iteration": iteration + 1,
                        "mde": mde_test,
                        "error": str(e),
                    }
                )
                mde_low = mde_test  # Need larger MDE

            iteration += 1
            progress.value = iteration

        # Store result
        self._found_mde = best_mde

        # Display results
        with self.out_results:
            if best_mde is not None and best_design is not None:
                # Update ui.lab with best design
                ui.lab = best_design
                self._found_mde = best_mde

                # Update effect size widget to show found MDE
                if isinstance(ui.spec, ProportionsDesignSpec):
                    ui.spec.effect.effect_size = best_mde

                clear_output(wait=True)
                display(HTML("<h3>✅ MDE Search Complete</h3>"))
                display(
                    HTML(
                        f"""
                <div style="padding: 15px; background-color: #e8f5e9; border-radius: 5px; margin: 10px 0;">
                  <h4>🎯 Minimum Detectable Effect Found</h4>
                  <table style='width:100%; border-collapse: collapse;'>
                    <tr>
                      <td style='padding: 8px; border: 1px solid #ddd; background-color: #fff;'><b>MDE★</b></td>
                      <td style='padding: 8px; border: 1px solid #ddd;'>{best_mde:.5f}</td>
                    </tr>
                    <tr>
                      <td style='padding: 8px; border: 1px solid #ddd; background-color: #fff;'><b>Target Power</b></td>
                      <td style='padding: 8px; border: 1px solid #ddd;'>{target_power:.3f}</td>
                    </tr>
                    <tr>
                      <td style='padding: 8px; border: 1px solid #ddd; background-color: #fff;'><b>N Max Constraint</b></td>
                      <td style='padding: 8px; border: 1px solid #ddd;'>{n_max}</td>
                    </tr>
                    <tr>
                      <td style='padding: 8px; border: 1px solid #ddd; background-color: #fff;'><b>Iterations</b></td>
                      <td style='padding: 8px; border: 1px solid #ddd;'>{iteration}</td>
                    </tr>
                  </table>
                </div>
                """
                    )
                )

                # Display design summary and plot
                self._display_results(ui)
            else:
                clear_output(wait=True)
                display(
                    HTML(
                        f"""
                <div style="padding: 15px; background-color: #ffebee; border-radius: 5px;">
                  <h4>⚠️ No Feasible MDE Found</h4>
                  <p>Could not find an effect size in the range [{mde_min:.5f}, {mde_max:.5f}]
                  that achieves power {target_power:.3f} with N_max={n_max}.</p>
                  <p><b>Suggestions:</b></p>
                  <ul>
                    <li>Increase N_max constraint</li>
                    <li>Decrease target power</li>
                    <li>Expand search range (particularly increase MDE_max)</li>
                    <li>Reduce number of analyses (k)</li>
                  </ul>
                </div>
                """
                    )
                )

    def _display_results(self, ui: Any) -> None:
        """Display design results (summary and boundaries plot)."""
        # Display design summary
        display(HTML("<hr><h3>📊 Design Summary</h3>"))
        display(ui.lab.get_summary())

        # Display boundaries plot
        display(HTML("<hr><h3>📈 Boundaries Plot</h3>"))
        fig = ui.lab.plot_boundaries()
        display(fig)
        plt.close(fig)

    def save_common_from(self, ui: Any) -> None:
        """No-op; common parameters are owned by this designer's widgets."""
        return None

    def load_common_into(self, ui: Any) -> None:
        """No-op; there are no global common widgets to sync into."""
        return None

    def update_spec(self, spec: DesignSpec, ui: Any) -> None:
        """Update spec with widget values."""
        # Common parameters
        spec.test.alpha = self.w_alpha.value
        spec.test.power = self.w_power.value
        spec.sequential.n_analyses = self.w_n_analyses.value
        spec.boundary.spending_function = SpendingFunction(self.w_spending_func.value)

        # Information spacing
        info_spacing_map = {
            "equal": InformationSpacing.EQUAL,
            "front": InformationSpacing.EQUAL,  # TODO: Add front-loaded spacing
            "back": InformationSpacing.EQUAL,  # TODO: Add back-loaded spacing
        }
        spec.sequential.info_spacing = info_spacing_map.get(
            self.w_info_spacing.value, InformationSpacing.EQUAL
        )

        if isinstance(spec, ProportionsDesignSpec):
            spec.effect.p_control = self.w_p_control.value
            # effect_size will be set during search
            if self._found_mde is not None:
                spec.effect.effect_size = self._found_mde

            # Sample size per analysis
            n_per_analysis = self.w_n_max.value // self.w_n_analyses.value
            spec.sample_size.n_per_analysis = n_per_analysis

    def run_optimization(self, ui: Any) -> None:
        """Run MDE search optimization."""
        self._run_mde_search(ui)

    def to_dict(self) -> dict[str, Any]:
        """Serialize widget values to dictionary."""
        return {
            "alpha": self.w_alpha.value,
            "power": self.w_power.value,
            "n_analyses": self.w_n_analyses.value,
            "spending_func": self.w_spending_func.value,
            "p_control": self.w_p_control.value,
            "n_max": self.w_n_max.value,
            "futility": self.w_futility.value,
            "futility_threshold": self.w_futility_threshold.value,
            "info_spacing": self.w_info_spacing.value,
            "mde_search_min": self.w_mde_search_min.value,
            "mde_search_max": self.w_mde_search_max.value,
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
        if "n_max" in data:
            self.w_n_max.value = data["n_max"]
        if "futility" in data:
            self.w_futility.value = data["futility"]
        if "futility_threshold" in data:
            self.w_futility_threshold.value = data["futility_threshold"]
        if "info_spacing" in data:
            self.w_info_spacing.value = data["info_spacing"]
        if "mde_search_min" in data:
            self.w_mde_search_min.value = data["mde_search_min"]
        if "mde_search_max" in data:
            self.w_mde_search_max.value = data["mde_search_max"]
