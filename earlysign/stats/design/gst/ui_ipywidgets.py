"""
Group Sequential Trial Design UI Module.

This module provides an interactive Jupyter widget-based UI for exploring
and optimizing group sequential trial designs.
"""

import json
from typing import Any, Optional, Protocol

import ipywidgets as widgets
from IPython.display import clear_output, display

from earlysign.stats.design.gst.common.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.optimization_adapter import (
    DesignOptimizer,
)
from earlysign.stats.design.gst.common.types import (
    DesignMode,
)
from earlysign.stats.design.gst.designers.fixed_power import FixedPowerDesigner
from earlysign.stats.design.gst.designers.fixed_timing import FixedTimingDesigner
from earlysign.stats.design.gst.designers.nmax_fixed_min_mde import (
    NMaxFixedMinMDEDesigner,
)
from earlysign.stats.design.gst.designers.optimize_asn import OptimizeASNDesigner
from earlysign.stats.design.gst.designers.optimize_design import OptimizeDesignDesigner


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

    # ---- Spec update hooks ----
    def update_spec(self, spec: DesignSpec, ui: "GSTDesignUI") -> None:
        """Write both common and mode-specific values into the given DesignSpec."""

        ...

    # ---- State serialization ----
    def to_dict(self) -> dict[str, Any]:
        """Serialize controller's widget values to a dictionary.

        Returns
        -------
        dict
            Dictionary containing all widget values for this controller.
        """
        ...

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore controller's widget values from a dictionary.

        Uses a best-effort approach: loads available fields and ignores
        missing or unknown fields. This ensures compatibility across versions.

        Parameters
        ----------
        data : dict
            Dictionary containing widget values to restore. Missing fields
            are left at their current (default) values.
        """
        ...


# Note: Controller classes (FixedTimingDesigner, OptimizeASNDesigner, etc.)
# are now imported from their respective modules as Designer classes.


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

    Notes
    -----
    Multiple independent instances can be created for side-by-side comparison:

    >>> ui1 = GSTDesignUI()  # doctest: +SKIP
    >>> ui2 = GSTDesignUI()  # doctest: +SKIP
    """

    def __init__(self, initial_spec: Optional[DesignSpec] = None):
        """
        Initialize the UI.

        Parameters
        ----------
        initial_spec : DesignSpec, optional
            Initial design specification. Defaults to ProportionsDesignSpec.
        """

        self.spec = initial_spec or ProportionsDesignSpec()
        self.lab = DesignLab(self.spec)
        self.optimizer: Optional[DesignOptimizer] = None
        self.current_mode = DesignMode.NMAX_FIXED_MIN_MDE

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
                ("N-Max Fixed → Find Min MDE", DesignMode.NMAX_FIXED_MIN_MDE.value),
                ("Fixed Timing → Compute Boundaries", DesignMode.FIXED_TIMING.value),
                ("Fixed Max N → Optimize ASN", DesignMode.OPTIMIZE_ASN.value),
                (
                    "Flexible Parameters → Optimize Design",
                    DesignMode.OPTIMIZE_DESIGN.value,
                ),
                ("Fixed Power → Compute Required N", DesignMode.FIXED_POWER.value),
            ],
            value=DesignMode.NMAX_FIXED_MIN_MDE.value,
            description="Design Mode:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="600px"),
        )

        # Event handlers
        self.w_mode.observe(self._on_mode_change, "value")

    def _init_mode_controllers(self) -> None:
        """Create per-mode controllers and initialize their saved common state."""
        self._controllers: dict[DesignMode, ModeController] = {
            DesignMode.NMAX_FIXED_MIN_MDE: NMaxFixedMinMDEDesigner(),
            DesignMode.FIXED_TIMING: FixedTimingDesigner(),
            DesignMode.OPTIMIZE_ASN: OptimizeASNDesigner(),
            DesignMode.OPTIMIZE_DESIGN: OptimizeDesignDesigner(),
            DesignMode.FIXED_POWER: FixedPowerDesigner(),
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

    def to_json(self) -> str:
        """
        Serialize current UI state to JSON string.

        This method captures the current state of all widget values across all modes,
        allowing the design configuration to be saved and restored later.

        Returns
        -------
        str
            JSON string representation of the UI state.

        Examples
        --------
        >>> ui = GSTDesignUI()  # doctest: +SKIP
        >>> json_str = ui.to_json()  # Get JSON string  # doctest: +SKIP
        >>> # Save to file if needed
        >>> with open("config.json", "w") as f:  # doctest: +SKIP
        ...     f.write(json_str)  # doctest: +SKIP
        """
        state = {
            "version": "1.0",
            "current_mode": self.current_mode.value,
            "controllers": {
                mode.value: controller.to_dict()
                for mode, controller in self._controllers.items()
            },
        }

        return json.dumps(state, indent=2)

    def from_json(self, json_str: str) -> None:
        """
        Restore UI state from JSON string with best-effort loading.

        This method uses a best-effort approach: it loads all available fields from
        the saved configuration and leaves missing fields at their default values.
        This ensures forward and backward compatibility across versions.

        Parameters
        ----------
        json_str : str
            JSON string containing the saved state.

        Raises
        ------
        json.JSONDecodeError
            If the input is not valid JSON.

        Notes
        -----
        The version field is informational only. The method will attempt to load
        any JSON configuration regardless of version, loading available fields
        and ignoring missing or unknown fields.

        Examples
        --------
        >>> ui = GSTDesignUI()  # doctest: +SKIP
        >>> json_str = ui.to_json()  # doctest: +SKIP
        >>> ui.from_json(json_str)  # Restore state  # doctest: +SKIP
        >>> # Or load from file
        >>> with open("config.json", "r") as f:  # doctest: +SKIP
        ...     ui.from_json(f.read())  # doctest: +SKIP
        >>> # Even old versions will load successfully
        >>> old_json = '{"version": "0.9", "controllers": {...}}'  # doctest: +SKIP
        >>> ui.from_json(old_json)  # Works! Loads available fields  # doctest: +SKIP
        """
        state = json.loads(json_str)

        # Version is informational only - we use best-effort loading
        # so any version can be loaded as long as the structure is valid

        # Restore each controller's state (best-effort)
        controllers_data = state.get("controllers", {})
        for mode_value, controller_data in controllers_data.items():
            try:
                mode = DesignMode(mode_value)
                if mode in self._controllers:
                    # Controller's from_dict also uses best-effort approach
                    self._controllers[mode].from_dict(controller_data)
            except (ValueError, KeyError):
                # Skip unknown modes (forward compatibility)
                pass

        # Switch to the saved current mode if it exists
        if "current_mode" in state:
            try:
                saved_mode = DesignMode(state["current_mode"])
                # Always update the mode and re-render, even if it's the same mode
                # This ensures the UI reflects the restored controller state
                self.current_mode = saved_mode
                self.w_mode.value = saved_mode.value
                controller = self._controllers[self.current_mode]
                self._render_mode_specific(controller)
            except (ValueError, KeyError):
                # Keep current mode if saved mode is unknown
                pass

        # Update the design with the restored values
        try:
            self._update_design()
        except Exception:
            # If design update fails, continue - UI is still usable
            pass
