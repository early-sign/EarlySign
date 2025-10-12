"""
Group Sequential Trial Design UI Module.

This module provides an interactive Jupyter widget-based UI for exploring
and optimizing group sequential trial designs.
"""

from typing import Any, Optional, Protocol

import ipywidgets as widgets
from IPython.display import clear_output, display

from earlysign.stats.design.gst.common.config import DesignSpec, ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.optimization import (
    DesignOptimizer,
)
from earlysign.stats.design.gst.common.types import (
    DesignMode,
)
from earlysign.stats.design.gst.fixed_power.ui_ipywidgets import (
    FixedPowerDesigner,
)
from earlysign.stats.design.gst.fixed_timing.ui_ipywidgets import (
    FixedTimingDesigner,
)
from earlysign.stats.design.gst.optimize_asn.ui_ipywidgets import (
    OptimizeASNDesigner,
)
from earlysign.stats.design.gst.optimize_design.ui_ipywidgets import (
    OptimizeDesignDesigner,
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

        # Event handlers
        self.w_mode.observe(self._on_mode_change, "value")

    def _init_mode_controllers(self) -> None:
        """Create per-mode controllers and initialize their saved common state."""
        self._controllers: dict[DesignMode, ModeController] = {
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
