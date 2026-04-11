"""
Visualization-related schema and result objects.
"""

import io
from typing import Any, Optional, cast

import matplotlib.figure
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg


class VisualizationResult(dict[str, Any]):
    """
    A specialized dictionary for plotting results.
    Provides rich representation in Jupyter notebooks by automatically
    displaying both the summary table and the figure.
    """

    def __init__(
        self,
        summary: Optional[pd.DataFrame] = None,
        figure: Optional[matplotlib.figure.Figure] = None,
        **kwargs: Any,
    ):
        if figure is not None and figure.canvas is None:
            figure.set_canvas(FigureCanvasAgg(figure))
        super().__init__(summary=summary, figure=figure, **kwargs)

    @property
    def summary(self) -> Optional[pd.DataFrame]:
        return self.get("summary")

    @property
    def figure(self) -> Optional[matplotlib.figure.Figure]:
        val = self.get("figure")
        if val is None:
            return None
        return cast(matplotlib.figure.Figure, val)

    def _repr_html_(self) -> str:
        """
        Jupyter notebook rich representation.
        Returns a single HTML string containing the summary table and the plot.
        """
        parts = []

        # 1. Summary Table
        if self.summary is not None:
            # Use to_html() for dataframes/stylers
            if hasattr(self.summary, "to_html"):
                parts.append(self.summary.to_html())
            else:
                # Fallback for plain dataframes if needed
                parts.append(pd.DataFrame(self.summary).to_html())

        # 2. Figure (SVG encoded)
        if self.figure is not None:
            buf = io.StringIO()
            self.figure.savefig(buf, format="svg", bbox_inches="tight")
            buf.seek(0)
            svg_str = buf.getvalue()
            parts.append(f'<div style="margin-top: 10px;">' f"{svg_str}" f"</div>")

        if not parts:
            return ""

        return "\n".join(parts)
