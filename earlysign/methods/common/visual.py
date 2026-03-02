"""
Visualization-related schema and result objects.
"""

import base64
import io
from typing import Any, Optional

import matplotlib.figure
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg


class VisualizationResult(dict):
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
        return self["figure"]

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

        # 2. Figure (Base64 encoded)
        if self.figure is not None:
            buf = io.BytesIO()
            # Use a high DPI for better quality in notebooks
            self.figure.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode("utf-8")
            parts.append(
                f'<div style="margin-top: 10px;">'
                f'<img src="data:image/png;base64,{img_str}" />'
                f"</div>"
            )

        if not parts:
            return ""

        return "\n".join(parts)
