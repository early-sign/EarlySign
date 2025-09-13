"""
earlysign.reporting.two_proportions
===================================

Two-proportions specific reporter that understands structured payload fields
and works with ibis-based ledgers for backend-agnostic data operations.

This reporter leverages the ibis-framework for efficient cross-backend queries
and uses JSON operations to extract structured payload data.

Examples
--------
With ibis-based ledger:
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.reporting.two_proportions import TwoPropGSTReporter
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "test")
>>> rep = TwoPropGSTReporter(ledger)  # Direct dataclass initialization
"""

from dataclasses import dataclass
import json
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


@dataclass
class TwoPropGSTReporter:
    """Two-proportions GST progress view with structured payload support."""

    ledger: "Ledger"

    def progress_table(self) -> Any:
        """
        Returns one row per look with numeric columns:
        - look, t, z, upper, lower, nA, nB, mA, mB, stopped ('yes'/'no')

        Uses ibis operations to query the ledger directly.
        """
        # Get the base table from the ledger
        table = self.ledger.table

        # Query statistics data (WaldZ) using ibis
        stats_filtered = table.filter(
            (table.namespace == "stats")
            & (table.kind == "updated")
            & (table.payload_type == "WaldZ")
        )

        # Extract JSON payload fields for stats using elegant ibis JSON operations
        stats = stats_filtered.select(
            table.time_index,
            table.ts,
            table.entity,
            # Extract look number from time_index (remove 't' prefix)
            table.time_index.substr(2).cast("int64").name("look"),
            # Extract values from JSON payload using elegant syntax
            z=table.payload["z"].cast("float64"),
            nA=table.payload["nA"].cast("int64"),
            nB=table.payload["nB"].cast("int64"),
            mA=table.payload["mA"].cast("int64"),
            mB=table.payload["mB"].cast("int64"),
        )

        # Query criteria data (GSTBoundary) using ibis
        crit_filtered = table.filter(
            (table.namespace == "criteria")
            & (table.kind == "updated")
            & (table.payload_type == "GSTBoundary")
        )

        # Extract JSON payload fields for criteria using elegant ibis syntax
        crit = crit_filtered.select(
            table.time_index,
            upper=table.payload["upper"].cast("float64"),
            lower=table.payload["lower"].cast("float64"),
            t=table.payload["info_time"].cast("float64"),
        )

        # Join stats and criteria on time_index
        joined = stats.left_join(crit, "time_index")

        # Add stopped column based on whether |z| >= upper
        # For now, we'll compute this after executing the query since ibis case operations are complex
        result = joined.order_by("look")

        return result

    def _planned_design(self) -> Optional[Dict[str, Any]]:
        """
        Read the last 'design/registered' event (if any) and return the decoded payload dict.
        Expected keys: 'alpha', 'spending', 't_grid' (list of floats).

        Uses ibis operations to query the ledger directly.
        """
        # Get the base table from the ledger
        table = self.ledger.table

        # Query for design/registered events
        design_events = (
            table.filter((table.namespace == "design") & (table.kind == "registered"))
            .order_by(table.ts.desc())
            .limit(1)
        )

        # Execute query and get results
        try:
            results = design_events.execute()
            if not results:
                return None

            # Get the first (and only) result
            row = list(results)[0]
            payload_str = row.get("payload", "")

            try:
                return (
                    json.loads(payload_str)
                    if isinstance(payload_str, str)
                    else dict(payload_str)
                )
            except Exception:
                return None
        except Exception:
            return None

    def plot(self, show: bool = True, mark_stop: bool = True) -> None:
        """
        Plot Wald Z vs information fraction with full planned boundaries.

        - Observed Z trajectory comes from progress_table() (up to stop).
        - Boundaries are recomputed across the full planned t_grid from the design
          so you can see the entire time axis even if the test stopped early.
        """
        prog = self.progress_table()

        # Execute the ibis query to get actual data
        try:
            prog_data = list(prog.execute())
            # Add stopped column logic post-query
            for row in prog_data:
                z_val = row.get("z", 0.0)
                upper_val = row.get("upper", float("inf"))
                if z_val is not None and upper_val is not None:
                    row["stopped"] = "yes" if abs(z_val) >= upper_val else "no"
                else:
                    row["stopped"] = "no"
        except Exception as e:
            print(f"Error executing progress query: {e}")
            return

        if not prog_data:
            print("(no progress)")
            return

        # Extract observed points from executed data
        xs_obs: List[float] = [
            row.get("t", 0.0) for row in prog_data if row.get("t") is not None
        ]
        zs_obs: List[float] = [
            row.get("z", 0.0) for row in prog_data if row.get("z") is not None
        ]

        # Planned design (preferred) or fallback to observed 't'
        design = self._planned_design()
        if design and "t_grid" in design:
            t_grid_full: List[float] = list(design["t_grid"])
            alpha_total = float(design.get("alpha", 0.05))
            style = str(design.get("spending", "obf"))
        else:
            # Fallback: just use observed t's for x and assume OBF+0.05
            t_grid_full = xs_obs
            alpha_total = 0.05
            style = "obf"

        # Recompute full boundaries on planned grid
        from scipy.stats import norm
        from earlysign.stats.common.group_sequential import (
            lan_demets_spending,
        )

        uppers_full, lowers_full = [], []
        for t in t_grid_full:
            cum_alpha = lan_demets_spending(alpha_total, float(t), style)
            alpha_i = max(min(cum_alpha, alpha_total), 1e-12)
            thr = float(norm.ppf(1 - alpha_i / 2))
            uppers_full.append(thr)
            lowers_full.append(-thr)

        # Plot
        plt.figure(figsize=(6.5, 4.2))
        # Full boundaries across the design
        plt.plot(
            t_grid_full,
            uppers_full,
            linestyle="--",
            marker="s",
            markersize=4,
            label="Upper boundary",
        )
        plt.plot(
            t_grid_full,
            lowers_full,
            linestyle="--",
            marker="s",
            markersize=4,
            label="Lower boundary",
        )
        # Observed Z up to stop
        plt.plot(xs_obs, zs_obs, marker="o", label="Observed Wald Z")

        # Optional: mark stopping look (if any)
        if mark_stop:
            # Find stopped rows
            stopped_rows = [row for row in prog_data if row.get("stopped") == "yes"]
            if stopped_rows:
                last_stop = stopped_rows[-1]
                t_stop = float(last_stop.get("t", 0))
                z_stop = float(last_stop.get("z", 0))
                plt.scatter(
                    [t_stop], [z_stop], s=70, color="red", zorder=5, label="Stop"
                )

        plt.axhline(0.0, linestyle=":", linewidth=1)
        plt.xlabel("Information fraction (t)")
        plt.ylabel("Z")
        plt.title("GST Progress (Two-Proportions)")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
