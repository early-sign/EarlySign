"""
earlysign.reporting.two_proportions
===================================

Two-proportions specific reporter that understands structured payload fields
and can work with both ibis-based and legacy Polars ledgers.

This reporter leverages the structured payload columns for efficient queries
without JSON parsing when using the new ibis-based ledger system.

Examples
--------
With ibis-based ledger (recommended):
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.reporting.two_proportions import TwoPropGSTReporter
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "test")
>>> rep = TwoPropGSTReporter.from_ledger(ledger)

Legacy compatibility with pandas DataFrame:
>>> import pandas as pd
>>> from earlysign.reporting.two_proportions import TwoPropGSTReporter
>>> # Assuming you have a pandas DataFrame with ledger data
>>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})  # your ledger data
>>> rep = TwoPropGSTReporter(df)
"""

from dataclasses import dataclass
import json
from typing import Optional, Dict, Any, List, TYPE_CHECKING

try:
    import polars as pl
    import matplotlib.pyplot as plt

    POLARS_AVAILABLE = True
except ImportError:
    pl = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]
    POLARS_AVAILABLE = False

if TYPE_CHECKING:
    from earlysign.core.ledger import Ledger


@dataclass
class TwoPropGSTReporter:
    """Two-proportions GST progress view with structured payload support."""

    df: Any  # pl.DataFrame when available

    @classmethod
    def from_ledger(cls, ledger: "Ledger") -> "TwoPropGSTReporter":
        """
        Create reporter from ibis-based ledger with efficient structured queries.

        This method uses structured payload columns to avoid JSON parsing.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required for two-proportions reporting")

        # Get data as Polars DataFrame via ibis
        table = ledger.raw_table

        # Convert problematic column types for Polars compatibility
        try:
            # Try to convert UUIDs to strings and handle JSON columns
            table_for_polars = table

            # Convert UUID columns to string if they exist
            if "uuid" in table.columns:
                table_for_polars = table_for_polars.mutate(
                    uuid=table_for_polars.uuid.cast("string")
                )

            polars_df = table_for_polars.to_polars()
        except (NotImplementedError, AttributeError) as e:
            # Fallback: use pandas instead of polars for incompatible types
            pandas_df = table.to_pandas()
            # Convert to polars from pandas if possible
            try:
                import polars as pl

                polars_df = pl.from_pandas(pandas_df)
            except Exception:
                # Final fallback: return pandas DataFrame
                polars_df = pandas_df

        return cls(polars_df)

    @classmethod
    def from_polars_ledger(cls, polars_ledger: Any) -> "TwoPropGSTReporter":
        """Create reporter from PolarsLedger (legacy compatibility)."""
        return cls(polars_ledger.frame())

    def progress_table(self) -> Any:
        """
        Returns one row per look with numeric columns:
        - look, t, z, upper, lower, nA, nB, mA, mB, stopped ('yes'/'no')
        """
        stats = (
            self.df.filter(
                (pl.col("namespace") == "stats")
                & (pl.col("kind") == "updated")
                & (pl.col("payload_type") == "WaldZ")
            )
            .with_columns(
                pl.col("payload")
                .str.json_path_match("$.z")
                .cast(pl.Float64)
                .alias("z"),
                pl.col("payload")
                .str.json_path_match("$.nA")
                .cast(pl.Int64)
                .alias("nA"),
                pl.col("payload")
                .str.json_path_match("$.nB")
                .cast(pl.Int64)
                .alias("nB"),
                pl.col("payload")
                .str.json_path_match("$.mA")
                .cast(pl.Int64)
                .alias("mA"),
                pl.col("payload")
                .str.json_path_match("$.mB")
                .cast(pl.Int64)
                .alias("mB"),
                pl.col("time_index").str.strip_prefix("t").cast(pl.Int64).alias("look"),
            )
            .select("time_index", "ts", "entity", "look", "z", "nA", "nB", "mA", "mB")
        )

        crit = (
            self.df.filter(
                (pl.col("namespace") == "criteria")
                & (pl.col("kind") == "updated")
                & (pl.col("payload_type") == "GSTBoundary")
            )
            .with_columns(
                pl.col("payload")
                .str.json_path_match("$.upper")
                .cast(pl.Float64)
                .alias("upper"),
                pl.col("payload")
                .str.json_path_match("$.lower")
                .cast(pl.Float64)
                .alias("lower"),
                pl.col("payload")
                .str.json_path_match("$.info_time")
                .cast(pl.Float64)
                .alias("t"),
            )
            .select("time_index", "upper", "lower", "t")
        )

        out = (
            stats.join(crit, on="time_index", how="left")
            .with_columns(
                pl.when(pl.col("z").abs() >= pl.col("upper"))
                .then(pl.lit("yes"))
                .otherwise(pl.lit("no"))
                .alias("stopped")
            )
            .sort("look")
        )
        return out

    def _planned_design(self) -> Optional[Dict[str, Any]]:
        """
        Read the last 'design/registered' event (if any) and return the decoded payload dict.
        Expected keys: 'alpha', 'spending', 't_grid' (list of floats).
        """
        q = (
            self.df.filter(
                (pl.col("namespace") == "design") & (pl.col("kind") == "registered")
            )
            .sort("ts")
            .tail(1)
            .to_dicts()
        )
        if not q:
            return None
        payload_str = q[0].get("payload", "")
        try:
            return (
                json.loads(payload_str)
                if isinstance(payload_str, str)
                else dict(payload_str)
            )
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
        if prog.is_empty():
            print("(no progress)")
            return

        # Observed points
        xs_obs: List[float] = prog["t"].to_list()
        zs_obs: List[float] = prog["z"].to_list()

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
        from earlysign.methods.group_sequential.two_proportions import (
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
            # stopped rows are labeled 'yes'
            stopped_rows = prog.filter(pl.col("stopped") == "yes")
            if stopped_rows.height > 0:
                t_stop = float(stopped_rows["t"][-1])
                z_stop = float(stopped_rows["z"][-1])
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
