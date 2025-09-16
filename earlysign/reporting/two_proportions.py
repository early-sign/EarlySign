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
import pandas as pd

from earlysign.core.components import Namespace

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
        table = self.ledger.t

        # Query statistics data (WaldZ) using ibis
        stats_filtered = table.filter(
            (table.labels["namespace"].str == str(Namespace.STATS.value))
            & (table.labels["kind"].str == "updated")
            & (table.payload_type == "WaldZ")
        )

        # Extract JSON payload fields for stats using elegant ibis JSON operations
        stats = stats_filtered.select(
            table.ts,
            table.labels["entity"],
            # Extract look number from step_key (remove 'look-' prefix), handle empty strings
            table.labels["step_key"]
            .str.substr(5)
            .nullif("")
            .coalesce("0")
            .cast("int64")
            .name("look"),
            step_key=table.labels["step_key"],
            # Extract values from JSON payload using elegant syntax
            z=table.payload["z"].cast("float64"),
            nA=table.payload["nA"].cast("int64"),
            nB=table.payload["nB"].cast("int64"),
            mA=table.payload["mA"].cast("int64"),
            mB=table.payload["mB"].cast("int64"),
        )

        # Query criteria data (GSTBoundary) using ibis
        crit_filtered = table.filter(
            (table.labels["namespace"].str == str(Namespace.CRITERIA.value))
            # & (table.labels["kind"].str == "updated")
            & (table.payload_type == "GSTBoundary")
        )

        # Extract JSON payload fields for criteria using elegant ibis syntax
        crit = crit_filtered.select(
            step_key=table.labels["step_key"],
            upper=table.payload["upper"].cast("float64"),
            lower=table.payload["lower"].cast("float64"),
            t=table.payload["info_time"].cast("float64"),
        )

        # Join stats and criteria on step_key
        joined = stats.left_join(crit, "step_key")

        # Add stopped column based on whether |z| >= upper
        # For now, we'll compute this after executing the query since ibis case operations are complex
        result = joined.order_by("look")

        return result.execute()

    def _planned_design(self) -> Optional[Dict[str, Any]]:
        """
        Read the last 'design/registered' event (if any) and return the decoded payload dict.
        Expected keys: 'alpha', 'spending', 't_grid' (list of floats).

        Uses ibis operations to query the ledger directly.
        """
        # Get the base table from the ledger
        table = self.ledger.t

        # Query for design/registered events
        design_events = (
            table.filter(
                (table.labels["namespace"].str == str(Namespace.DESIGN.value))
                & (table.labels["kind"].str == "experiment_design")
            )
            .order_by(table.ts.desc())
            .limit(1)
        )

        # Execute query and get results
        results = design_events.execute()
        if len(results) == 0:
            return None

        # Get the first (and only) result
        if hasattr(results, "iloc"):
            # pandas DataFrame
            row = results.iloc[0]
            payload_str = (
                row.get("payload", "") if hasattr(row, "get") else row["payload"]
            )
        else:
            # Other format
            row = list(results)[0]
            payload_str = (
                row.get("payload", "") if hasattr(row, "get") else row["payload"]
            )

        return (
            json.loads(payload_str)
            if isinstance(payload_str, str)
            else dict(payload_str)
        )

    def plot(self, show: bool = True, mark_stop: bool = True) -> None:
        """
        Plot Wald Z vs information fraction with full planned boundaries.

        - Observed Z trajectory comes from progress_table() (up to stop).
        - Boundaries are recomputed across the full planned t_grid from the design
          so you can see the entire time axis even if the test stopped early.
        """
        prog_df = self.progress_table()

        # Convert DataFrame to records (list of dictionaries)
        if hasattr(prog_df, "to_dict"):
            prog_data = prog_df.to_dict("records")
        else:
            # Fallback for other formats - convert to list
            prog_data = list(prog_df)

        # Add stopped column logic post-query
        for row in prog_data:
            z_val = row.get("z", 0.0)
            upper_val = row.get("upper", float("inf"))
            if z_val is not None and upper_val is not None:
                row["stopped"] = "yes" if abs(z_val) >= upper_val else "no"
            else:
                row["stopped"] = "no"

        # if len(prog_data) == 0:
        #     print("(no progress)")
        #     return

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
        plt.xlim(-0.1, 1.1)  # Show full information time range with margins
        plt.tight_layout()
        if show:
            plt.show()


@dataclass
class TwoPropSafeReporter:
    """Two-proportions safe testing progress view with e-value support."""

    ledger: "Ledger"

    def progress_table(self) -> Any:
        """
        Returns one row per step with numeric columns:
        - step, e_value, threshold, nA, nB, mA, mB, stopped ('yes'/'no')

        Uses pandas operations after getting raw data from ledger.
        """
        # Get the raw table from the ledger
        table = self.ledger.t
        all_data = table.execute()

        # Filter for e-value statistics data
        stats_data = all_data[
            (
                all_data["labels"].apply(
                    lambda x: x.get("namespace") == str(Namespace.STATS.value)
                )
            )
            & (all_data["labels"].apply(lambda x: x.get("kind") == "updated"))
            & (all_data["payload_type"] == "BetaBinomialEValue")
        ].copy()

        # Filter for criteria data (thresholds)
        criteria_data = all_data[
            (
                all_data["labels"].apply(
                    lambda x: x.get("namespace") == str(Namespace.CRITERIA.value)
                )
            )
            & (all_data["payload_type"] == "SafeThreshold")
        ].copy()

        if len(stats_data) == 0:
            return pd.DataFrame()

        # Extract step_key and payload data for stats
        stats_data["step_key"] = stats_data["labels"].apply(lambda x: x.get("step_key"))
        stats_data["e_value"] = stats_data["payload"].apply(lambda x: x.get("e_value"))
        stats_data["nA"] = stats_data["payload"].apply(lambda x: x.get("nA"))
        stats_data["nB"] = stats_data["payload"].apply(lambda x: x.get("nB"))
        stats_data["mA"] = stats_data["payload"].apply(lambda x: x.get("mA"))
        stats_data["mB"] = stats_data["payload"].apply(lambda x: x.get("mB"))

        # Extract step_key and threshold for criteria
        if len(criteria_data) > 0:
            criteria_data["step_key"] = criteria_data["labels"].apply(
                lambda x: x.get("step_key")
            )
            criteria_data["threshold"] = criteria_data["payload"].apply(
                lambda x: x.get("threshold")
            )

        # Select relevant columns and sort by timestamp
        progress_df = (
            stats_data[["ts", "step_key", "e_value", "nA", "nB", "mA", "mB"]]
            .sort_values("ts")
            .reset_index(drop=True)
        )

        # Add step numbers
        progress_df["step"] = range(1, len(progress_df) + 1)

        # Merge with criteria data to get thresholds
        if len(criteria_data) > 0:
            threshold_df = criteria_data[["step_key", "threshold"]]
            progress_df = progress_df.merge(threshold_df, on="step_key", how="left")
        else:
            progress_df["threshold"] = None

        # Add stopped column based on signal data
        signal_data = all_data[
            (
                all_data["labels"].apply(
                    lambda x: x.get("namespace") == str(Namespace.SIGNALS.value)
                )
            )
            & (all_data["labels"].apply(lambda x: x.get("kind") == "decision"))
        ].copy()

        if len(signal_data) > 0:
            signal_data["step_key"] = signal_data["labels"].apply(
                lambda x: x.get("step_key")
            )
            signal_data["action"] = signal_data["payload"].apply(
                lambda x: x.get("action")
            )
            signal_df = signal_data[["step_key", "action"]]
            progress_df = progress_df.merge(signal_df, on="step_key", how="left")
            progress_df["stopped"] = progress_df["action"].apply(
                lambda x: "yes" if x == "stop" else "no"
            )
            progress_df = progress_df.drop("action", axis=1)
        else:
            progress_df["stopped"] = "no"

        # Reorder columns nicely
        columns_order = [
            "step",
            "step_key",
            "e_value",
            "threshold",
            "nA",
            "nB",
            "mA",
            "mB",
            "stopped",
            "ts",
        ]
        progress_df = progress_df[
            [col for col in columns_order if col in progress_df.columns]
        ]

        return progress_df

    def _planned_design(self) -> Optional[Dict[str, Any]]:
        """
        Read the last 'design/registered' event and return the decoded payload dict.
        Expected keys: 'alpha', 'prior_params', etc.

        Uses ibis operations to query the ledger directly.
        """
        # Get the base table from the ledger
        table = self.ledger.t

        # Query for design/registered events
        design_events = (
            table.filter(
                (table.labels["namespace"] == str(Namespace.DESIGN.value))
                & (table.labels["kind"] == "experiment_design")
            )
            .order_by(table.ts.desc())
            .limit(1)
        )

        # Execute query and get results
        results = design_events.execute()
        if len(results) == 0:
            return None

        # Get the first (and only) result
        if hasattr(results, "iloc"):
            # pandas DataFrame
            row = results.iloc[0]
            payload_str = (
                row.get("payload", "") if hasattr(row, "get") else row["payload"]
            )
        else:
            # Other format
            row = list(results)[0]
            payload_str = (
                row.get("payload", "") if hasattr(row, "get") else row["payload"]
            )

        return (
            json.loads(payload_str)
            if isinstance(payload_str, str)
            else dict(payload_str)
        )

    def plot(self, show: bool = True, mark_stop: bool = True) -> None:
        """
        Plot e-value evolution over time with threshold line.

        - Observed e-value trajectory comes from progress_table()
        - Threshold line shows stopping criterion
        """
        prog_df = self.progress_table()

        # Convert DataFrame to records (list of dictionaries) if needed
        if hasattr(prog_df, "to_dict"):
            prog_data = prog_df.to_dict("records")
        else:
            # Fallback for other formats - convert to list
            prog_data = list(prog_df)

        if len(prog_data) == 0:
            print("(no progress data)")
            return

        # Extract observed points from executed data
        xs_obs: List[int] = [
            row.get("step", 0) for row in prog_data if row.get("step") is not None
        ]
        e_values_obs: List[float] = [
            row.get("e_value", 1.0)
            for row in prog_data
            if row.get("e_value") is not None
        ]
        thresholds_obs: List[float] = [
            row.get("threshold", 20.0)
            for row in prog_data
            if row.get("threshold") is not None
        ]

        # Use first threshold as the line (should be constant for safe tests)
        threshold_line = thresholds_obs[0] if thresholds_obs else 20.0

        # Plot
        plt.figure(figsize=(6.5, 4.2))
        # E-value trajectory
        plt.plot(xs_obs, e_values_obs, marker="o", linewidth=2, label="E-value")
        # Threshold line
        plt.axhline(
            y=threshold_line,
            linestyle="--",
            color="red",
            alpha=0.7,
            label=f"Threshold (1/α = {threshold_line:.0f})",
        )

        # Optional: mark stopping step (if any)
        if mark_stop:
            # Find stopped rows
            stopped_rows = [row for row in prog_data if row.get("stopped") == "yes"]
            if stopped_rows:
                last_stop = stopped_rows[-1]
                step_stop = int(last_stop.get("step", 0))
                e_stop = float(last_stop.get("e_value", 1.0))
                plt.scatter(
                    [step_stop], [e_stop], s=70, color="red", zorder=5, label="Stop"
                )

        plt.xlabel("Step")
        plt.ylabel("E-value")
        plt.title("Safe Testing Progress (Two-Proportions)")
        plt.yscale("log")  # E-values can grow very large
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if show:
            plt.show()
