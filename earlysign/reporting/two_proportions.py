"""
earlysign.reporting.two_proportions
===================================

Two-proportions specific reporter that understands:
- 'WaldZ' (stats) payloads
- 'GSTBoundary' (criteria) payloads

Doctest (smoke; no plotting):
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>> from earlysign.core.names import Namespace
>>> from earlysign.reporting.two_proportions import TwoPropGSTReporter
>>> L = PolarsLedger()
>>> # minimal two rows to exercise parsing paths
>>> L.write_event(time_index="t1", namespace=Namespace.STATS, kind="updated",
...               experiment_id="E", step_key="s1", payload_type="WaldZ",
...               payload={"z":1.2,"se":0.1,"nA":10,"nB":10,"mA":1,"mB":2,"pA_hat":0.1,"pB_hat":0.2})
>>> L.write_event(time_index="t1", namespace=Namespace.CRITERIA, kind="updated",
...               experiment_id="E", step_key="s1", payload_type="GSTBoundary",
...               payload={"upper":2.5,"lower":-2.5,"info_time":0.25,"alpha_i":0.01})
>>> rep = TwoPropGSTReporter(L.frame())
>>> pt = rep.progress_table(); pt.height == 1 and "z" in pt.columns
True
"""


from dataclasses import dataclass
import json
from typing import Optional, Dict, Any, List

import polars as pl
import matplotlib.pyplot as plt

@dataclass
class TwoPropGSTReporter:
    """Two-proportions GST progress view (table and simple plot)."""
    df: pl.DataFrame

    def progress_table(self) -> pl.DataFrame:
        """
        Returns one row per look with numeric columns:
        - look, t, z, upper, lower, nA, nB, mA, mB, stopped ('yes'/'no')
        """
        stats = (
            self.df
            .filter((pl.col("namespace") == "stats") & (pl.col("kind") == "updated") & (pl.col("payload_type") == "WaldZ"))
            .with_columns(
                pl.col("payload").str.json_path_match("$.z").cast(pl.Float64).alias("z"),
                pl.col("payload").str.json_path_match("$.nA").cast(pl.Int64).alias("nA"),
                pl.col("payload").str.json_path_match("$.nB").cast(pl.Int64).alias("nB"),
                pl.col("payload").str.json_path_match("$.mA").cast(pl.Int64).alias("mA"),
                pl.col("payload").str.json_path_match("$.mB").cast(pl.Int64).alias("mB"),
                pl.col("time_index").str.strip_prefix("t").cast(pl.Int64).alias("look"),
            )
            .select("time_index", "ts", "entity", "look", "z", "nA", "nB", "mA", "mB")
        )

        crit = (
            self.df
            .filter((pl.col("namespace") == "criteria") & (pl.col("kind") == "updated") & (pl.col("payload_type") == "GSTBoundary"))
            .with_columns(
                pl.col("payload").str.json_path_match("$.upper").cast(pl.Float64).alias("upper"),
                pl.col("payload").str.json_path_match("$.lower").cast(pl.Float64).alias("lower"),
                pl.col("payload").str.json_path_match("$.info_time").cast(pl.Float64).alias("t"),
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
            self.df
            .filter((pl.col("namespace") == "design") & (pl.col("kind") == "registered"))
            .sort("ts")
            .tail(1)
            .to_dicts()
        )
        if not q:
            return None
        payload_str = q[0].get("payload", "")
        try:
            return json.loads(payload_str) if isinstance(payload_str, str) else dict(payload_str)
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
        from earlysign.methods.group_sequential.two_proportions import lan_demets_spending

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
        plt.plot(t_grid_full, uppers_full, linestyle="--", marker=None, label="Upper boundary")
        plt.plot(t_grid_full, lowers_full, linestyle="--", marker=None, label="Lower boundary")
        # Observed Z up to stop
        plt.plot(xs_obs, zs_obs, marker="o", label="Observed Wald Z")

        # Optional: mark stopping look (if any)
        if mark_stop:
            # stopped rows are labeled 'yes'
            stopped_rows = prog.filter(pl.col("stopped") == "yes")
            if stopped_rows.height > 0:
                t_stop = float(stopped_rows["t"][-1])
                z_stop = float(stopped_rows["z"][-1])
                plt.scatter([t_stop], [z_stop], s=70, color="red", zorder=5, label="Stop")

        plt.axhline(0.0, linestyle=":", linewidth=1)
        plt.xlabel("Information fraction (t)")
        plt.ylabel("Z")
        plt.title("GST Progress (Two-Proportions)")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
