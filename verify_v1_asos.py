"""
Verification Script: Pattern G (EarlySign v1) - ASOS Dataset with Intermediate Facts.
Demonstrates:
1. Dynamic Protocol Design.
2. Progressive Ingestion.
3. Tiered Projection: Intermediate Facts (Tier 1) -> Z-Analytics (Tier 2).
4. Correct look timing (0.65, 0.8, 1.0).
"""

import os
import pandas as pd
import numpy as np
import ibis
from pydantic import BaseModel
from typing import List, Any, Optional

from earlysign.core.ledger import Ledger
from earlysign.v1.methods.protocols import GSTProtocol
from earlysign.v1.methods.group_sequential.design_planner import DesignPlanner
from earlysign.v1.stats.binary import BinomialSummaryFact, BatchObservation
from earlysign.v1.templates.binomial_ab import BinomialABTemplate


# --- 1. Domain Models ---


class GSTResult(BaseModel):
    look: int
    z_stat: float
    boundary: float
    is_rejected: bool


class DecisionRecord(BaseModel):
    status: str
    message: str


# --- 2. Utils ---
def get_increments(df: pd.DataFrame):
    df = df.sort_values("time_since_start")
    sc = df["count_c"] * df["mean_c"]
    dn_c = df["count_c"].diff().fillna(df["count_c"]).astype(int)
    ds_c = sc.diff().fillna(sc).astype(int)
    st = df["count_t"] * df["mean_t"]
    dn_t = df["count_t"].diff().fillna(df["count_t"]).astype(int)
    ds_t = st.diff().fillna(st).astype(int)
    return dn_c, ds_c, dn_t, ds_t


# --- 3. Main Workflow ---
def main():
    print("=== Pattern G Verification (ASOS + BinomialABTemplate) ===")

    con = ibis.duckdb.connect(":memory:")
    ledger = Ledger(con, "events")
    ledger.ensure()

    # Load Data (Experiment 3b4300, Treatment 1)
    data_path = "docs/source/tutorial/data/asos_digital_experiments_dataset.parquet"
    df_raw = pd.read_parquet(data_path)
    exp_id = "3b4300"
    df = df_raw[
        (df_raw["experiment_id"] == exp_id)
        & (df_raw["metric_id"] == 1)
        & (df_raw["variant_id"] == 1)
    ].copy()

    dn_c, ds_c, dn_t, ds_t = get_increments(df)
    df["dn_c"], df["ds_c"], df["dn_t"], df["ds_t"] = dn_c, ds_c, dn_t, ds_t
    df = df[(df["dn_c"] > 0) | (df["dn_t"] > 0)].copy()

    # 1. Initialize Template and Protocol
    protocol = GSTProtocol(alpha=0.05, power=0.8, delta=0.005, K=3)
    trial = BinomialABTemplate(ledger)
    trial.set_protocol(protocol)

    # 2. Design (Calculated based on first look for p_control)
    p_control = df.iloc[0]["mean_c"]
    planner = DesignPlanner()
    design = planner.plan_binomial_ab(
        alpha=protocol.alpha,
        power=protocol.power,
        p_control=p_control,
        delta=protocol.delta,
        k=protocol.K,
    )

    trial.update_protocol(
        n_max=design["n_max"],
        milestones=[0.65, 0.8, 1.0],
        boundaries=design["boundaries"],
    )

    print(
        f"-> Planned Design: n_max={design['n_max']}, Boundaries={[round(b, 2) for b in design['boundaries']]}"
    )

    # 3. Execution Loop
    for i, (_, row) in enumerate(df.iterrows()):
        # Prepare minibatch
        batch = []
        if row["dn_c"] > 0:
            batch.append(
                BatchObservation(n=row["dn_c"], success=row["ds_c"], variant="C")
            )
        if row["dn_t"] > 0:
            batch.append(
                BatchObservation(n=row["dn_t"], success=row["ds_t"], variant="T")
            )

        # Update Trial (Ingest -> Read -> Analyze -> Decide)
        res = trial.update(batch)

        if res["look"]:
            print(
                f"\n   >>> Look {res['look']} Triggered (Info Frac: {res['info_frac']:.2f})"
            )
            print(f"       Control: n and s reconstructed via logic.")
            print(
                f"       Z: {res['z_stat']:.4f} (Bound: {trial.boundaries[res['look']-1]:.2f}) - Reject: {res['is_rejected']}"
            )

            if res["is_rejected"]:
                print(f"       [DECISION] STOP.")
                break

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    main()
