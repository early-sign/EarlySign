"""Benchmarks current earlysign.api.ab_tests.BinomialABTest.

This script mirrors the workload from earlysign_dsl_design_26.py so we can
compare runtime/ledger output between the production implementation and the
experimental DSL variants. It spins up two experiments sharing the same ledger
(same table name, different labels) and executes the same batches used in v26.
"""
from __future__ import annotations

import cProfile
import json
import pstats
from typing import Dict, List

import ibis

from earlysign.api.ab_tests import BinomialABTest

SHARED_TABLE = "ledger_api_bench_shared"
DESIGN_PAYLOAD: Dict[str, object] = {
    "alpha": 0.05,
    "planned_max_n": 1000,
    "planned_info_times": [0.25, 0.5, 0.75, 1.0],
    "efficacy": {"style": "alpha_spending", "family": "obrien_fleming"},
    "hypothesis": {"structure": "two_sided_symmetric"},
    "statistic": {"kind": "wald_z", "scale": "z"},
    "futility": {"mode": "none", "binding_mode": "non_binding"},
}

UPDATES_ALPHA: List[Dict[str, int]] = [
    {"nA": 80, "mA": 8, "nB": 80, "mB": 9},
    {"nA": 60, "mA": 6, "nB": 60, "mB": 8},
    {"nA": 200, "mA": 18, "nB": 200, "mB": 25},
]

UPDATES_BETA: List[Dict[str, int]] = [
    {"nA": 100, "mA": 10, "nB": 100, "mB": 12},
    {"nA": 50, "mA": 5, "nB": 50, "mB": 6},
    {"nA": 150, "mA": 10, "nB": 150, "mB": 20},
    {"nA": 100, "mA": 5, "nB": 100, "mB": 35},
]


def run_workload():
    con = ibis.duckdb.connect()
    con.raw_sql(f"DROP TABLE IF EXISTS {SHARED_TABLE}")
    test_a = BinomialABTest(con, "exp_alpha_api", table_name=SHARED_TABLE)
    test_b = BinomialABTest(con, "exp_beta_api", table_name=SHARED_TABLE)

    # test_a.set_design(DESIGN_PAYLOAD)
    # for payload in UPDATES_ALPHA:
    #     test_a.update(payload)

    test_b.set_design(DESIGN_PAYLOAD)
    for payload in UPDATES_BETA:
        test_b.update(payload)
    return con, test_a, test_b


def _sorted_ledger_df(test: BinomialABTest):
    df = test.ledger.show().copy()
    if "labels" in df.columns:
        df["labels_str"] = df["labels"].map(lambda x: json.dumps(x, sort_keys=True))
        sort_cols = ["labels_str"]
    else:
        sort_cols = []
    first_col = df.columns[0]
    sort_cols.append(first_col)
    return df.sort_values(by=sort_cols)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    con, test_a, test_b = run_workload()
    profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(20)
    print("=== exp_alpha_api ===")
    # Ensure all columns are visible in the output for easier inspection
    import pandas as pd
    pd.set_option("display.max_columns", None)
    # print(test_a.ledger.show().drop(columns=["payload_name", "labels"]).to_dict(orient="records"))
    # print(test_a.ledger.show())
    print(test_a.status())
    # print(_sorted_ledger_df(test_a))
    print("=== exp_beta_api ===")
    # print(test_b.ledger.show().drop(columns=["payload_name", "labels"]).to_dict(orient="records"))
    print(test_b.status())
    # print(test_b.ledger.show())
    # print(_sorted_ledger_df(test_b))


if __name__ == "__main__":
    main()
