import ibis
import pandas as pd
from earlysign.core.ledger import Ledger
from earlysign.v1.templates.binomial_ab import BinomialABTemplate, BinomialABTaskSpec
import earlysign.schema.ES3.GST as GST
from earlysign.schema.ES3.GST import DecisionStatus
from earlysign.v1.tests.util import BinomialStream
import json


def verify_traces():
    print("=== Verifying Traces ===")

    # 1. Setup Environment
    conn = ibis.duckdb.connect(":memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="trace_test_001")

    # 2. Setup Task & Protocol
    task = BinomialABTaskSpec(
        arms=["control", "treatment"],
        efficacy=GST.EfficacyRequirement(alpha=0.05),
        futility=GST.FutilityRequirement(power=0.8),
        hypotheses=GST.HypothesisSpec(
            h_null="Diff <= 0",
            h_alt="Diff > 0.02",
            test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
            target_effect=GST.BinaryEffectSize(
                proportions={"control": 0.20, "treatment": 0.22}
            ),
        ),
    )

    template = BinomialABTemplate(ledger)
    protocol = template.design(task=task, looks=2)
    template.set_protocol(protocol)

    # 3. generate data that triggers a stop
    # Large effect to ensures stopping
    stream = BinomialStream(
        n_per_batch=5000, p_control=0.20, p_treatment=0.30, seed=42  # Big lift
    )

    # 4. Run until stop
    print("Running simulation...")
    decision_trace = None
    result_trace = None

    for batch in stream:
        template.update(batch)
        result = template.report_progress()
        if (
            result["status"] != "CONTINUE"
            and result["status"] != DecisionStatus.CONTINUE
        ):
            print(f"Stopped with status: {result['status']}")
            break

    # 5. Inspect Ledger
    print("\nlnspecting Ledger...")
    df = ledger.t.execute()

    # Check for Result.BinomialTestResult
    results = df[df["payload_type"] == "Result.BinomialTestResult"]
    if results.empty:
        print("FAIL: No Result.BinomialTestResult found in ledger.")
        exit(1)
    else:
        print(f"PASS: Found {len(results)} Result.BinomialTestResult records.")

    # Check for Decision
    decisions = df[df["payload_type"] == "ABDecisionRecord"]
    if decisions.empty:
        print("FAIL: No ABDecisionRecord found in ledger.")
        exit(1)
    else:
        print(f"PASS: Found {len(decisions)} ABDecisionRecord records.")

    # 6. Verify Lineage
    # Get the last decision (should be the stop)
    last_decision = decisions.iloc[-1]
    last_decision = decisions.iloc[-1]

    # We can check if `Result.BinomialTestResult` has `is_result=True`.
    last_result = results.iloc[-1]
    last_result_labels = (
        json.loads(last_result["labels"])
        if isinstance(last_result["labels"], str)
        else last_result["labels"]
    )

    if not last_result_labels.get("is_result"):
        print("FAIL: Result record is not marked as is_result.")
        exit(1)

    print(f"PASS: Result record marked as is_result.")

    print("Verification Successful.")


if __name__ == "__main__":
    verify_traces()
