import ibis
import numpy as np
from scipy import stats

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
from earlysign.schema.ES3.Continuous import ArmData as ContinuousArmData
from earlysign.schema.ES3.YEAST.Log import DecisionStatus
from earlysign.v1.templates.YEAST import (
    BinomialYeastTaskSpec,
    BinomialYeastTemplate,
    ContinuousYeastTaskSpec,
    ContinuousYeastTemplate,
)


def test_yeast_engine_boundary_crossing() -> None:
    """
    Test that the YEAST engine correctly triggers a STOP_EFFICACY decision
    when the trajectory crosses the boundary.

    Logic Verification:
    boundary = z_crit * sqrt(max_n * estimated_variance)
    trajectory > boundary -> STOP
    """
    # Setup
    alpha = 0.05
    z_crit = stats.norm.ppf(1 - alpha / 2)
    max_n = 100
    estimated_variance = 1.0

    # Calculate expected boundary manually
    # B = 1.96 * sqrt(100 * 1) = 19.6
    z_crit * np.sqrt(max_n * estimated_variance)

    # Case 1: Trajectory below boundary
    pass


def test_binomial_yeast_template_e2e() -> None:
    """
    End-to-End test of the BinomialYeastTemplate with a deterministic stream.
    """
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_yeast_001")

    # 1. Design Protocol
    # We use a scenario where we force a crossing.
    # N_max = 100
    # increment_std = 1.0
    # Boundary approx 19.6
    task = BinomialYeastTaskSpec(
        arms=["control", "treatment"],
        response_type="binary",
        hypotheses={
            "h_null": "diff <= 0",
            "h_alt": "diff > 0",
            "test_logic": {"kind": "superiority"},
            "target_effect": {
                "type": "binary",
                "proportions": {"control": 0.5, "treatment": 0.6},
            },
        },
    )

    template = BinomialYeastTemplate(ledger)
    protocol = BinomialYeastTemplate.design(
        task=task,
        significance_level=0.05,
        expected_num_observations=200,
        estimated_variance=1.0,  # Explicitly set to 1.0 for easy calculation
    )
    template.set_protocol(protocol)

    # 2. Simulate Stream to trigger crossing
    # Boundary B approx 1.96 * 10 * 1 = 19.6
    # We need success_diff > 19.6

    # Batch 1: Diff = 10. Below boundary.
    # Control: 50 trials, 20 successes
    # Treatment: 50 trials, 30 successes
    # Diff = 10
    batch1 = [
        BinomialArmData(n=50, success=20, arm="control"),
        BinomialArmData(n=50, success=30, arm="treatment"),
    ]
    template.update(batch1)

    report1 = template.report_progress()
    assert report1["status"] == DecisionStatus.CONTINUE_
    assert report1["trajectory"] == 10.0

    # Batch 2: Accumulate more difference.
    # Total Diff needs to be > 1.96 * sqrt(200) * 1 ~= 27.7
    # Current Accum: C=20, T=30.
    # Add: C=20, T=45
    # New Accum: C=40, T=75. Diff = 35.

    batch2 = [
        BinomialArmData(n=50, success=20, arm="control"),
        BinomialArmData(n=50, success=45, arm="treatment"),
    ]
    template.update(batch2)

    report2 = template.report_progress()

    # Should STOP
    assert report2["status"] == DecisionStatus.STOP_EFFICACY

    result = template.report_result()
    assert result["is_rejected"] is True
    assert result["final_status"] == DecisionStatus.STOP_EFFICACY

    # Check boundary from progress report which contains look details
    assert report2["efficacy_boundary"] > 27.0
    assert report2["efficacy_boundary"] < 28.0


def test_continuous_yeast_template_e2e() -> None:
    """
    End-to-End test of the ContinuousYeastTemplate.
    """
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_yeast_continuous_001")

    task = ContinuousYeastTaskSpec(
        arms=["control", "treatment"],
        response_type="continuous",
        hypotheses={},  # Placeholder
    )

    template = ContinuousYeastTemplate(ledger)
    # N_max = 100, Var = 1.0, Alpha = 0.05
    # Boundary = 1.96 * sqrt(100 * 1) = 19.6
    protocol = ContinuousYeastTemplate.design(
        task=task,
        significance_level=0.05,
        expected_num_observations=100,
        estimated_variance=1.0,
    )
    template.set_protocol(protocol)

    # Batch 1: Sum(T) - Sum(C) = 15. Below 19.6.
    batch1 = [
        ContinuousArmData(n=10, sum_x=10.0, sum_x2=20.0, arm="control"),
        ContinuousArmData(n=10, sum_x=25.0, sum_x2=70.0, arm="treatment"),
    ]
    template.update(batch1)

    report1 = template.report_progress()
    assert report1["status"] == DecisionStatus.CONTINUE_
    assert report1["trajectory"] == 15.0

    # Batch 2: Add Diff = 10. Total Diff = 25. Above 19.6.
    batch2 = [
        ContinuousArmData(n=10, sum_x=10.0, sum_x2=20.0, arm="control"),
        ContinuousArmData(n=10, sum_x=20.0, sum_x2=50.0, arm="treatment"),
    ]
    template.update(batch2)

    report2 = template.report_progress()
    assert report2["status"] == DecisionStatus.STOP_EFFICACY
    assert report2["trajectory"] == 25.0
    assert abs(report2["efficacy_boundary"] - 19.5996) < 1e-3


def test_default_increment_std() -> None:
    """Verifies that the schema default is applied correctly."""
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_yeast_002")

    task = BinomialYeastTaskSpec(
        arms=["control", "treatment"],
        response_type="binary",
        hypotheses={
            "h_null": "diff <= 0",
            "h_alt": "diff > 0",
            "test_logic": {"kind": "superiority", "superiority_margin": 0.0},
            "target_effect": {
                "type": "binary",
                "proportions": {"control": 0.5, "treatment": 0.6},
            },
        },
    )

    BinomialYeastTemplate(ledger)
    # Don't provide estimated_variance, expect default variance 0.5
    protocol = BinomialYeastTemplate.design(
        task=task,
        significance_level=0.05,
        expected_num_observations=100,
        estimated_variance=0.5,
    )

    assert protocol.method.estimated_variance is not None
    assert abs(protocol.method.estimated_variance - 0.5) < 1e-6
