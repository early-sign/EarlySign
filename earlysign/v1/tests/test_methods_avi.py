import ibis

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.AVI.Log import DecisionStatus
from earlysign.schema.ES3.Binomial import ArmData as BinomialArmData
from earlysign.v1.templates.AVI import (
    BinomialAVITemplate,
)


def test_binomial_gavi_template_e2e() -> None:
    """
    End-to-End test of the BinomialAVITemplate using GAVI algorithm.
    """
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_gavi_binom_001")

    # Design GAVI
    # Alpha 0.05, Variance 0.25 (e.g., p=0.5, var = p(1-p)=0.25), Two-sided
    # max_n = 1000
    template = BinomialAVITemplate(ledger)
    protocol = BinomialAVITemplate.design_gavi(
        arms=["control", "treatment"],
        alpha=0.05,
        variance=0.25,
        sides="two",
        max_n=1000,
    )
    template.set_protocol(protocol)

    # 1. Update with small difference
    batch1 = [
        BinomialArmData(n=100, success=50, arm="control"),  # p=0.5
        BinomialArmData(n=100, success=52, arm="treatment"),  # p=0.52
    ]
    template.update(batch1)
    report1 = template.report_progress()
    # Diff = 0.02. Boundary should be wide early on.
    assert report1["status"] == DecisionStatus.CONTINUE_
    assert abs(report1["trajectory"] - 0.02) < 1e-9
    assert report1["boundary"] > 0.02

    # 2. Update with large difference crossing boundary
    # Control p=0.5, Treatment p=0.7. Diff = 0.2
    # n=500
    batch2 = [
        BinomialArmData(n=400, success=200, arm="control"),  # Total 500, 250 success
        BinomialArmData(n=400, success=298, arm="treatment"),  # Total 500, 350 success
    ]
    template.update(batch2)
    report2 = template.report_progress()

    # Check current status
    # Total n per arm = 500.
    # Total control: 50 + 200 = 250 (p=0.5)
    # Total treatment: 52 + 298 = 350 (p=0.7)
    # Diff = 0.2
    assert (
        report2["sample_n"] == 1000
    )  # Total for both arms? No, engine returns n_c + n_t = 1000.
    assert abs(report2["trajectory"] - 0.2) < 1e-9

    # GAVI boundary should be crossed with such large difference and n=500.
    # CI width at n=500 with sigma2=0.25 roughly?
    # V = 2 * 0.25 / 500 = 0.001.
    # sqrt(V) = 0.0316
    # Boundary multiplier > 2 but unlikely > 6 (0.2/0.0316)
    assert report2["status"] == DecisionStatus.STOP_EFFICACY

    result = template.report_result()
    assert result["is_rejected"] is True


def test_binomial_m_sprt_template_e2e() -> None:
    """
    End-to-End test of the BinomialAVITemplate using mSPRT algorithm.
    """
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_msprt_binom_001")

    # Design mSPRT
    # MDE = 0.1
    template = BinomialAVITemplate(ledger)
    protocol = BinomialAVITemplate.design_m_sprt(
        arms=["control", "treatment"],
        alpha=0.05,
        variance=0.25,
        sides="two",
        mde=0.1,
    )
    template.set_protocol(protocol)

    # 1. Update with small difference
    batch1 = [
        BinomialArmData(n=100, success=50, arm="control"),
        BinomialArmData(n=100, success=51, arm="treatment"),
    ]
    template.update(batch1)
    report1 = template.report_progress()
    assert report1["status"] == DecisionStatus.CONTINUE_

    # 2. Update with large difference
    # Diff = 0.15 ( > MDE=0.1)
    batch2 = [
        BinomialArmData(
            n=400, success=200, arm="control"
        ),  # Total 500, 250 success (0.5)
        BinomialArmData(
            n=400, success=274, arm="treatment"
        ),  # Total 500, 325 success (0.65)
    ]
    template.update(batch2)
    report2 = template.report_progress()

    # Diff = 0.15
    assert abs(report2["trajectory"] - 0.15) < 1e-9
    # mSPRT should reject if effect size > MDE was sustained or accumulated enough evidence.
    assert report2["status"] == DecisionStatus.STOP_EFFICACY


def test_gavi_one_sided() -> None:
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_gavi_one_sided")

    template = BinomialAVITemplate(ledger)
    # One-sided. Positive trajectory needed.
    protocol = BinomialAVITemplate.design_gavi(
        arms=["control", "treatment"],
        alpha=0.05,
        variance=0.25,
        sides="one",
        max_n=1000,
    )
    template.set_protocol(protocol)

    # Negative trajectory should not reject
    batch = [
        BinomialArmData(n=500, success=250, arm="control"),  # 0.5
        BinomialArmData(n=500, success=200, arm="treatment"),  # 0.4
    ]
    template.update(batch)
    report = template.report_progress()
    assert report["trajectory"] < 0
    assert report["status"] == DecisionStatus.CONTINUE_
    assert report["boundary"] > 0  # Boundary is always positive CI width

    # If using two-sided, -0.1 might cross if boundary is small enough (<0.1)
