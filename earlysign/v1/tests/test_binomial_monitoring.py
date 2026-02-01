import ibis

from earlysign.core.ledger import Ledger
from earlysign.schema.ES3.Binomial import ArmData
from earlysign.v1.methods.AVI.engines.binomial_e_value import EProcessProtocol
from earlysign.v1.templates.binomial_monitoring import BinomialMonitoringTemplate


def test_binomial_monitoring_template_e2e():
    """
    End-to-End test of the BinomialMonitoringTemplate after migration.
    """
    conn = ibis.connect("duckdb://:memory:")
    ledger = Ledger(conn, "events")
    ledger.ensure()
    ledger = ledger.bind(experiment_id="test_binom_mon_001")

    # Design
    # H0: p=0.5, H1: p=0.7, Alpha=0.05
    protocol = EProcessProtocol(null_p=0.5, alt_p=0.7, alpha=0.05)

    template = BinomialMonitoringTemplate(ledger)
    template.set_protocol(protocol)

    # 1. Update with H0-like data
    batch1 = ArmData(n=100, success=50, arm="control")
    template.update([batch1])

    report1 = template.report_progress()
    assert report1["sample_n"] == 100
    # successes is not in ProgressReport directly, it's in arms metrics
    assert report1["arms"]["control"]["successes"] == 50
    assert report1["status"] == "continue"

    # 2. Update with H1-like data to cross threshold
    batch2 = ArmData(n=900, success=650, arm="control")
    template.update([batch2])

    report2 = template.report_progress()
    assert report2["sample_n"] == 1000
    assert report2["arms"]["control"]["successes"] == 700
    assert report2["status"] == "stop_efficacy"

    final_report = template.report_result()
    assert final_report["is_rejected"] is True
    assert final_report["final_status"] == "stop_efficacy"
