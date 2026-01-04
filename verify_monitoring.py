import ibis
from earlysign.core.ledger import Ledger
from earlysign.v1.templates.binomial_monitoring import BinomialMonitoringTemplate
from earlysign.v1.methods.anytime_valid.protocol import EProcessProtocol
from earlysign.v1.methods.binomial import BatchObservation
from earlysign.v1.methods.actions import Ingest
from earlysign.v1.framework.session import Session


def main():
    print("=== Binomial Monitoring Verification ===")
    con = ibis.duckdb.connect(":memory:")
    ledger = Ledger(con, "events")
    ledger.ensure()

    template = BinomialMonitoringTemplate(ledger)

    # 1. Set Protocol (null_p=0.5, alt_p=0.6)
    protocol = EProcessProtocol(null_p=0.5, alt_p=0.6, alpha=0.05)
    template.set_protocol(protocol)
    print("-> Protocol Set: null_p=0.5, alt_p=0.6")

    # 2. Ingest some evidence
    with Session(ledger) as sess:
        Ingest(sess, BatchObservation(n=100, success=65, arm="C"))
    print("-> Ingested: N=100, Successes=65")

    # 3. Check Monitoring
    res = template.report_progress()
    print(
        f"-> Monitoring Result: E-Value={res['e_value']:.4f}, Rejected={res['is_rejected']}"
    )

    # 4. Verify that no snapshot was created (identity="monitoring_summary")
    df = ledger.t.execute()
    snapshots = df[df["payload_type"] == "Snapshot"]
    if snapshots.empty:
        print("-> [SUCCESS] No snapshot created during check.")
    else:
        print(f"-> [FAILURE] Found {len(snapshots)} snapshots in ledger.")

    print("=== Verification Complete ===")


if __name__ == "__main__":
    main()
