"""
Verification Script: - Trace Verification.
Checks if the implicit trace accumulation and CommitCallResult lineage are working.
"""

import ibis
from pydantic import BaseModel
from earlysign.core.ledger import Ledger
from earlysign.v1.framework.session import Session
from earlysign.v1.framework.write_models import WriteModel
from earlysign.v1.framework.trace import Traced, TraceHash
from earlysign.v1.framework.projector import Projector, ProjectionResult


class Fact(BaseModel):
    val: int


class Result(BaseModel):
    total: int


class SimpleProjector(Projector[Fact]):
    def project(self, table: ibis.Expr) -> ProjectionResult[Fact]:
        # Filter for Fact events
        facts = table.filter(table.payload_type == "Fact").execute()
        total = facts["payload"].apply(lambda x: x["val"]).sum()
        # Trace is the list of trace_hashes from labels
        hashes = (
            table.filter(table.payload_type == "Fact")
            .labels["trace_hash"]
            .execute()
            .tolist()
        )
        return ProjectionResult(
            data=Fact(val=total), trace=[TraceHash(h) for h in hashes]
        )


def main():
    print("=== Trace Verification Script ===")
    con = ibis.duckdb.connect(":memory:")
    ledger = Ledger(con, "events")
    ledger.ensure()

    # 1. Ingest
    with Session(ledger) as sess:
        h1 = WriteModel.Commit(sess, Fact(val=10), trace=[TraceHash("evt_1")])
        h2 = WriteModel.Commit(sess, Fact(val=20), trace=[TraceHash("evt_2")])

    print(f"-> Ingested facts with hashes: {h1}, {h2}")

    # 2. Read
    with Session(ledger) as sess:
        traced_fact = sess.Read(SimpleProjector())
        print(
            f"-> Read traced data: {traced_fact.data.val}, trace: {traced_fact.trace}"
        )

        # 3. Compute
        def double(f: Fact):
            return {"total": f.val * 2}

        res = WriteModel.CommitCallResult(sess, Result, double, traced_fact)
        print(f"-> Computed result: {res.total}")

        # Verify result's trace in ledger
        df = ledger.t.execute()
        res_row = df[df["payload_type"] == "Result.Result"].iloc[0]
        res_trace_hash = res_row["labels"]["trace_hash"]
        print(f"-> Result trace hash in ledger: {res_trace_hash}")

    print("=== Trace Verification Complete ===")


if __name__ == "__main__":
    main()
