"""
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.framework.scoped import ScopedLedger
>>> from earlysign.stats.schemes.two_binomials.records import BinomialCountsRecord
>>> from earlysign.stats.schemes.two_binomials.operators import WaldZStatistic
>>> from earlysign.stats.common.operators import InformationTime, GSTBoundary, Decision
>>> import ibis

# 1) Setup Ledger (no schema import) and ensure table exists
>>> con = ibis.duckdb.connect(":memory:")
>>> base = Ledger().set_connector(con).use_default_table("events")
>>> base.ensure()  # create base table via strategy if missing
>>> scoped = ScopedLedger(connector=base.connector, table_name=base.table_name).bind(experiment_id="exp_demo")

# 2) Input record family (counts) — attach to scoped ledger
>>> counts = BinomialCountsRecord(id="exp_demo:counts").attach(scoped)

# 3) Write snapshots (binomial)
>>> _ = counts.insert({"nA": 100, "mA": 22, "nB": 95, "mB": 30})
>>> _ = counts.insert({"nA": 150, "mA": 35, "nB": 140, "mB": 48})

# 4) Compute Wald Z
>>> wz = WaldZStatistic(counts=counts)
>>> (wald_rec,) = wz.materialize_outputs(scoped)
>>> res_z = wz.run(wald_rec)
>>> "z" in res_z and "se" in res_z
True

# 5) Info time
>>> it = InformationTime(counts=counts, max_sample_size=600)
>>> (it_rec,) = it.materialize_outputs(scoped)
>>> res_it = it.run(it_rec)
>>> 0.0 < res_it["fraction"] <= 1.0
True

# 6) Boundary
>>> gst = GSTBoundary(info=it_rec, alpha=0.05, style="obf")
>>> (bnd_rec,) = gst.materialize_outputs(scoped)
>>> res_bnd = gst.run(bnd_rec)
>>> "upper" in res_bnd and res_bnd["upper"] >= 0.0
True

# 7) Decision
>>> dec = Decision(zstat=wald_rec, boundary=bnd_rec)
>>> (sig_rec,) = dec.materialize_outputs(scoped)
>>> res_sig = dec.run(sig_rec)
>>> res_sig["decision"] in ("continue", "stop_success", "no_data")
True

# 8) Query helpers from QueryMixin (no execution in helpers)
>>> q = wald_rec.latest()
>>> isinstance(q, ibis.expr.types.Table)
True
"""
