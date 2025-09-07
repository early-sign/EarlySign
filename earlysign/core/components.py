"""
earlysign.core.components
=========================

Thin base classes for components that *write to* and *read from* the ledger.

They only centralize **namespace/tag conventions** so all concrete implementations
stay consistent. Concrete classes must implement `step()`.

Doctest (smoke):
>>> from earlysign.core.traits import LedgerOps
>>> from earlysign.core.ledger import Row, LedgerReader
>>> from earlysign.core.names import Namespace
>>> class Host(LedgerOps):
...   def __init__(self): self.rows=[]
...   def append(self, **kw):
...     self.rows.append(Row(uuid="u", time_index=kw["time_index"], ts=kw["ts"],
...       namespace=str(kw["namespace"]), kind=kw["kind"], entity=kw["entity"],
...       snapshot_id=kw["snapshot_id"], tag=kw.get("tag"),
...       payload_type=kw["payload_type"], payload=kw["payload"]))
...     return self
...   def emit_signal(self, **kw):
...     return self.append(time_index=kw["time_index"], ts=kw["ts"],
...       namespace=kw.get("namespace", Namespace.SIGNALS.value), kind=kw.get("kind","emitted"),
...       entity=kw["entity"], snapshot_id=kw["snapshot_id"],
...       payload_type="Signal", payload={"topic": kw["topic"], "body": kw["body"]},
...       tag=kw.get("tag","signal"))
...   class R(LedgerReader):
...     def __init__(self, host): self.h=host
...     def iter_rows(self, **f): return iter(self.h.rows)
...     def latest(self, **f): return self.h.rows[-1] if self.h.rows else None
...     def count(self, **f): return len(self.h.rows)
...   def reader(self): return Host.R(self)
>>> class DummyStat(Statistic):
...   def step(self, ledger, experiment_id, step_key, time_index):
...     ledger.write_event(time_index=time_index, namespace=self.ns_stats, kind="updated",
...                        experiment_id=experiment_id, step_key=step_key,
...                        payload_type="Stat", payload={"v":1}, tag=self.tag_stats)
>>> host = Host(); DummyStat().step(host, "exp#1", "s1", "t1"); host.reader().count()
1
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

from earlysign.core.names import Namespace
from earlysign.core.ledger import NamespaceLike


@dataclass(kw_only=True)
class Statistic:
    """Base class for statistics updaters."""
    ns_stats: NamespaceLike = Namespace.STATS
    tag_stats: Optional[str] = "stat:generic"
    def step(self, ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        raise NotImplementedError

@dataclass(kw_only=True)
class Criteria:
    """Base class for criteria/boundary updaters."""
    ns_crit: NamespaceLike = Namespace.CRITERIA
    tag_crit: Optional[str] = "crit:generic"
    def step(self, ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        raise NotImplementedError

@dataclass(kw_only=True)
class Signaler:
    """Base class for signal emitters (e.g., stop/continue)."""
    ns_sig: NamespaceLike = Namespace.SIGNALS
    def step(self, ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        raise NotImplementedError

@dataclass(kw_only=True)
class Recommender:
    """Base class for recommendation emitters (optional)."""
    ns_sig: NamespaceLike = Namespace.SIGNALS
    def step(self, ledger, experiment_id: str, step_key: str, time_index: str) -> None:
        raise NotImplementedError
