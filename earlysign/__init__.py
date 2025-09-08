"""
earlysign — a package for building and operating sequential statistical procedures.

Many flavors of sequential testing exist—group sequential, fully sequential
(anytime-valid), Bayesian—and they often phrase problems differently. What
they all share is a timeline of events. earlysign centers on that commonality
by using a *typed, append-only ledger*.
In other words, we adopt the event-sourcing pattern to design this library.

Every fact—observations, plans, method-specific info updates, signals,
decisions, notes, TODOs, cache (intermediate stats), and runtime lifecycle—
is appended as first-class data. Statistics are *pull-based*: they update
themselves exclusively by reading the ledger through an injected, read-only,
typed query interface. As a result, every statistical procedure both consumes
and writes to a single ledger that acts as the experiment’s sole source of truth.
This yields strict reproducibility without sacrificing flexibility.

Multiple runtimes can coexist and operate over shared namespaces; each may emit
its own signals and register action recommendations (TODOs). Whether you accept
or ignore those recommendations, the outcome is captured in the ledger. All
lifecycle events (start/stop) are likewise recorded, supporting end-to-end
auditability and replay.

Example
-------
>>> import earlysign
>>> assert hasattr(earlysign, "core")
>>> assert hasattr(earlysign, "stats")
"""
