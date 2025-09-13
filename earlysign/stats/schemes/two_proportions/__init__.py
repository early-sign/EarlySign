"""
Two-proportion testing schemes for A/B experiments.

This module provides complete implementations for comparing two binomial
proportions using different statistical methodologies with event-sourcing
architecture for regulatory compliance and reproducible research.

**Statistical Methods Supported:**
- Group Sequential Testing (GST) with error spending functions
- Safe Testing using beta-binomial e-processes for anytime-valid inference
- Complete audit trails and immutable event histories

**Module Organization:**

Core Infrastructure
-------------------
- `core`: Data models, observation components, and aggregation utilities
- `statistics`: All statistical computation components (GST and Safe Testing)
- `experiments`: High-level experiment templates and orchestration

Example Usage
-------------
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.stats.schemes.two_proportions.experiments import TwoPropGSTTemplate
>>> from earlysign.stats.schemes.two_proportions.common import ObservationBatch
>>>
>>> # Create GST experiment
>>> experiment = TwoPropGSTTemplate("my_ab_test", alpha_total=0.05, looks=4)
>>> conn = ibis.duckdb.connect(":memory:")
>>> ledger = Ledger(conn, "test")
>>> experiment.setup(ledger)
>>>
>>> # Create observation batch
>>> batch = ObservationBatch()
>>> batch.add_group_a_observations(successes=45, total=100)
>>> batch.add_group_b_observations(successes=52, total=100)
>>>
>>> # Validate batch
>>> batch.validate()
True

Safe Testing Example
--------------------
>>> from earlysign.stats.schemes.two_proportions.experiments import TwoPropSafeTemplate
>>>
>>> # Create Safe Testing experiment
>>> safe_exp = TwoPropSafeTemplate("safe_test", alpha_total=0.05)
>>> isinstance(safe_exp.experiment_id, str)
True

Component-Level Usage
--------------------
>>> from earlysign.stats.schemes.two_proportions.group_sequential import WaldZStatistic
>>> from earlysign.stats.schemes.two_proportions.common import TwoPropObservation
>>>
>>> # Individual components for custom workflows
>>> observation = TwoPropObservation()
>>> statistic = WaldZStatistic()
>>> isinstance(observation, TwoPropObservation)
True

Design Principles
-----------------
- **Event-sourcing**: All state changes captured as immutable events
- **Separation of concerns**: Statistics compute, users decide
- **Regulatory compliance**: Complete audit trails and reproducibility
- **Framework agnostic**: Core concepts portable across languages
"""
