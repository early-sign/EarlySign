"""
earlysign.runtime
=================

Runtime environment for executing EarlySign experiments.

This namespace contains the execution infrastructure that runs experiment templates,
including base classes, runners, and execution strategies.

Key Components
--------------
- `ExperimentTemplate`: Base class for all experiment definitions
- `AnalysisResult`: Standard result container for experiment outcomes
- `SequentialRunner`: Sequential execution of experiment workflows
- `BatchRunner`: Batch processing capabilities

Examples
--------
>>> # Import runtime components
>>> import ibis
>>> from earlysign.core.ledger import Ledger
>>> from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult
>>> from earlysign.runtime.runners import SequentialRunner
>>>
>>> # Create custom template (implementation needed)
>>> # conn = ibis.duckdb.connect(":memory:")
>>> # ledger = Ledger(conn, "test")
>>> # runner = SequentialRunner(custom_template, ledger)
>>> # result = runner.analyze()
"""
