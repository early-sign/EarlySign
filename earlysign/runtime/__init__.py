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
>>> from earlysign.runtime.experiment_template import ExperimentTemplate, AnalysisResult
>>> from earlysign.runtime.runners import SequentialRunner
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>>
>>> # Create custom template (implementation needed)
>>> # runner = SequentialRunner(custom_template, PolarsLedger())
>>> # result = runner.analyze()
"""
