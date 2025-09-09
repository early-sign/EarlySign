"""
earlysign.runtime
=================

Runtime environment for executing EarlySign experiments.

This namespace contains the execution infrastructure that runs experiment modules,
including base classes, runners, and execution strategies.

Key Components
--------------
- `ExperimentModule`: Base class for all experiment definitions
- `AnalysisResult`: Standard result container for experiment outcomes
- `SequentialRunner`: Sequential execution of experiment workflows
- `BatchRunner`: Batch processing capabilities

Examples
--------
>>> # Import runtime components
>>> from earlysign.runtime.experiment_module import ExperimentModule, AnalysisResult
>>> from earlysign.runtime.runners import SequentialRunner
>>> from earlysign.backends.polars.ledger import PolarsLedger
>>>
>>> # Create custom module (implementation needed)
>>> # runner = SequentialRunner(custom_module, PolarsLedger())
>>> # result = runner.analyze()
"""
