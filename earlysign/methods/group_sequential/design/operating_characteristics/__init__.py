"""Operating Characteristics Evaluation Package.

This package contains tools for evaluating the performance (Operating Characteristics)
of Group Sequential Designs.

Submodules:
    engines: Core evaluation classes (Monte Carlo, Numerical).
    binomial: Domain-specific evaluators (e.g., BinomialEvaluator).
"""

from earlysign.methods.group_sequential.design.operating_characteristics.binomial import (
    BinomialOperatingCharacteristicsEvaluator,
)
from earlysign.methods.group_sequential.design.operating_characteristics.continuous import (
    ContinuousOperatingCharacteristicsEvaluator,
)
from earlysign.methods.group_sequential.design.operating_characteristics.engines import (
    AsymptoticSimulator,
    EvaluationResult,
    NumericalCalculator,
    OperatingCharacteristicsEvaluator,
    ProtocolEvaluator,
    SimulationCurve,
)

__all__ = [
    "BinomialOperatingCharacteristicsEvaluator",
    "ContinuousOperatingCharacteristicsEvaluator",
    "EvaluationResult",
    "AsymptoticSimulator",
    "NumericalCalculator",
    "OperatingCharacteristicsEvaluator",
    "ProtocolEvaluator",
    "SimulationCurve",
]
