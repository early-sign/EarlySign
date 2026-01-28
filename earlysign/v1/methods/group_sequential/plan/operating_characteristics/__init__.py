"""Operating Characteristics Evaluation Package.

This package contains tools for evaluating the performance (Operating Characteristics)
of Group Sequential Designs.

Submodules:
    engines: Core evaluation classes (Monte Carlo, Numerical).
    binomial: Domain-specific evaluators (e.g., BinomialEvaluator).
"""

from earlysign.v1.methods.group_sequential.plan.operating_characteristics.binomial import (
    BinomialABOperatingCharacteristicsEvaluator,
)
from earlysign.v1.methods.group_sequential.plan.operating_characteristics.engines import (
    AsymptoticSimulator,
    EvaluationResult,
    NumericalCalculator,
    OperatingCharacteristicsEvaluator,
    SimulationCurve,
)

__all__ = [
    "BinomialABOperatingCharacteristicsEvaluator",
    "EvaluationResult",
    "AsymptoticSimulator",
    "NumericalCalculator",
    "OperatingCharacteristicsEvaluator",
    "SimulationCurve",
]
