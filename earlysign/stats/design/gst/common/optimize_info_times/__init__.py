"""Information time optimization strategies for group sequential designs.

This package provides different methods for optimizing information times
in group sequential trial designs:

- nelder_mead: Nelder-Mead optimization with parallel restarts
- dirichlet_random: Random sampling on Dirichlet simplex

Both methods aim to minimize expected sample size under H1 while maintaining
statistical properties (type I error, power).
"""

from earlysign.stats.design.gst.common.optimize_info_times.nelder_mead import (
    BalancedDesign,
    DesignObjective,
    DesignOptimizer,
    MaximizePower,
    MinimizeASN,
    NormalMeansTwoArmEngine,
    SpendingStrategy,
    get_design_characteristics,
    get_optimal_information_rates,
)

__all__ = [
    "BalancedDesign",
    "DesignObjective",
    "DesignOptimizer",
    "MaximizePower",
    "MinimizeASN",
    "NormalMeansTwoArmEngine",
    "SpendingStrategy",
    "get_design_characteristics",
    "get_optimal_information_rates",
]
