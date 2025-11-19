"""
High-level optimization routines that orchestrate timing workflows.
"""

import copy
from typing import Any, Tuple

import numpy as np

from earlysign.integration.design.group_sequential.initial_design import (
    schema as design_schema,
)
from earlysign.integration.design.group_sequential.initial_design.workflows.optimize_timing.minimize_asn import (
    MinimizeASNOptimizer,
)
from earlysign.integration.design.group_sequential.initial_design.workflows.optimize_timing.objectives import (
    DesignObjective,
)
from earlysign.stats.methods.group_sequential import spending as spending_mod
from earlysign.stats.schemes.two_means.asn import NormalMeansASNCalculator

DesignSpec = design_schema.DesignSpec
InformationSpacing = design_schema.InformationSpacing
SpendingStrategy = spending_mod.SpendingFunction


def _spec_to_spending_strategy(spec: DesignSpec) -> SpendingStrategy:
    spending_type = spec.boundary.spending_function
    alpha = spec.test.alpha
    sided = 2 if spec.test.sided == "two" else 1

    if spending_type == design_schema.SpendingFunction.POCOCK:
        return spending_mod.PocockSpending(alpha=alpha)
    if spending_type == design_schema.SpendingFunction.HSD:
        return spending_mod.HSDSpending(alpha=alpha, gamma=spec.boundary.hsd_gamma)
    return spending_mod.OBFSpending(alpha=alpha, sided=sided)


class DesignOptimizer:
    """High-level optimization engine used by UI and API layers."""

    def __init__(self, base_spec: DesignSpec, objective: DesignObjective):
        self.base_spec = base_spec
        self.objective = objective
        self.optimization_history: list[dict[str, Any]] = []

    def _clone_spec(self) -> DesignSpec:
        return copy.deepcopy(self.base_spec)

    def optimize_info_times(self, n_analyses: int, n_samples: int = 100) -> np.ndarray:
        spec = self.base_spec

        alpha = spec.test.alpha
        beta = 1.0 - spec.test.power
        sided = 2 if spec.test.sided == "two" else 1
        allocation_ratio = getattr(spec.allocation, "alloc_ratio", 1.0) or 1.0

        alternative = 0.2
        st_dev = 1.0
        effect = getattr(spec, "effect", None)
        if effect is not None:
            effect_size = getattr(effect, "effect_size", None)
            if effect_size is not None:
                alternative = float(effect_size)
            elif hasattr(effect, "hazard_ratio"):
                alternative = float(np.log(getattr(effect, "hazard_ratio")))

            if hasattr(effect, "p_control"):
                p_control = float(getattr(effect, "p_control"))
                if hasattr(effect, "get_treatment_proportion"):
                    p_treatment = float(effect.get_treatment_proportion())
                    alternative = p_treatment - p_control
                else:
                    p_treatment = p_control + alternative
                p_pooled = 0.5 * (p_control + p_treatment)
                st_dev = float(np.sqrt(max(p_pooled * (1 - p_pooled), 1e-9)))
            elif hasattr(effect, "std_dev"):
                st_dev = float(getattr(effect, "std_dev"))

        spending = _spec_to_spending_strategy(spec)

        try:
            # Build an ASN calculator instance and inject it into the optimizer.
            calculator = NormalMeansASNCalculator(
                alpha=alpha,
                beta=beta,
                sided=sided,
                alternative=alternative,
                st_dev=st_dev,
                allocation_ratio=allocation_ratio,
                spending=spending,
            )

            optimizer = MinimizeASNOptimizer(
                calculator=calculator,
                k_max=n_analyses,
                min_gap=0.02,
                seed=42,
                n_jobs=1,
                n_restarts=12,
                restart_scale=0.2,
            )
            info_rates = optimizer.minimize()
            if not info_rates:
                return np.linspace(0, 1, n_analyses + 1)[1:]
            return np.array(info_rates, dtype=float)
        except Exception:
            return np.linspace(0, 1, n_analyses + 1)[1:]

    def optimize_info_times_and_n_analyses(
        self, max_k: int, min_k: int = 2, n_samples: int = 20
    ) -> Tuple[int, np.ndarray]:
        best_score = float("inf")
        best_k = min_k
        best_times = np.linspace(0, 1, min_k + 1)[1:]

        for k in range(min_k, max_k + 1):
            try:
                info_times = self.optimize_info_times(k)
                spec = self._clone_spec()
                spec.sequential.n_analyses = k
                spec.sequential.info_times = info_times.tolist()
                spec.sequential.info_spacing = InformationSpacing.CUSTOM
                score = self.objective.evaluate(spec)
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_times = info_times
            except Exception:
                continue

        return best_k, best_times

    def optimize_n_analyses(self, min_k: int = 2, max_k: int = 10) -> int:
        best_k, _ = self.optimize_info_times_and_n_analyses(max_k=max_k, min_k=min_k)
        return best_k

    def optimize_comprehensive(self) -> DesignSpec:
        spec = self._clone_spec()
        optimal_k = self.optimize_n_analyses()
        spec.sequential.n_analyses = optimal_k
        optimal_times = self.optimize_info_times(optimal_k)
        spec.sequential.info_times = optimal_times.tolist()
        spec.sequential.info_spacing = InformationSpacing.CUSTOM
        return spec

    def get_optimization_summary(self) -> Any:
        import pandas as pd

        if not self.optimization_history:
            return pd.DataFrame()
        return pd.DataFrame(self.optimization_history)
