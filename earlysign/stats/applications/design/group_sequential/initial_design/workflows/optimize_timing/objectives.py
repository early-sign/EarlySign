"""
Optimization objectives for group sequential timing workflows.
"""

from abc import ABC, abstractmethod
from typing import Any

from earlysign.stats.design.gst.common.config import DesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab


class DesignObjective(ABC):
    """Abstract base class for group sequential design optimization objectives."""

    @abstractmethod
    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        """Return scalar objective (lower is better)."""

    @abstractmethod
    def get_constraints(self, spec: DesignSpec) -> dict[str, Any]:
        """Return constraint information for reporting."""


class MinimizeASN(DesignObjective):
    """Objective that minimizes expected sample size subject to constraints."""

    def __init__(self, planned_max_n: int, target_power: float = 0.90):
        self.planned_max_n = planned_max_n
        self.target_power = target_power

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        try:
            lab.compute_boundaries()
            lab.run_simulations()
            assert lab.simulation_results is not None

            power = lab.simulation_results["power"]
            if power < self.target_power:
                return float(
                    self.planned_max_n * 10 + (self.target_power - power) * 10000
                )

            max_sample = lab.simulation_results["max_sample_size"]
            if max_sample > self.planned_max_n:
                return float(
                    self.planned_max_n * 10 + (max_sample - self.planned_max_n) * 100
                )

            return float(lab.simulation_results["expected_sample_size"])
        except Exception:
            return float(self.planned_max_n * 100)

    def get_constraints(self, spec: DesignSpec) -> dict[str, Any]:
        return {"planned_max_n": self.planned_max_n, "target_power": self.target_power}


class MaximizePower(DesignObjective):
    """Objective that maximizes power under a maximum sample size constraint."""

    def __init__(self, planned_max_n: int):
        self.planned_max_n = planned_max_n

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        try:
            lab.compute_boundaries()
            lab.run_simulations()
            assert lab.simulation_results is not None

            max_sample = lab.simulation_results["max_sample_size"]
            if max_sample > self.planned_max_n:
                return 1.0

            power = lab.simulation_results["power"]
            return float(-power)
        except Exception:
            return 1.0

    def get_constraints(self, spec: DesignSpec) -> dict[str, Any]:
        return {"planned_max_n": self.planned_max_n}


class BalancedDesign(DesignObjective):
    """Composite objective balancing power deviation and expected sample size."""

    def __init__(
        self,
        *,
        power_weight: float = 1.0,
        sample_weight: float = 1.0,
        target_power: float = 0.90,
        target_n: int = 3000,
    ):
        self.power_weight = power_weight
        self.sample_weight = sample_weight
        self.target_power = target_power
        self.target_n = target_n

    def evaluate(self, spec: DesignSpec, lab: DesignLab) -> float:
        try:
            lab.compute_boundaries()
            lab.run_simulations()
            assert lab.simulation_results is not None

            power = lab.simulation_results["power"]
            ess = lab.simulation_results["expected_sample_size"]

            power_score = abs(power - self.target_power) / self.target_power
            sample_score = abs(ess - self.target_n) / self.target_n

            return float(
                self.power_weight * power_score + self.sample_weight * sample_score
            )
        except Exception:
            return 100.0

    def get_constraints(self, spec: DesignSpec) -> dict[str, Any]:
        return {
            "target_power": self.target_power,
            "target_n": self.target_n,
            "power_weight": self.power_weight,
            "sample_weight": self.sample_weight,
        }
