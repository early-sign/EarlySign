"""
Scheme hooks shared by group-sequential design helpers.

The adapters defined here capture scheme-specific callbacks (e.g.,
fixed-sample planners and simulator builders) so scenario code can remain
agnostic to the underlying statistical family.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from earlysign.stats.applications.design.group_sequential.initial_design.helpers.protocols import (
    ProcedureFactory,
)
from earlysign.stats.essentials.methods.group_sequential import simulation
from earlysign.stats.essentials.methods.group_sequential.asn import ASNCalculator
from earlysign.stats.essentials.methods.group_sequential.operating_characteristics import (
    Simulator,
)
from earlysign.stats.essentials.methods.group_sequential.spending import (
    SpendingFunction,
)

FSDPlanner = Callable[[], Dict[str, int]]
SimulatorFactory = Callable[[int, float], Simulator]
SimulatorKwargsBuilder = Callable[
    [float, int, simulation.SamplingStrategy], Dict[str, object]
]
ASNFactoryBuilder = Callable[[SpendingFunction], Callable[[], ASNCalculator]]
ProcedureFactoryBuilder = Callable[[SpendingFunction, float], ProcedureFactory]


@dataclass(frozen=True)
class GSTSchemeHooks:
    """Scheme-specific hooks consumed by :class:`AddInterimToFixedSampleTest`."""

    name: str
    target_effect: float
    effect_sizes: Sequence[float]
    null_reference: Optional[float]
    fsd_planner: FSDPlanner
    simulator_factory: SimulatorFactory
    simulator_kwargs_builder: SimulatorKwargsBuilder
    asn_factory_builder: ASNFactoryBuilder
    procedure_factory_builder: ProcedureFactoryBuilder

    def with_effect_sizes(
        self, overrides: Optional[Sequence[float]]
    ) -> "GSTSchemeHooks":
        if overrides is None:
            return self
        return GSTSchemeHooks(
            name=self.name,
            target_effect=self.target_effect,
            effect_sizes=[float(x) for x in overrides],
            null_reference=self.null_reference,
            fsd_planner=self.fsd_planner,
            simulator_factory=self.simulator_factory,
            simulator_kwargs_builder=self.simulator_kwargs_builder,
            asn_factory_builder=self.asn_factory_builder,
            procedure_factory_builder=self.procedure_factory_builder,
        )
