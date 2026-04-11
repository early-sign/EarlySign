import numpy as np

from earlysign.builtin.group_sequential.adapters import binomial, continuous
from earlysign.builtin.group_sequential.schema.hypotheses import (
    BinaryEffectSize,
    ContinuousEffectSize,
    SurvivalEffectSize,
)
from earlysign.builtin.group_sequential.schema.protocol import Protocol
from earlysign.builtin.group_sequential.schema.timers import (
    EquidistantSchedule,
    FixedSchedule,
    ScheduleSpec,
)


def get_standardized_drift(protocol: Protocol) -> float:
    """
    Dispatches to domain-specific drift calculations.
    """
    task = protocol.task
    effect = task.hypotheses.target_effect

    if isinstance(effect, BinaryEffectSize):
        props = list(effect.proportions.values())
        if len(props) < 2:
            raise ValueError("BinaryEffectSize must define at least 2 arm proportions.")
        return binomial.get_standardized_drift(props[0], props[1])

    elif isinstance(effect, ContinuousEffectSize):
        means = list(effect.means.values())
        if len(means) < 2:
            raise ValueError("ContinuousEffectSize must define at least 2 arm means.")
        delta = abs(means[0] - means[1])
        return continuous.get_standardized_drift(
            delta, effect.standard_deviation, n_arms=len(means)
        )

    elif isinstance(effect, SurvivalEffectSize):
        # theta = |log(HR)| / 2
        hrs = list(effect.hazard_ratios.values())
        if len(hrs) < 2:
            raise ValueError(
                "SurvivalEffectSize must define at least 2 arm hazard ratios."
            )
        hr = hrs[1] / hrs[0] if hrs[0] != 0 else 1.0
        return float(abs(np.log(hr)) / 2.0)

    raise ValueError(f"Unsupported effect size type: {type(effect)}")


def get_info_times(schedule: ScheduleSpec) -> np.ndarray:
    """
    Returns the array of information fractions (0 < t <= 1) defined by the Schedule.
    """
    if isinstance(schedule, FixedSchedule):
        return np.array(schedule.analyses)
    elif isinstance(schedule, EquidistantSchedule):
        return np.linspace(1.0 / schedule.n_looks, 1.0, schedule.n_looks)
    return np.array([1.0])


def get_final_efficacy_boundary(protocol: Protocol) -> float:
    """
    Calculates the efficacy boundary (Z-scale) at the final analysis (t=1.0).
    """
    from earlysign.builtin.group_sequential.core.model import CanonicalJointModel

    spec = protocol.method.stopping_policy
    schedule = spec.schedule
    info_times = get_info_times(schedule)

    # Ensure 1.0 is in there for the "final" look
    if not np.any(np.isclose(info_times, 1.0)):
        info_times = np.append(info_times, 1.0)
    info_times.sort()

    model = CanonicalJointModel.from_spec(protocol)

    try:
        drift = get_standardized_drift(protocol)
    except ValueError:
        drift = 0.0

    # Solve for all boundaries using the model's policy
    eff_bounds, _ = model.solve_boundaries(drift=drift)

    if eff_bounds is not None and len(eff_bounds) > 0:
        return float(eff_bounds[-1])

    raise ValueError("Failed to calculate final efficacy boundary for protocol.")
