import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.execution.stopping_policy import (
    StoppingPolicyFactory,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)


def get_standardized_drift(protocol: GST.Protocol) -> float:
    """
    Extracts the assumed standardized effect size (drift) from the protocol.
    Standardized drift (theta) is defined such that E[Z_t] = theta * sqrt(t * I_max).
    At t=1.0, E[Z_1] = drift = theta * sqrt(I_max).
    """
    task = protocol.task
    effect = task.hypotheses.target_effect

    if isinstance(effect, GST.BinaryEffectSize):
        props = list(effect.proportions.values())
        if len(props) < 2:
            return 0.0
        p1, p2 = props[0], props[1]
        delta = abs(p1 - p2)
        p_bar = (p1 + p2) / 2.0
        sigma2 = p_bar * (1.0 - p_bar)
        # For a two-arm binomial test with equal allocation:
        # theta = delta / sqrt(4 * sigma2)
        if sigma2 <= 0:
            return 0.0
        return float(delta / np.sqrt(4.0 * sigma2))

    elif isinstance(effect, GST.ContinuousEffectSize):
        means = list(effect.means.values())
        if len(means) < 2:
            return 0.0
        delta = abs(means[0] - means[1])
        sigma = effect.standard_deviation
        # theta = delta / (2 * sigma)
        if sigma <= 0:
            return 0.0
        return float(delta / (2.0 * sigma))

    elif isinstance(effect, GST.SurvivalEffectSize):
        # theta = |log(HR)| / 2
        hrs = list(effect.hazard_ratios.values())
        if len(hrs) < 2:
            return 0.0
        hr = hrs[1] / hrs[0] if hrs[0] != 0 else 1.0
        return float(abs(np.log(hr)) / 2.0)

    return 0.0


def get_final_efficacy_boundary(protocol: GST.Protocol) -> float:
    """
    Calculates the efficacy boundary (Z-scale) at the final analysis (t=1.0).
    """
    spec = protocol.method.stopping_policy
    schedule = spec.schedule

    if isinstance(schedule, GST.FixedSchedule):
        info_times = np.array(schedule.analyses)
    elif isinstance(schedule, GST.EquidistantSchedule):
        info_times = np.linspace(1.0 / schedule.n_looks, 1.0, schedule.n_looks)
    else:
        info_times = np.array([1.0])

    # Ensure 1.0 is in there for the "final" look
    if not np.any(np.isclose(info_times, 1.0)):
        info_times = np.append(info_times, 1.0)
    info_times.sort()

    model = CanonicalJointModel(Config(info_times=info_times))
    policy = StoppingPolicyFactory.build_from_spec(spec)

    # Solve for all boundaries
    eff_bounds, _ = model.solve_boundaries_from_policy(policy)

    if eff_bounds is not None and len(eff_bounds) > 0:
        return float(eff_bounds[-1])

    return 1.96  # Fallback
