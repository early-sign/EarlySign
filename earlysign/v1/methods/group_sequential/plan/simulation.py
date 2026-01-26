"""
Operating Characteristics Simulation
====================================

Provides simulation engines for evaluating Group Sequential Designs.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
)
from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess


def calculate_stopping_probabilities(
    info_times: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    drift: float,
    seed: Optional[int] = 42,
    method: str = "numerical_integration",
    n_sims: int = 10000,
) -> Dict[str, np.ndarray]:
    """
    Calculates the stage-wise stopping probabilities.

    Returns:
        Dict with:
        - "total": Array of total stopping probs at each look.
        - "upper": Array of upper boundary crossing probs (approx power contribution).
        - "lower": Array of lower boundary crossing probs.
    """
    k = len(info_times)

    # Note: calculate_stopping_probabilities currently returns total stopping probability
    # (crossing either upper or lower bound) for ASN calculation.
    # Directed probabilities (e.g. specifically crossing upper) are not separated here.

    cum_stop_probs = np.zeros(k)
    np.zeros(k)  # Approximation if possible

    gp = CanonicalGaussianProcess(drift=drift, rng=np.random.default_rng(seed))

    for i in range(k):
        t_sub = info_times[: i + 1]
        u_sub = upper[: i + 1] if upper is not None else None
        l_sub = lower[: i + 1] if lower is not None else None

        cp = gp.compute_crossing_probability(
            t=t_sub,
            upper=u_sub,
            lower=l_sub,
            method=method,
            n_sims=n_sims,
        )
        cum_stop_probs[i] = cp

        # If we want power, we need P(cross Upper).
        # We can compute P(cross Upper only) by setting lower=None.
        # Ignores lower binding logic for this simplified check
        # Actually, usually we care about "Reject H0" = Cross Upper.
        # If binding futility exists, "Cross Upper" implies "Didn't cross Lower before".

        if u_sub is not None:
            # Power calculation would require specific upper-crossing probabilities.
            # For this visualization-focused implementation, we skip precise power tracking.
            pass

    probs = np.zeros(k)
    probs[0] = cum_stop_probs[0]
    probs[1:] = np.diff(cum_stop_probs)

    # Handle forced stopping at last look
    current_total = cum_stop_probs[-1]
    if current_total < 1.0:
        probs[-1] += 1.0 - current_total

    return {"probs": probs, "cum_total": cum_stop_probs}


@dataclass
class OperatingCharacteristicsResult:
    """Container for simulation results."""

    x_values: np.ndarray  # The effect sizes evaluated on the requested axis
    metric_type: str  # e.g., 'relative_lift_pct'
    p_control: float  # Baseline p_c used
    results: List[Dict[str, Any]]  # List of {p_t, asn, probs, ns}

    # Metadata for plotting
    null_x_value: float
    target_x_value: float
    n_max: int
    task_description: str = ""


class BinomialOperatingCharacteristicsSimulator:
    """
    Simulator for Binomial A/B Group Sequential Designs.
    """

    def __init__(self, method: GST.MethodSpec, task: GST.TaskSpec):
        self.method = method
        self.task = task
        self._validate()

    @classmethod
    def from_protocol(
        cls, protocol: GST.Protocol
    ) -> "BinomialOperatingCharacteristicsSimulator":
        return cls(protocol.method, protocol.task)

    def _validate(self) -> None:
        # Basic validation
        if not isinstance(self.task.efficacy, GST.EfficacyRequirement):
            raise ValueError("Task is missing EfficacyRequirement.")

    def simulate(
        self,
        range_min: float = -0.5,  # -50%
        range_max: float = 0.5,  # +50%
        n_points: int = 100,
        n_sims: int = 10000,
        metric_type: str = "relative_lift_pct",
        force_p_control: Optional[float] = None,
        ignore_futility: bool = False,
    ) -> OperatingCharacteristicsResult:
        """
        Runs the simulation over the specified range.

        Args:
            range_min: Min value for effect axis (e.g. -0.5 for -50% relative lift).
            range_max: Max value for effect axis.

            metric_type: 'relative_lift_pct' or 'absolute_diff_pct'.
            force_p_control: Override control proportion if not inferable.
            ignore_futility: If True, simulates behavior as if futility boundaries do not exist.
                             (Useful for visualizing conservative ASN in Non-Binding designs).
        """
        # 1. Infer Parameters
        p_c = 0.5
        task_target_p_t = 0.5

        # Try to infer from task
        if isinstance(self.task.hypotheses.target_effect, GST.BinaryEffectSize):
            props = self.task.hypotheses.target_effect.proportions
            if "control" in props:
                p_c = props["control"]
            elif len(props) > 0:
                p_c = list(props.values())[0]

            if "treatment" in props:
                task_target_p_t = props["treatment"]
            elif len(props) > 1:
                task_target_p_t = list(props.values())[1]

        if force_p_control is not None:
            p_c = force_p_control

        # 2. Define Treatment Range based on Metric
        if metric_type == "relative_lift_pct":
            # range_min/max are fractions (e.g. -0.5, 0.5) for consistency with internal logic,
            # but usually passed as decimal.
            # Convert range to p_treatments
            # Lift = (p_t / p_c) - 1
            # p_t = p_c * (1 + Lift)
            p_min = p_c * (1 + range_min)
            p_max = p_c * (1 + range_max)

            x_values = np.linspace(range_min * 100, range_max * 100, n_points)
            null_x = 0.0
            target_x = (task_target_p_t / p_c - 1) * 100

        elif metric_type == "absolute_diff_pct":
            # range_min is absolute diff (e.g. -0.05 for -5pp)
            # p_t = p_c + diff
            p_min = p_c + range_min
            p_max = p_c + range_max

            x_values = np.linspace(range_min * 100, range_max * 100, n_points)
            null_x = 0.0
            target_x = (task_target_p_t - p_c) * 100
        else:
            raise ValueError(f"Unsupported metric: {metric_type}")

        p_treatments = np.linspace(p_min, p_max, n_points)

        # 3. Setup Model
        timer = self.method.stopping_policy.timer
        if not isinstance(timer, GST.SampleSizeTimer):
            raise ValueError("Simulation only supports SampleSizeTimer")
        n_max = int(timer.max_sample_size)

        sigma = np.sqrt(p_c * (1 - p_c))
        i_max = n_max / (4 * sigma**2)

        # We construct a protocol-like object or use CanonicalJointModel directly
        # Easiest way is to wrap in a Protocol object for parsing,
        # or manually reconstruct if we want to avoid dependency cycles.
        # But we accept Protocol in classmethod, so we can just use self.method and self.task

        temp_proto = GST.Protocol(name="Sim", task=self.task, method=self.method)
        model = CanonicalJointModel.from_spec(temp_proto, n_sims=n_sims)

        # Compute drift corresponding to the target effect size
        target_delta_abs = abs(task_target_p_t - p_c)
        target_drift_ref = target_delta_abs * np.sqrt(i_max)

        # Solve boundaries for this target drift
        upper, lower = model.solve_boundaries(drift=target_drift_ref)

        if upper is None:
            upper = np.full(len(model.info_times), np.inf)

        # Determine effective lower boundaries
        # If ignore_futility is True (e.g. for Non-Binding ASN visualization),
        # we treat the lower boundary as -infinity (never stop for futility).
        if ignore_futility:
            lower = np.full(len(model.info_times), -np.inf)
        elif lower is None:
            lower = np.full(len(model.info_times), -np.inf)

        # 4. Run Loop
        results = []
        ns = np.ceil(model.info_times * n_max).astype(int)

        for i, p_t in enumerate(p_treatments):
            # Signed drift
            delta = p_t - p_c
            d = delta * np.sqrt(i_max)

            res_probs = calculate_stopping_probabilities(
                model.info_times,
                upper,
                lower,
                drift=d,
                n_sims=n_sims,
                method="simulation",
            )

            asn = np.sum(res_probs["probs"] * ns)

            results.append(
                {"p_t": p_t, "asn": asn, "probs": res_probs["probs"], "ns": ns}
            )

        return OperatingCharacteristicsResult(
            x_values=x_values,
            metric_type=metric_type,
            p_control=p_c,
            results=results,
            null_x_value=null_x,
            target_x_value=target_x,
            n_max=n_max,
        )
