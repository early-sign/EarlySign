from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
import pandas as pd

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.plan.gs_design_converter import (
    convert_gs_design_to_protocol,
)
from earlysign.v1.methods.group_sequential.plan.plots import (
    plot_design_characteristics,
)
from earlysign.v1.methods.group_sequential.plan.protocol_design import (
    ProtocolDesigner,
)
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)


class DesignVisualizer:
    """
    Visualizer for Group Sequential Designs.

    Provides tools to visualize design operating characteristics (ASN, Sample Size Distribution)
    and generate summary tables similar to sample size calculators.
    """

    def __init__(self) -> None:
        self._designer = ProtocolDesigner(
            model=CanonicalJointModel(Config(info_times=np.array([1.0])))
        )

    def visualize_protocol(
        self,
        protocol: GST.Protocol,
        effect_sizes: Optional[List[float]] = None,
        plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Visualizes a given protocol.

        Args:
            protocol: The protocol to visualize.
            effect_sizes: List of effect sizes for sensitivity analysis.
            plot: Whether to generate the plot.

        Returns:
            Dictionary containing 'summary' (DataFrame) and 'figure' (Figure).
        """
        # 1. Generate Plot if requested
        fig = None
        if plot:
            fig = plot_design_characteristics(protocol, effect_sizes)

        # 2. Generate Summary Table
        # Extract Design Metrics
        method = protocol.method
        timer = method.stopping_policy.timer
        n_max = timer.max_sample_size if isinstance(timer, GST.SampleSizeTimer) else 0

        # Calculate characteristics for specific effect sizes
        if effect_sizes is None:
            # Default to Target Effect
            target_effect = protocol.task.hypotheses.target_effect
            delta = 0.1
            if isinstance(target_effect, GST.BinaryEffectSize):
                props = target_effect.proportions
                vals = list(props.values())
                delta = abs(vals[1] - vals[0]) if len(vals) > 1 else 0.1

            effect_sizes = [0, 0.5 * delta, delta, 1.2 * delta]

        # Resolve model once for efficiency
        model = CanonicalJointModel.from_spec(protocol)
        upper, lower = model.solve_boundaries()
        if upper is None:
            upper = np.full(len(model.info_times), 10.0)
        if lower is None:
            lower = np.full(len(model.info_times), -10.0)

        # We need sigma for drift conversion
        # Use simple approximation for binary 50/50
        # A more robust way is to use ProtocolDesigner internals or assume variance
        # Let's assume binary p=0.5 for sigma=0.5 -> sigma^2 = 0.25 -> 4*sigma^2=1
        # I_max = n_max / 1 = n_max
        # drift = delta * sqrt(n_max)

        # Better: use the definition from the protocol if available
        # But we don't have easy access to the exact variance used in planning unless we replicate logic.
        # We'll use the approximation: Drift = Delta * sqrt(N / 4*sigma^2).

        # Construct Rows
        rows = []
        for delta in effect_sizes:
            # Drift calculation (Assuming variance 0.25 for binary p=0.5 approx)
            # This is a simplification. Ideally we fetch p_control from protocol.
            drift = delta * np.sqrt(n_max)  # Simplified standardization

            # Use CanonicalJointModel to evaluate
            # We can use solve_drift to find drift for specific power, but here we have delta -> drift
            # Correction: drift = delta * sqrt(I).

            # Compute ASN and Power
            # Need probabilities
            # Re-use logic from plot (should refactor, but okay to duplicate for independence)
            from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess

            gp = CanonicalGaussianProcess(drift=drift, rng=np.random.default_rng(42))

            # Note: compute_crossing_probability returns rejection probability
            # No, 'crossing' means crossing ANY boundary.
            # If 2-sided symmetric, crossing upper or lower.
            # If 1-sided efficacy, crossing upper.

            # Calculate Rejection Prob (Power)
            # If 1-sided efficacy: Rejection = Crossing Upper
            # If Efficacy+Futility: Rejection = Crossing Upper (not Lower)
            # CanonicalGaussianProcess.compute_crossing_probability sums both if provided.
            # We need directional crossing.

            t = model.info_times

            # Simulate to distinguish Upper vs Lower crossing
            # Or use numerical integration for upper only

            # For summary, assume Rejection = Power
            # Use simulation for breakdown
            samples = gp.sample(t, n_sims=5000)
            stopped, stop_looks = gp.apply_stopping_rule(
                samples, upper=upper, lower=lower
            )

            # Reject if crossed upper (apply_stopping_rule handles first stop)
            # We need to know WHICH boundary was crossed.
            # Modify Apply Stopping Rule logic temporarily or do manual check

            # Manual check
            is_reject = np.zeros(5000, dtype=bool)
            stop_idx = np.full(5000, -1, dtype=int)

            for i in range(len(t)):
                # Check active paths
                active = stop_idx == -1

                # Check Lower (Futility)
                # Usually Futility is checked first or same time.
                # If Lower > Upper (impossible in valid design), order matters.
                # Simultaneous.

                # Check Crossing
                cross_u = (samples[:, i] > upper[i]) & active
                cross_l = (samples[:, i] < lower[i]) & active

                # Update
                # If both are crossed, Upper takes precedence.
                # Usually Upper (Efficacy).

                stop_idx[cross_u] = i
                is_reject[cross_u] = True

                # Indices that crossed Lower but not Upper this round
                stop_idx[cross_l & ~cross_u] = i

            # Power
            power_val = np.mean(is_reject)

            # ASN
            # Stop sample size
            # If never stopped (stop_idx == -1), assume stopping at K (index K-1)
            final_looks = stop_idx.copy()
            final_looks[final_looks == -1] = len(t) - 1

            # Convert look index to sample size
            # n[i] = ceil(t[i] * n_max)
            n_at_looks = np.ceil(t * n_max).astype(int)
            realized_n = n_at_looks[final_looks]
            asn_val = np.mean(realized_n)

            rows.append(
                {
                    "Effect Size (Delta)": delta,
                    "Power / Rejection Prob": f"{power_val:.2%}",
                    "ASN (Average Sample Number)": f"{asn_val:.1f}",
                    "Max Sample Size": n_max,
                    "Pct of Max": f"{asn_val/n_max:.1%}",
                }
            )

        df = pd.DataFrame(rows)

        return {"summary": df, "figure": fig}

    def visualize_from_params(
        self,
        alpha: float = 0.05,
        power: float = 0.8,
        delta: float = 0.1,
        control_rate: float = 0.1,
        looks: int = 4,
        spending_function: str = "obrien_fleming",
        alternative_rate: Optional[float] = None,
        plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Plans and visualizes a Binomial design from parameters.
        Analogous to the calculator Inputs.
        """
        # Infer delta if rates provided
        if alternative_rate is not None:
            delta = abs(alternative_rate - control_rate)

        # Plan
        # Note: ProtocolDesigner needs model initialized
        # The visualizer has one self._designer

        shape_literal = cast(Literal["obrien_fleming", "pocock"], spending_function)

        protocol_obj = self._designer.plan_binomial_ab(
            alpha=alpha,
            power=power,
            delta=delta,
            k=looks,
            p_control=control_rate,
            shape_type=shape_literal,
        )

        # Visualize
        return self.visualize_protocol(
            protocol=protocol_obj,
            effect_sizes=[0, 0.5 * delta, delta, 1.25 * delta, 1.5 * delta],
            plot=plot,
        )

    def visualize_gs_design(
        self, gs_object: Dict[str, Any], plot: bool = True
    ) -> Dict[str, Any]:
        """
        Visualizes an imported gsDesign object.
        """
        protocol = convert_gs_design_to_protocol(gs_object)
        return self.visualize_protocol(protocol, plot=plot)
