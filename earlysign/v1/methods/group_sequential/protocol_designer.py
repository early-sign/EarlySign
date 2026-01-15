from typing import Any, Dict, Literal, Optional

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.canonical_dist import (
    CanonicalJointModel,
)


class ProtocolDesigner:
    """
    Designer for group sequential protocols.
    Translates scientific intent (alpha, power, delta) into a realized design (boundaries, sample size).
    """

    def __init__(self, model: Optional[CanonicalJointModel] = None):
        self._model = model

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ProtocolDesigner":
        """
        Creates a ProtocolDesigner instance from a configuration dictionary.
        Supported keys:
            - 'model': 'canonical_joint' (mapped to CanonicalJointModel)
            - 'model_params': Dict containing 'rng_seed', etc.
        """
        model = None
        model_type = config.get("model")
        model_params = config.get("model_params", {})
        
        if model_type == "canonical_joint":
            from earlysign.v1.methods.group_sequential.canonical_dist import Config
            model = CanonicalJointModel(Config(
                info_times=np.array([1.0]), # Placeholder for design-phase use
                rng_seed=model_params.get("rng_seed")
            ))
        return cls(model=model)

    def plan_binomial_ab(
        self,
        alpha: float,
        power: float,
        delta: float,
        k: int,
        p_control: float,
        shape_type: Literal["obrien_fleming", "pocock"] = "obrien_fleming",
        side: int = 1,
        rho: float = 3.0,
    ) -> GST.Protocol:
        """
        Plans a binomial A/B design and returns a fully populated GST.Protocol.

        Args:
            alpha: Type-1 error rate.
            power: Target power (1 - beta).
            delta: Minimum clinically meaningful difference.
            k: Number of looks.
            p_control: Baseline conversion rate (for variance estimation).
            shape_type: Spending function / boundary shape.
            side: Number of sides (1 or 2).
            rho: Parameter for spending function if applicable.

        Returns:
            GST.Protocol with design fields populated.
        """
        # Average variance under H0 approx: p_control * (1 - p_control)
        sigma2 = p_control * (1.0 - p_control)
        theta = delta  # difference in proportions

        info_times = np.linspace(1 / k, 1.0, k)

        if self._model is None:
            raise ValueError("ProtocolDesigner must be initialized with a CanonicalJointModel for planning.")
        model = self._model

        c_val = model.solve_boundary_constant(
            info_times.tolist(), alpha, shape_type=shape_type
        )

        if shape_type == "pocock":
            c_shape = np.ones(k)
        elif shape_type == "obrien_fleming":
            c_shape = 1.0 / np.sqrt(info_times)
        else:
            raise ValueError(f"Unsupported shape: {shape_type}")

        boundaries = (c_val * c_shape).tolist()

        # 2. Solve for standardized drift delta = theta * sqrt(I_max)
        drift = model.solve_drift(
            info_times.tolist(), boundaries, target_power=power
        )

        # 3. Calculate I_max = (drift / theta)^2
        i_max = (drift / theta) ** 2

        # 4. Map to sample size n_max (total for both arms)
        # Information I = n_total / (4 * sigma^2) => n_total = 4 * I * sigma^2
        n_max_float = 4 * i_max * sigma2
        n_max = int(np.ceil(n_max_float))

        # 5. Calculate Schedule Points
        n_schedule = [float(int(np.ceil(n_max * t))) for t in info_times]

        # Construct the realized protocol
        return GST.Protocol(
            # Intent
            name="Designed Protocol",
            task=GST.TaskSpec(
                kind="group_sequential",
                arms=["control", "treatment"],  # Default/Placeholder
                response_type=GST.ResponseType.BINARY,
                hypotheses=GST.HypothesisSpec(
                    h_null="Difference <= 0",
                    h_alt=f"Difference > {delta}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(
                        proportions={
                            "control": p_control,
                            "treatment": p_control + delta,
                        }
                    ),
                ),
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
            ),
            method=GST.MethodSpec(
                kind="group_sequential",
                efficacy=GST.StoppingRule(
                    schedule=GST.ScheduleSpec(
                        unit=GST.Unit.SAMPLE_SIZE,
                        n_looks=k,
                        interim_points=n_schedule,  # Store the sample sizes
                    ),
                    boundary=GST.SpendingBoundary(
                        kind="spending",
                        boundary_scale=GST.BoundaryScale.Z_SCORE,
                        binding=True,
                        reference_model=GST.BinaryModel(
                            kind="binary",
                            test_statistic=GST.TestStatistic.Z,
                            link_function=GST.LinkFunction.IDENTITY,
                            use_canonical_joint_distribution=True,
                        ),
                        spending_function=GST.SpendingFunctionSpec(type=shape_type),
                    ),
                ),
            ),
        )

    def method_from_task_spec(
        self, task: GST.TaskSpec, params: Dict[str, Any]
    ) -> GST.MethodSpec:
        """
        Derives a MethodSpec from a TaskSpec effectively serving as a 'Design Strategy'.

        Args:
            task: The generic task specification containing requirements.
            params: Dictionary containing design parameters (e.g. 'looks', 'spending_function').

        Returns:
            A populated GST.MethodSpec.
        """
        efficacy = task.efficacy
        if not efficacy:
            raise ValueError("Task is missing efficacy requirements.")
        alpha = efficacy.alpha

        futility = task.futility
        if not futility:
            raise ValueError("Task is missing futility requirements.")
        power = futility.power
        k = params.get("looks", 2)
        shape_type = params.get("spending_function", "obrien_fleming")

        hypotheses = task.hypotheses
        if not hypotheses:
            raise ValueError("Task is missing hypotheses.")

        if not isinstance(hypotheses.target_effect, GST.BinaryEffectSize):
            raise ValueError("Task must have BinaryEffectSize for Binomial Design")

        props = hypotheses.target_effect.proportions
        p_c = props.get("control") or list(props.values())[0]

        # Heuristic to find treatment or second value
        p_t = props.get("treatment")
        if p_t is None:
            # Fallback: if there is a second key
            keys = list(props.keys())
            if len(keys) > 1 and keys[1] != "control":
                p_t = props[keys[1]]
            else:
                raise ValueError("Could not identify treatment proportion")

        delta = abs(p_t - p_c)

        # 2. Plan the design
        generic_proto = self.plan_binomial_ab(
            alpha=alpha,
            power=power,
            delta=delta,
            k=k,
            p_control=p_c,
            shape_type=shape_type,
        )

        return generic_proto.method
