from typing import Any, Dict, Optional, Self

import numpy as np

import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)
from earlysign.v1.methods.group_sequential.shared.spending import (
    SpendingFunction,
    SpendingFunctionFactory,
)


class ProtocolDesigner:
    """
    Designer for group sequential protocols.
    Translates scientific intent (alpha, power, delta) into a realized design (boundaries, sample size).
    """

    def __init__(self, model: Optional[CanonicalJointModel] = None):
        self._model = model

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Self:
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
            model = CanonicalJointModel(
                Config(
                    info_times=np.array([1.0]),  # Placeholder for design-phase use
                    rng_seed=model_params.get("rng_seed"),
                )
            )
        return cls(model=model)

    def plan_binomial_ab(
        self,
        alpha: float,
        power: float,
        delta: float,
        k: int,
        p_control: float,
        spending_fn: Optional[SpendingFunction] = None,
        side: int = 1,
        rho: float = 3.0,
    ) -> GST.Protocol:
        """
        Plans a binomial A/B design and returns a fully populated GST.Protocol.
        """
        # Average variance under H0 approx: p_control * (1 - p_control)
        sigma2 = p_control * (1.0 - p_control)
        theta = delta  # difference in proportions

        info_times = np.linspace(1 / k, 1.0, k)

        if self._model is None:
            raise ValueError(
                "ProtocolDesigner must be initialized with a CanonicalJointModel for planning."
            )

        # Create a new model instance for this specific design planning
        # to ensure info_times match the requested k.
        base_seed = self._model.config.rng_seed
        plan_config = Config(
            info_times=info_times,
            rng_seed=base_seed,
        )
        model = CanonicalJointModel(plan_config)

        # Use provided spending function for bound solving
        # Fallback to OBF for safety if None
        if spending_fn is None:
            # Default to OBF if not provided
            factory = SpendingFunctionFactory(budget=alpha)
            spending_fn = factory.build_from_spec(
                GST.SpendingFunction(family="obrien_fleming")
            )

        boundaries, _ = model.solve_boundaries(
            efficacy_spending=spending_fn,
        )
        if boundaries is None:
            raise ValueError("Failed to solve boundaries.")
        boundaries_list = boundaries.tolist()

        # 2. Solve for standardized drift delta = theta * sqrt(I_max)
        drift = model.solve_drift(
            info_times.tolist(), boundaries_list, target_power=power
        )

        # 3. Calculate I_max = (drift / theta) ** 2
        i_max = (drift / theta) ** 2

        # 4. Map to sample size n_max (total for both arms)
        n_max_float = 4 * i_max * sigma2
        n_max = int(np.ceil(n_max_float))

        # Construct the realized protocol with new schema
        return GST.Protocol(
            name="Designed Protocol",
            task=GST.TaskSpec(
                kind="group_sequential",
                arms=["control", "treatment"],
                response_type=GST.ResponseType.BINARY,
                hypotheses=GST.HypothesisSpec(
                    h_null_description="Difference <= 0",
                    h_alt_description=f"Difference > {delta}",
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
                stopping_policy=GST.StoppingPolicySpec(
                    statistic=GST.TwoArmBinomialZ(
                        variance_estimation=GST.VarianceEstimation.POOLED
                    ),
                    strategy=GST.AlphaSpendingStrategy(
                        spending_fn=GST.SpendingFunction(family=spending_fn.name),
                        budget=alpha,
                        sided=GST.Sided.ONE,
                        statistical_model=GST.CanonicalGaussianModel(),
                    ),
                    timer=GST.SampleSizeTimer(
                        unit=GST.Unit.INDIVIDUALS,
                        max_sample_size=n_max,
                    ),
                    schedule=GST.FixedSchedule(
                        analyses=info_times.tolist(),
                    ),
                ),
            ),
        )

    def method_from_task_spec(
        self, task: GST.TaskSpec, params: Dict[str, Any]
    ) -> GST.MethodSpec:
        """
        Derives a MethodSpec from a TaskSpec effectively serving as a 'Design Strategy'.
        """
        efficacy = task.efficacy
        if not efficacy:
            raise ValueError("Task is missing efficacy requirements.")
        alpha = efficacy.alpha

        futility = task.futility
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
            keys = list(props.keys())
            if len(keys) > 1 and keys[1] != "control":
                p_t = props[keys[1]]
            else:
                raise ValueError("Could not identify treatment proportion")

        delta = abs(p_t - p_c)

        spending_params = params.get("spending_params", {})

        # Instantiate spending function
        factory = SpendingFunctionFactory(budget=alpha)
        spending_spec = GST.SpendingFunction(family=shape_type, params=spending_params)
        spending_fn = factory.build_from_spec(spending_spec)

        # Determine stopping policy based on presence of futility
        if futility:
            power = futility.power
            beta = 1.0 - power
            stopping_policy: (
                GST.AlphaSpendingStrategy
                | GST.BetaSpendingStrategy
                | GST.AlphaBetaSpendingStrategy
            ) = GST.AlphaBetaSpendingStrategy(
                alpha_spending_fn=GST.SpendingFunction(
                    family=shape_type, params=spending_params
                ),
                beta_spending_fn=GST.SpendingFunction(
                    family=shape_type, params=spending_params
                ),
                alpha_budget=alpha,
                beta_budget=beta,
                alpha_binding=(
                    efficacy.binding if efficacy.binding is not None else True
                ),
                beta_binding=(
                    futility.binding if futility.binding is not None else False
                ),
                statistical_model=GST.CanonicalGaussianModel(),
            )
        else:
            stopping_policy = GST.AlphaSpendingStrategy(
                spending_fn=GST.SpendingFunction(
                    family=shape_type, params=spending_params
                ),
                budget=alpha,
                sided=GST.Sided.ONE,
                statistical_model=GST.CanonicalGaussianModel(),
            )

        # Calculate schedule from planning
        power_for_plan = futility.power if futility else 0.8
        generic_proto = self.plan_binomial_ab(
            alpha=alpha,
            power=power_for_plan,
            delta=delta,
            k=k,
            p_control=p_c,
            spending_fn=spending_fn,
        )

        return GST.MethodSpec(
            kind="group_sequential",
            stopping_policy=GST.StoppingPolicySpec(
                statistic=GST.TwoArmBinomialZ(
                    variance_estimation=GST.VarianceEstimation.POOLED
                ),
                strategy=stopping_policy,
                timer=generic_proto.method.stopping_policy.timer,
                schedule=generic_proto.method.stopping_policy.schedule,
            ),
        )

    def plan_continuous_ab(
        self,
        alpha: float,
        power: float,
        delta: float,
        sigma: float,
        k: int,
        spending_fn: Optional[SpendingFunction] = None,
    ) -> GST.Protocol:
        """
        Plans a Continuous (Two Means) A/B design.

        Parameters
        ----------
        alpha : float
            Significance level (one-sided).
        power : float
            Target power (1 - beta).
        delta : float
            Difference in means (mu_treatment - mu_control).
        sigma : float
            Assumed common standard deviation.
        k : int
            Number of looks.
        spending_fn : Optional[SpendingFunction]
            Spending function configuration.

        Returns
        -------
        GST.Protocol
            Populated protocol with TwoArmContinuousZ statistic.
        """
        # Standardized effect size
        # theta = delta / (2 * sigma) for equal allocation n1=n2=N/2
        theta = delta / (2 * sigma)

        info_times = np.linspace(1 / k, 1.0, k)

        if self._model is None:
            self._model = CanonicalJointModel(Config(info_times=np.array([1.0])))

        # Use temporary model for planning
        base_seed = self._model.config.rng_seed
        model = CanonicalJointModel(Config(info_times=info_times, rng_seed=base_seed))

        # Resolve spending function
        if spending_fn is None:
            factory = SpendingFunctionFactory(budget=alpha)
            spending_fn = factory.build_from_spec(
                GST.SpendingFunction(family="obrien_fleming")
            )

        boundaries, _ = model.solve_boundaries(efficacy_spending=spending_fn)
        if boundaries is None:
            raise ValueError("Failed to solve boundaries.")
        boundaries_list = boundaries.tolist()

        # Solve for drift
        drift = model.solve_drift(
            info_times.tolist(), boundaries_list, target_power=power
        )

        # Calculate I_max (Total Sample Size N)
        # drift = theta * sqrt(I_max) => I_max = (drift / theta) ** 2
        i_max = (drift / theta) ** 2
        n_max = int(np.ceil(i_max))

        return GST.Protocol(
            name="Continuous AB Protocol",
            task=GST.TaskSpec(
                kind="group_sequential",
                arms=["control", "treatment"],
                response_type=GST.ResponseType.CONTINUOUS,
                hypotheses=GST.HypothesisSpec(
                    h_null_description="Difference <= 0",
                    h_alt_description=f"Difference > {delta}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.ContinuousEffectSize(
                        means={"control": 0.0, "treatment": delta},
                        standard_deviation=sigma,
                    ),
                ),
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
            ),
            method=GST.MethodSpec(
                kind="group_sequential",
                stopping_policy=GST.StoppingPolicySpec(
                    statistic=GST.TwoArmContinuousZ(
                        information_unit="fisher_information",
                        variance=GST.TwoArmEstimatedVariance(
                            kind="estimated", method=GST.MethodModel.POOLED
                        ),
                    ),
                    strategy=GST.AlphaSpendingStrategy(
                        spending_fn=GST.SpendingFunction(family=spending_fn.name),
                        budget=alpha,
                        sided=GST.Sided.ONE,
                        statistical_model=GST.CanonicalGaussianModel(),
                    ),
                    timer=GST.SampleSizeTimer(
                        unit=GST.Unit.INDIVIDUALS,
                        max_sample_size=n_max,
                    ),
                    schedule=GST.FixedSchedule(
                        analyses=info_times.tolist(),
                    ),
                ),
            ),
        )

    def plan_survival_ab(
        self,
        alpha: float,
        power: float,
        hazard_ratio: float,
        k: int,
        spending_fn: Optional[SpendingFunction] = None,
    ) -> GST.Protocol:
        """
        Plans a Survival (Time-to-Event) A/B design using Log-Rank Test.

        Parameters
        ----------
        hazard_ratio : float
            Target Hazard Ratio (lambda_treatment / lambda_control).
            Assumption: < 1 indicates benefit.
            Converted to standardized effect: theta = |log(HR)| / 2.

        Returns
        -------
        GST.Protocol
            Populated protocol with EventCountTimer.
        """
        log_hr = np.log(hazard_ratio)
        theta = abs(log_hr) / 2.0

        info_times = np.linspace(1 / k, 1.0, k)

        if self._model is None:
            self._model = CanonicalJointModel(Config(info_times=np.array([1.0])))

        base_seed = self._model.config.rng_seed
        model = CanonicalJointModel(Config(info_times=info_times, rng_seed=base_seed))

        if spending_fn is None:
            factory = SpendingFunctionFactory(budget=alpha)
            spending_fn = factory.build_from_spec(
                GST.SpendingFunction(family="obrien_fleming")
            )

        boundaries, _ = model.solve_boundaries(efficacy_spending=spending_fn)
        if boundaries is None:
            raise ValueError("Failed to solve boundaries.")
        boundaries_list = boundaries.tolist()

        drift = model.solve_drift(
            info_times.tolist(), boundaries_list, target_power=power
        )

        # I_max (Total Events)
        # drift = theta * sqrt(Events)
        events_max = int(np.ceil((drift / theta) ** 2))

        return GST.Protocol(
            name="Survival AB Protocol",
            task=GST.TaskSpec(
                kind="group_sequential",
                arms=["control", "treatment"],
                response_type=GST.ResponseType.TIME_TO_EVENT,
                hypotheses=GST.HypothesisSpec(
                    h_null_description="HR >= 1",
                    h_alt_description=f"HR < {hazard_ratio}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.SurvivalEffectSize(
                        hazard_ratios={"control": 1.0, "treatment": hazard_ratio}
                    ),
                ),
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
            ),
            method=GST.MethodSpec(
                kind="group_sequential",
                stopping_policy=GST.StoppingPolicySpec(
                    statistic=GST.TwoArmContinuousZ(
                        kind="two_arm_continuous_z",
                        information_unit="fisher_information",
                        variance=GST.KnownVariance(value=1.0),
                    ),
                    strategy=GST.AlphaSpendingStrategy(
                        spending_fn=GST.SpendingFunction(family=spending_fn.name),
                        budget=alpha,
                        sided=GST.Sided.ONE,
                        statistical_model=GST.CanonicalGaussianModel(),
                    ),
                    timer=GST.EventCountTimer(
                        max_events=events_max,
                    ),
                    schedule=GST.FixedSchedule(
                        analyses=info_times.tolist(),
                    ),
                ),
            ),
        )
