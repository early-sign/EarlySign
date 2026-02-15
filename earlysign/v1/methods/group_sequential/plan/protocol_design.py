from decimal import Decimal
from typing import Any, Dict, Optional, Self

import numpy as np
from pydantic import BaseModel, Field

import earlysign.schema.ES3.Base as ES3_BASE
import earlysign.schema.ES3.GST as GST
from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import (
    CanonicalJointModel,
    Config,
)
from earlysign.v1.methods.group_sequential.shared.spending import (
    SpendingFunction,
    SpendingFunctionFactory,
)


class MethodDesignParams(BaseModel):
    """Internal model for design parameter validation."""

    looks: int = Field(..., gt=0)
    spending_function: Optional[str] = None
    spending_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    power: Optional[float] = None
    ssr_method: Optional[str] = None


class ProtocolDesigner:
    """Designer for group sequential protocols.

    Translates scientific intent (alpha, power, delta) into a realized
    design (boundaries, sample size).
    """

    def __init__(self, model: Optional[CanonicalJointModel] = None):
        self._model = model

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Self:
        """Creates a ProtocolDesigner instance from a configuration dictionary.

        Supported keys:
            - 'model': 'canonical_joint' (mapped to CanonicalJointModel)
            - 'model_params': Dict containing 'rng_seed', etc.

        Args:
            config: A dictionary containing configuration parameters.

        Returns:
            A ProtocolDesigner instance.
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

    def design_gs_binomial(
        self,
        alpha: float,
        power: float,
        p_control: float,
        p_treatment: float,
        looks: int,
        spending_function: str,
        scheduling: str | np.ndarray = "equidistant",
        spending_params: Optional[Dict[str, Any]] = None,
        futility: bool = True,
        futility_binding: bool = False,
        tails: int = 1,
        rng_seed: Optional[int] = None,
    ) -> tuple[GST.MethodSpec, int]:
        """Core logic for designing a Binomial Group Sequential Test.

        Args:
            alpha: Type I error rate.
            power: Statistical power.
            p_control: Baseline proportion.
            p_treatment: Target proportion.
            looks: Number of analyses (if info_times not provided).
            scheduling: "equidistant", "asn_minimizer", or array of fractions.
            spending_function: Spending function family.
            spending_params: Spending function parameters.
            futility: Whether to include futility boundaries.
            futility_binding: Whether futility boundaries are binding.
            tails: 1 or 2 sided.
            arms: 1 or 2.
            rng_seed: Random seed for model.

        Returns:
            A tuple of (MethodSpec, n_max).
        """
        # 0. Setup Spending Function Specimens (for Model Solver)
        sf_factory = SpendingFunctionFactory(budget=alpha)
        sf_eff = sf_factory.build_from_spec(
            GST.SpendingFunction(family=spending_function, params=spending_params)
        )
        sf_fut = None
        if futility:
            # Use Decimal for the budget calculation 1.0 - power to avoid floating-point
            # artifacts (e.g., 1.0 - 0.8 becoming 0.19999999999999996).
            beta_budget = float(Decimal("1.0") - Decimal(str(power)))
            sf_factory_fut = SpendingFunctionFactory(budget=beta_budget)
            sf_fut = sf_factory_fut.build_from_spec(
                GST.SpendingFunction(family=spending_function, params=spending_params)
            )

        # 1. Determine Schedule
        if isinstance(scheduling, (np.ndarray, list)):
            info_times = np.array(scheduling)
        elif scheduling == "equidistant":
            info_times = np.linspace(1 / looks, 1.0, looks)
        elif scheduling == "asn_minimizer":
            from earlysign.v1.methods.group_sequential.plan.schedule_optimization import (
                optimize_schedule,
            )

            # 1a. Solve Drift for Standard Target (using equidistant proxy) to drive optimizer
            proxy_times = np.linspace(1 / looks, 1.0, looks)
            proxy_cfg = Config(
                info_times=proxy_times,
                rng_seed=rng_seed,
                efficacy_binding=True,
                futility_binding=futility_binding,
                tails=tails,
            )
            proxy_model = CanonicalJointModel(proxy_cfg)
            u_prox, l_prox = proxy_model.solve_boundaries(
                efficacy_spending=sf_eff,
                futility_spending=sf_fut,
                drift=1.0,
            )
            drift_target = proxy_model.solve_drift(
                proxy_times.tolist(),
                u_prox.tolist() if u_prox is not None else [],
                target_power=power,
                futility_boundaries=l_prox.tolist() if l_prox is not None else None,
                tails=tails,
            )

            # 1b. Run Schedule Optimization
            res = optimize_schedule(
                k_looks=looks,
                efficacy_spending=sf_eff,
                futility_spending=sf_fut,
                alpha=alpha,
                drift=drift_target,
                tails=tails,
                method="simulation",
                seed=rng_seed,
            )
            info_times = res.schedule
        else:
            raise ValueError(f"Invalid scheduling option: {scheduling}")

        # 2. Setup Final Model
        model_cfg = Config(
            info_times=info_times,
            rng_seed=rng_seed,
            efficacy_binding=True,
            futility_binding=futility_binding,
            tails=tails,
        )
        model = CanonicalJointModel(model_cfg)

        # 3. Solve Boundaries
        # During design, we solve for boundaries at a canonical drift of 1.0.
        # This determines the 'shape' of the boundaries on the Z-scale.
        # Note: Since Boundaries on Z-scale are generally scale-invariant for
        # spending functions, solving at drift=1.0 provides the standard
        # normalized boundaries which are then scaled to the target power.
        upper, lower = model.solve_boundaries(
            efficacy_spending=sf_eff,
            futility_spending=sf_fut,
            drift=1.0,
        )

        # 4. Solve Drift
        drift = model.solve_drift(
            info_times.tolist(),
            upper.tolist() if upper is not None else [],
            target_power=power,
            futility_boundaries=lower.tolist() if lower is not None else None,
            tails=tails,
        )

        # 5. Calculate Sample Size (n_max)
        delta = abs(p_treatment - p_control)
        sigma2 = p_control * (1.0 - p_control)
        theta = delta
        i_max = (drift / theta) ** 2

        # Two-sample balanced
        n_max = int(np.ceil(4 * i_max * sigma2))

        # 6. Construct MethodSpec
        strategy: Any
        if futility:
            strategy = GST.AlphaBetaSpendingStrategy(
                alpha_spending_fn=GST.SpendingFunction(
                    family=spending_function, params=spending_params
                ),
                beta_spending_fn=GST.SpendingFunction(
                    family=spending_function, params=spending_params
                ),
                alpha_budget=alpha,
                beta_budget=float(Decimal("1.0") - Decimal(str(power))),
                alpha_binding=True,
                beta_binding=futility_binding,
                statistical_model=GST.CanonicalGaussianModel(),
            )
        else:
            strategy = GST.AlphaSpendingStrategy(
                spending_fn=GST.SpendingFunction(
                    family=spending_function, params=spending_params
                ),
                budget=alpha,
                sided=GST.Sided.ONE if tails == 1 else GST.Sided.TWO,
                statistical_model=GST.CanonicalGaussianModel(),
            )

        method_spec = GST.MethodSpec(
            kind="group_sequential",
            stopping_policy=GST.StoppingPolicySpec(
                statistic=GST.TwoArmBinomialZ(
                    variance_estimation=GST.VarianceEstimation.POOLED
                ),
                strategy=strategy,
                timer=GST.SampleSizeTimer(
                    unit=GST.Unit.INDIVIDUALS,
                    max_sample_size=n_max,
                ),
                schedule=GST.FixedSchedule(analyses=info_times.tolist()),
            ),
        )

        return method_spec, n_max

    def design_gs_classic(
        self,
        alpha: float,
        power: float,
        delta: float,
        looks: int,
        type: str,
        p_control: Optional[float] = None,
        sigma: Optional[float] = None,
        wang_tsiatis_delta: float = 0.25,
        tails: int = 2,
        arms: int = 2,
        rng_seed: Optional[int] = None,
    ) -> tuple[GST.MethodSpec, int]:
        """Core logic for designing a Classic (Fixed Shape) Group Sequential Test.

        Args:
            alpha: Type I error rate.
            power: Statistical power.
            delta: Absolute effect size.
            looks: Number of analyses.
            type: "pocock", "obrien_fleming", or "wang_tsiatis".
            p_control: Required for Binomial designs.
            sigma: Required for Continuous designs.
            wang_tsiatis_delta: Delta for Wang-Tsiatis family (default 0.25).
            tails: 1 or 2 sided.
            arms: 1 or 2.
            rng_seed: Random seed.

        Returns:
            A tuple of (MethodSpec, n_max).
        """
        # 1. Setup Model & Strategy
        info_times = np.linspace(1 / looks, 1.0, looks)
        model = CanonicalJointModel(Config(info_times=info_times, rng_seed=rng_seed))

        shape_params = (
            {"delta_wt": wang_tsiatis_delta} if type == "wang_tsiatis" else {}
        )

        # 2. Solve Constant and Boundaries
        c_val = model.solve_boundary_constant(
            info_times=info_times.tolist(),
            alpha=alpha,
            shape_type=type,
            tails=tails,
            shape_params=shape_params,
        )

        if type == "pocock":
            strategy_cls: Any = GST.PocockStrategy
            bound_shape = np.ones(looks)
        elif type == "obrien_fleming":
            strategy_cls = GST.OBrienFlemingStrategy
            bound_shape = 1.0 / np.sqrt(info_times)
        elif type == "wang_tsiatis":
            strategy_cls = GST.WangTsiatisStrategy
            bound_shape = info_times ** (wang_tsiatis_delta - 0.5)
        else:
            raise ValueError(f"Unknown classic design type: {type}")

        boundaries = c_val * bound_shape

        # 3. Solve Drift
        drift = model.solve_drift(
            info_times.tolist(),
            boundaries.tolist(),
            target_power=power,
            tails=tails,
        )

        # 4. Map to Sample Size
        if p_control is not None:
            # Binomial
            sigma2_unit = p_control * (1.0 - p_control)
            theta = delta
            # Z = (p1-p2)/sqrt(var1/n1 + var2/n2). For balanced 2-arm: I = n / (4*sigma2).
            # For 1-arm Z = (p-p0)/sqrt(sigma2/n): I = n / sigma2.
            i_max = (drift / theta) ** 2
            n_max = int(np.ceil((4 if arms == 2 else 1) * i_max * sigma2_unit))
            timer_unit = GST.Unit.INDIVIDUALS
            stat_spec: Any = (
                GST.TwoArmBinomialZ(variance_estimation=GST.VarianceEstimation.POOLED)
                if arms == 2
                else GST.OneArmBinomialZ(
                    variance_source=GST.VarianceSource.NULL_HYPOTHESIS
                )
            )
        elif sigma is not None:
            # Continuous
            # Two-sample balanced: Z = delta / sqrt(4*sigma^2/n) = delta*sqrt(n)/(2*sigma).
            # theta = delta / (2*sigma). i_max = (drift / theta)**2. n_max = i_max.
            # One-sample: Z = delta / (sigma/sqrt(n)) = delta*sqrt(n)/sigma.
            # theta = delta / sigma. i_max = (drift/theta)**2. n_max = i_max.
            theta = delta / (2 * sigma if arms == 2 else sigma)
            i_max = (drift / theta) ** 2
            n_max = int(np.ceil(i_max))
            timer_unit = GST.Unit.INDIVIDUALS
            stat_spec = (
                GST.TwoArmContinuousZ(
                    information_unit="fisher_information",
                    variance=GST.TwoArmEstimatedVariance(
                        kind="estimated", method=GST.MethodModel.POOLED
                    ),
                )
                if arms == 2
                else GST.OneArmContinuousZ(
                    variance=GST.OneArmEstimatedVariance(kind="estimated")
                )
            )
        else:
            raise ValueError("Must provide either p_control or sigma.")

        # 5. Assemble MethodSpec
        # Prepare strategy arguments
        strategy_kwargs: Dict[str, Any] = {
            "alpha": alpha,
            "sided": GST.Sided.ONE if tails == 1 else GST.Sided.TWO,
            "statistical_model": GST.CanonicalGaussianModel(),
        }
        if type == "wang_tsiatis":
            strategy_kwargs["delta"] = wang_tsiatis_delta

        method_spec = GST.MethodSpec(
            kind="group_sequential",
            stopping_policy=GST.StoppingPolicySpec(
                statistic=stat_spec,
                strategy=strategy_cls(**strategy_kwargs),
                timer=GST.SampleSizeTimer(unit=timer_unit, max_sample_size=n_max),
                schedule=GST.FixedSchedule(analyses=info_times.tolist()),
            ),
        )

        return method_spec, n_max

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
        """Plans a binomial A/B design and returns a fully populated GST.Protocol.

        Args:
            alpha: Type I error rate.
            power: Statistical power (1 - Type II error rate).
            delta: The minimum detectable difference in proportions.
            k: The number of planned analyses (looks).
            p_control: The proportion in the control arm.
            spending_fn: Optional spending function to use for boundary calculation.
                If None, O'Brien-Fleming spending is used.
            side: The number of sides for the test (1 or 2).
            rho: Not used for binomial but kept for compatibility.

        Returns:
            A fully populated GST.Protocol representing the planned design.
        """
        info_times = np.linspace(1 / k, 1.0, k)
        p_treatment = p_control + delta

        spending_family = spending_fn.name if spending_fn else "obrien_fleming"
        spending_params = spending_fn.params if spending_fn else None

        method_spec, n_max = self.design_gs_binomial(
            alpha=alpha,
            power=power,
            p_control=p_control,
            p_treatment=p_treatment,
            looks=k,
            scheduling=info_times,
            spending_function=spending_family,
            spending_params=spending_params,
            futility=True,  # Default to including futility in planning
            tails=side,
            rng_seed=self._model.config.rng_seed if self._model else None,
        )

        # Construct the realized protocol
        return GST.Protocol(
            name="Designed Protocol",
            task=GST.TaskSpec(
                kind="group_sequential",
                arms=ES3_BASE.TwoArmComparison(
                    control_arm_name="control",
                    treatment_arm_name="treatment",
                ),
                response_type=GST.ResponseType.BINARY,
                hypotheses=GST.HypothesisSpec(
                    h_null_description="Difference <= 0",
                    h_alt_description=f"Difference > {delta}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(
                        proportions={
                            "control": p_control,
                            "treatment": p_treatment,
                        }
                    ),
                ),
                efficacy=GST.EfficacyRequirement(alpha=alpha),
                futility=GST.FutilityRequirement(power=power),
            ),
            method=method_spec,
        )

    def method_from_task_spec(
        self, task: GST.TaskSpec, params: Dict[str, Any]
    ) -> GST.MethodSpec:
        """
        Derives a MethodSpec from a TaskSpec effectively serving as a 'Design Strategy'.

        Examples:
            >>> from earlysign.v1.methods.group_sequential.plan.protocol_design import ProtocolDesigner
            >>> import earlysign.schema.ES3.Base as ES3_BASE
            >>> import earlysign.schema.ES3.GST as GST
            >>> from pydantic import ValidationError

            >>> designer = ProtocolDesigner()

            1. Valid power in task.futility

            >>> task = GST.TaskSpec(
            ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="A", treatment_arm_name="B"),
            ...     response_type=GST.ResponseType.BINARY,
            ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
            ...     futility=GST.FutilityRequirement(power=0.8),
            ...     hypotheses=GST.HypothesisSpec(
            ...         h_null_description="null",
            ...         h_alt_description="alt",
            ...         test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
            ...         target_effect=GST.BinaryEffectSize(proportions={"A": 0.1, "B": 0.2}),
            ...     ),
            ... )
            >>> params = {"looks": 2, "spending_function": "obrien_fleming"}
            >>> method = designer.method_from_task_spec(task, params)
            >>> method is not None
            True

            2. Valid power in params

            >>> task_no_fut = GST.TaskSpec(
            ...     arms=ES3_BASE.TwoArmComparison(control_arm_name="A", treatment_arm_name="B"),
            ...     response_type=GST.ResponseType.BINARY,
            ...     efficacy=GST.EfficacyRequirement(alpha=0.05),
            ...     futility=None,
            ...     hypotheses=GST.HypothesisSpec(
            ...         h_null_description="null",
            ...         h_alt_description="alt",
            ...         test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
            ...         target_effect=GST.BinaryEffectSize(proportions={"A": 0.1, "B": 0.2}),
            ...     ),
            ... )
            >>> params_with_power = {
            ...     "looks": 2,
            ...     "power": 0.8,
            ...     "spending_function": "obrien_fleming",
            ... }
            >>> method = designer.method_from_task_spec(task_no_fut, params_with_power)
            >>> method is not None
            True

            3. Invalid missing in both

            >>> params_no_power = {"looks": 2, "spending_function": "obrien_fleming"}
            >>> try:
            ...     designer.method_from_task_spec(task_no_fut, params_no_power)
            ... except ValueError as e:
            ...     print(e)
            Statistical 'power' must be provided either in TaskSpec.futility or in 'params' to determine maximum sample size.

            4. Invalid missing looks (Pydantic ValidationError)

            >>> params_no_looks = {"power": 0.8}
            >>> try:
            ...     designer.method_from_task_spec(task_no_fut, params_no_looks)
            ... except ValidationError:
            ...     print("Caught ValidationError")
            Caught ValidationError

            5. Invalid looks = 0 (Pydantic ValidationError)

            >>> params_zero_looks = {"looks": 0, "power": 0.8}
            >>> try:
            ...     designer.method_from_task_spec(task_no_fut, params_zero_looks)
            ... except ValidationError:
            ...     print("Caught ValidationError")
            Caught ValidationError
        """
        # Validate and extract parameters using Pydantic
        v_params = MethodDesignParams.model_validate(params)

        efficacy = task.efficacy
        if not efficacy:
            raise ValueError("Task is missing efficacy requirements.")

        hypotheses = task.hypotheses
        if not hypotheses:
            raise ValueError("Task is missing hypotheses.")

        if not isinstance(hypotheses.target_effect, GST.BinaryEffectSize):
            raise ValueError("Task must have BinaryEffectSize for Binomial Design")

        props = hypotheses.target_effect.proportions
        arms = task.arms
        if not isinstance(arms, ES3_BASE.TwoArmComparison):
            raise ValueError(
                f"method_from_task_spec (Binomial) requires TwoArmComparison, but got {type(arms).__name__}."
            )

        p_c = float(props[arms.control_arm_name])
        p_t = float(props[arms.treatment_arm_name])
        futility = task.futility
        target_power = futility.power if futility else v_params.power
        if target_power is None:
            raise ValueError(
                "Statistical 'power' must be provided either in TaskSpec.futility or in 'params' to determine maximum sample size."
            )

        if v_params.spending_function is None:
            raise ValueError(
                "'spending_function' must be provided in 'params' (e.g. 'obrien_fleming')."
            )

        method_spec, _ = self.design_gs_binomial(
            alpha=efficacy.alpha,
            power=float(target_power),
            p_control=p_c,
            p_treatment=p_t,
            looks=v_params.looks,
            spending_function=v_params.spending_function,
            scheduling=np.linspace(1 / v_params.looks, 1.0, v_params.looks),
            spending_params=v_params.spending_params,
            futility=futility is not None,
            futility_binding=bool(futility.binding) if futility else False,
            tails=1,  # Default to 1-sided for this template logic
            rng_seed=self._model.config.rng_seed if self._model else None,
        )

        return method_spec

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

        boundaries, _ = model.solve_boundaries(efficacy_spending=spending_fn, drift=1.0)
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
                arms=ES3_BASE.TwoArmComparison(
                    control_arm_name="control",
                    treatment_arm_name="treatment",
                ),
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
            Converted to standardized effect: theta = ``|log(HR)|`` / 2.

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
                arms=ES3_BASE.TwoArmComparison(
                    control_arm_name="control",
                    treatment_arm_name="treatment",
                ),
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
