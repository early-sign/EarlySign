from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Literal, Optional, Self, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

import earlysign.schema.ES3.Base as ES3_BASE
from earlysign.builtin.group_sequential import schema as GST
from earlysign.builtin.group_sequential.adapters import (
    binomial,
    continuous,
    protocol as adapter,
)
from earlysign.builtin.group_sequential.core.model import (
    CanonicalJointModel,
    Config,
    NumericalIntegrationConfig,
    SimulationConfig,
)
from earlysign.builtin.group_sequential.core.spending import (
    SpendingFunction,
    SpendingFunctionFactory,
)

# Internal numerical safety limits for GSD solving.
# A small number of iterations (2-3) typically suffice for convergence
# to double precision due to the smooth, monotone nature of the
# canonical joint distribution functions.
_MAX_GSD_SOLVE_ITERATIONS = 5
_GSD_SOLVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GSDDesign:
    """Standardized result of a group sequential design solve.

    Attributes:
        info_times: Information fractions (t_1, ..., t_K).
        upper_boundaries: Efficacy boundaries (Z-scale).
        lower_boundaries: Futility boundaries (Z-scale), or None.
        drift: Standardized drift (theta_1) corresponding to target power.
        power: Target statistical power.
        alpha: Target Type I error rate.
        inflation_factor: Ratio of (drift / drift_fixed)**2.
    """

    info_times: NDArray[np.float64]
    upper_boundaries: NDArray[np.float64]
    lower_boundaries: Optional[NDArray[np.float64]]
    drift: float
    power: float
    alpha: float

    @property
    def inflation_factor(self) -> float:
        """The ratio of maximum sample size relative to a fixed design."""
        from scipy.stats import norm

        # drift_fixed = z_{1-alpha} + z_{1-beta}
        z_alpha = norm.ppf(1 - self.alpha)
        z_beta = norm.ppf(self.power)
        drift_fixed = z_alpha + z_beta
        return float((self.drift / drift_fixed) ** 2)


class MethodDesignParams(BaseModel):
    """Internal model for design parameter validation."""

    looks: int = Field(..., gt=0)
    spending_function: Optional[str] = None
    spending_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    power: Optional[float] = None
    ssr_method: Optional[str] = None
    method: Literal["simulation", "numerical_integration"] = "simulation"
    method_config: Optional[Union[SimulationConfig, NumericalIntegrationConfig]] = None

    class Config:
        arbitrary_types_allowed = True


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
            - 'model_params': Dict containing 'rng_seed', 'n_sims', etc.

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
                    rng_seed=model_params.get("rng_seed", 42),
                    n_sims=model_params.get("n_sims", 2000),
                )
            )
        return cls(model=model)

    def solve_design(
        self,
        info_times: NDArray[np.float64],
        alpha: float,
        power: float,
        efficacy_spending: Optional[SpendingFunction] = None,
        futility_spending: Optional[SpendingFunction] = None,
        efficacy_binding: bool = True,
        futility_binding: bool = False,
        tails: int = 1,
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
        rng_seed: int = 42,
    ) -> GSDDesign:
        """Pure statistical solver for group sequential design.

        Replicates the core 'gsDesign' logic: solving for boundaries and the
        drift necessary to achieve target power.

        Examples:
            >>> import numpy as np
            >>> from earlysign.builtin.group_sequential.design.protocol_design import ProtocolDesigner
            >>> from earlysign.builtin.group_sequential.core.spending import HwangShihDeCaniSpending
            >>>
            >>> # Replicate gsDesign manual default (CLIP/CAPTURE trial)
            >>> # alpha=0.025, power=0.9, 3-look HSD(gamma=-4/-2)
            >>> designer = ProtocolDesigner()
            >>> t = np.linspace(1/3, 1.0, 3)
            >>> eff_sf = HwangShihDeCaniSpending(budget=0.025, gamma=-4.0)
            >>> fut_sf = HwangShihDeCaniSpending(budget=0.10, gamma=-2.0)
            >>>
            >>> design = designer.solve_design(
            ...     info_times=t, alpha=0.025, power=0.9,
            ...     efficacy_spending=eff_sf, futility_spending=fut_sf,
            ...     efficacy_binding=True, futility_binding=False,
            ...     method="simulation",
            ...     rng_seed=42
            ... )
            >>> np.round(design.upper_boundaries, 2)
            array([2.96, 2.5 , 1.97])
            >>> np.round(design.lower_boundaries, 2)
            array([-0.31,  0.9 ,  1.97])
            >>> round(design.drift, 3)
            3.296
            >>> round(design.inflation_factor, 2)
            1.03
        """
        # 1. Initial Guess for Drift (Fixed Design)
        from scipy.stats import norm

        # z_{1-alpha} + z_{1-beta}
        z_alpha = norm.ppf(1 - alpha)
        z_beta = norm.ppf(power)
        current_drift = z_alpha + z_beta

        # 2. Iterate to find stable drift and boundaries
        #
        # A circular dependency exists between drift and boundaries:
        # 1. Efficacy boundaries depend on alpha spending (drift = 0).
        # 2. Futility boundaries depend on beta spending (power under H_a).
        # 3. Required drift depends on both efficacy and futility boundaries.
        #
        # This is a standard numerical approach in GSD software (gsDesign, Stata).
        # Reference: Jennison & Turnbull (2000), Sections 7.3 and 8.4.
        upper = None
        lower = None

        config = Config(
            info_times=info_times,
            rng_seed=rng_seed,
            efficacy_binding=efficacy_binding,
            futility_binding=futility_binding,
            tails=tails,
        )
        model = CanonicalJointModel(config)

        for i in range(_MAX_GSD_SOLVE_ITERATIONS):
            prev_drift = current_drift
            upper, lower = model.solve_boundaries(
                efficacy_spending=efficacy_spending,
                futility_spending=futility_spending,
                drift=current_drift,
                method=method,
                method_config=method_config,
            )
            current_drift = model.solve_drift(
                info_times=info_times.tolist(),
                boundaries=upper.tolist() if upper is not None else [],
                target_power=power,
                futility_boundaries=(
                    lower.tolist() if futility_binding and lower is not None else None
                ),
                tails=tails,
                method=method,
                method_config=method_config,
                bracket=(current_drift * 0.8, current_drift * 1.2),
            )

            # Check for convergence on the drift parameter
            if i > 0 and abs(current_drift - prev_drift) < _GSD_SOLVE_TOLERANCE:
                break

        # 3. Final Look Unification
        # In standard GSD design, the final efficacy and futility boundaries
        # should meet. Under numerical solving with separate spending targets,
        # they might differ by a tiny amount. We unify them for consistency.
        if upper is not None and lower is not None:
            avg = (upper[-1] + lower[-1]) / 2.0
            if abs(upper[-1] - lower[-1]) < 0.1:
                upper[-1] = avg
                lower[-1] = avg

        return GSDDesign(
            info_times=info_times,
            upper_boundaries=upper if upper is not None else np.array([]),
            lower_boundaries=lower,
            drift=float(current_drift),
            power=float(power),
            alpha=float(alpha),
        )

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
        rng_seed: int = 42,
        allocation_ratios: Optional[Dict[str, float]] = None,
        control_arm_name: str = "control",
        treatment_arm_name: str = "treatment",
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
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
            rng_seed: Random seed for model.
            allocation_ratios: Ratios of treatments to control (defaults to {treatment: 1.0}).
            control_arm_name: Name of the control arm.
            treatment_arm_name: Name of the primary treatment arm (used for drift solving).
            method: Method for boundary solving ('simulation' or 'numerical_integration').
            method_config: Configuration object.

        Returns:
            A tuple of (MethodSpec, n_max). n_max is total (int).
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
            info_times = adapter.get_info_times(GST.EquidistantSchedule(n_looks=looks))
        elif scheduling == "asn_minimizer":
            if looks == 1:
                info_times = np.array([1.0])
            else:
                from earlysign.builtin.group_sequential.design.schedule_optimization import (
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
                    method=method,
                    method_config=method_config,
                )
                drift_target = proxy_model.solve_drift(
                    proxy_times.tolist(),
                    u_prox.tolist() if u_prox is not None else [],
                    target_power=power,
                    futility_boundaries=l_prox.tolist() if l_prox is not None else None,
                    tails=tails,
                    method=method,
                    method_config=method_config,
                )

                # 1b. Run Schedule Optimization
                res = optimize_schedule(
                    k_looks=looks,
                    efficacy_spending=sf_eff,
                    futility_spending=sf_fut,
                    alpha=alpha,
                    drift=drift_target,
                    tails=tails,
                    method=method,
                    method_config=method_config,
                    seed=rng_seed,
                )
                info_times = res.schedule
        else:
            raise ValueError(f"Invalid scheduling option: {scheduling}")

        # 2. Solve High-level Design (Statistical core)
        design = self.solve_design(
            info_times=info_times,
            alpha=alpha,
            power=power,
            efficacy_spending=sf_eff,
            futility_spending=sf_fut,
            efficacy_binding=True,
            futility_binding=futility_binding,
            tails=tails,
            method=method,
            method_config=method_config,
            rng_seed=rng_seed,
        )

        # 3. Calculate Sample Size (n_max)
        n_max = binomial.calculate_n_max(
            drift=design.drift,
            p_control=p_control,
            p_treatment=p_treatment,
            allocation_ratios=allocation_ratios,
            control_arm_name=control_arm_name,
            treatment_arm_name=treatment_arm_name,
        )
        total_n = sum(n_max.values())

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

        return method_spec, total_n

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
        arm_names: list[str] = ["control", "treatment"],
        rng_seed: int = 42,
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
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
            arm_names: List of arm names.
            rng_seed: Random seed.
            method: Computation method.
            method_config: Configuration object.

        Returns:
            A tuple of (MethodSpec, n_max).
        """
        # 1. Setup Model & Strategy
        info_times = adapter.get_info_times(GST.EquidistantSchedule(n_looks=looks))
        model = CanonicalJointModel(Config(info_times=info_times, rng_seed=rng_seed))

        shape_params = (
            {"delta_wt": wang_tsiatis_delta} if type == "wang_tsiatis" else {}
        )

        # 2. Solve High-level Design (Statistical core)

        # Mapping to internal policies to reuse their solver logic via solve_design
        # Mapping to internal policies for MethodSpec output
        if type == "pocock":
            strategy_cls: Any = GST.PocockStrategy
        elif type == "obrien_fleming":
            strategy_cls = GST.OBrienFlemingStrategy
        elif type == "wang_tsiatis":
            strategy_cls = GST.WangTsiatisStrategy
        else:
            raise ValueError(f"Unknown classic design type: {type}")

        # We can solve these via a temporary design solve.
        # Note: solve_design currently expects SpendingFunction objects.
        # Classic designs are "shape-based" so let's use the manual solve logic
        # but standardized.
        c_val = model.solve_boundary_constant(
            info_times=info_times.tolist(),
            alpha=alpha,
            shape_type=type,
            tails=tails,
            shape_params=shape_params,
            method=method,
            method_config=method_config,
        )

        if type == "pocock":
            bound_shape = np.ones(looks)
        elif type == "obrien_fleming":
            bound_shape = 1.0 / np.sqrt(info_times)
        elif type == "wang_tsiatis":
            bound_shape = info_times ** (wang_tsiatis_delta - 0.5)

        boundaries = c_val * bound_shape

        # Solve Drift
        drift = model.solve_drift(
            info_times.tolist(),
            boundaries.tolist(),
            target_power=power,
            tails=tails,
            method=method,
            method_config=method_config,
        )

        # 4. Map to Sample Size
        n_arms = len(arm_names)
        if p_control is not None:
            # Binomial
            n_max_dict = binomial.calculate_n_max(
                drift=drift,
                p_control=p_control,
                p_treatment=p_control + delta,
                control_arm_name=arm_names[0],
                treatment_arm_name=arm_names[1] if n_arms > 1 else arm_names[0],
            )
            timer_unit = GST.Unit.INDIVIDUALS
            stat_spec: Any = (
                GST.TwoArmBinomialZ(variance_estimation=GST.VarianceEstimation.POOLED)
                if n_arms == 2
                else GST.OneArmBinomialZ(
                    variance_source=GST.VarianceSource.NULL_HYPOTHESIS
                )
            )
        elif sigma is not None:
            # Continuous
            n_max_dict = continuous.calculate_n_max(
                drift=drift,
                delta=delta,
                sigma=sigma,
                n_arms=n_arms,
                arm_names=arm_names,
            )
            timer_unit = GST.Unit.INDIVIDUALS
            stat_spec = (
                GST.TwoArmContinuousZ(
                    information_unit="fisher_information",
                    variance=GST.TwoArmEstimatedVariance(
                        kind="estimated", method=GST.MethodModel.POOLED
                    ),
                )
                if n_arms == 2
                else GST.OneArmContinuousZ(
                    variance=GST.OneArmEstimatedVariance(kind="estimated")
                )
            )
        else:
            raise ValueError("Must provide either p_control or sigma.")

        n_total_calc = sum(n_max_dict.values())

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
                timer=GST.SampleSizeTimer(unit=timer_unit, max_sample_size=n_max_dict),
                schedule=GST.FixedSchedule(analyses=info_times.tolist()),
            ),
        )

        return method_spec, n_total_calc

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
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
        rng_seed: int = 42,
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
            method: Method for boundary solving ('simulation' or 'numerical_integration').
            method_config: Configuration object.

        Returns:
            A fully populated GST.Protocol representing the planned design.
        """
        info_times = np.linspace(1 / k, 1.0, k)
        p_treatment = p_control + delta

        spending_family = spending_fn.name if spending_fn else "obrien_fleming"
        spending_params = spending_fn.params if spending_fn else None

        method_spec, _ = self.design_gs_binomial(
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
            method=method,
            method_config=method_config,
            rng_seed=rng_seed,
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

    def plan_binomial_unequal(
        self,
        alpha: float,
        power: float,
        p_control: float,
        p_treatment: float,
        k: int,
        allocation_ratios: Optional[Dict[str, float]] = None,
        spending_fn: Optional[SpendingFunction] = None,
        side: int = 1,
        control_arm_name: str = "control",
        treatment_arm_name: str = "treatment",
        rng_seed: int = 42,
    ) -> GST.Protocol:
        """Plans a binomial design with unequal allocation.

        Args:
            alpha: Type I error rate.
            power: Statistical power.
            p_control: Baseline proportion.
            p_treatment: Target proportion.
            k: Number of looks.
            allocation_ratios: Use dict for unequal allocation (e.g. {'treatment': 2.0}).
            spending_fn: Spending function.
            side: 1 or 2 sided.
            control_arm_name: Name of the control arm.
            treatment_arm_name: Name of the primary treatment arm.

        Returns:
            A fully populated GST.Protocol.
        """
        info_times = np.linspace(1 / k, 1.0, k)
        spending_family = spending_fn.name if spending_fn else "obrien_fleming"
        spending_params = spending_fn.params if spending_fn else None

        method_spec, _ = self.design_gs_binomial(
            alpha=alpha,
            power=power,
            p_control=p_control,
            p_treatment=p_treatment,
            looks=k,
            scheduling=info_times,
            spending_function=spending_family,
            spending_params=spending_params,
            futility=True,
            tails=side,
            rng_seed=rng_seed,
            allocation_ratios=allocation_ratios,
            control_arm_name=control_arm_name,
            treatment_arm_name=treatment_arm_name,
        )

        if allocation_ratios and len(allocation_ratios) > 1:
            arms: ES3_BASE.ArmStructure = ES3_BASE.MultiArmComparison(
                control_arm_name=control_arm_name,
                treatment_arm_names=list(allocation_ratios.keys()),
                allocation_ratios=allocation_ratios,
            )
            props = {control_arm_name: p_control}
            props.update({arm: p_treatment for arm in allocation_ratios.keys()})
        else:
            ratio = 1.0
            if allocation_ratios:
                ratio = list(allocation_ratios.values())[0]

            # For TwoArmComparison, we must provide allocation_ratios as a dict
            # Expected key is treatment_arm_name
            two_arm_ratios = {treatment_arm_name: ratio}

            arms = ES3_BASE.TwoArmComparison(
                control_arm_name=control_arm_name,
                treatment_arm_name=treatment_arm_name,
                allocation_ratios=two_arm_ratios,
            )
            props = {control_arm_name: p_control, treatment_arm_name: p_treatment}

        return GST.Protocol(
            name="Designed Protocol (Unequal)",
            task=GST.TaskSpec(
                kind="group_sequential",
                arms=arms,
                response_type=GST.ResponseType.BINARY,
                hypotheses=GST.HypothesisSpec(
                    h_null_description="Difference <= 0",
                    h_alt_description=f"Difference > {p_treatment - p_control:.4f}",
                    test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
                    target_effect=GST.BinaryEffectSize(proportions=props),
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
            >>> from earlysign.builtin.group_sequential.design.protocol_design import ProtocolDesigner
            >>> import earlysign.schema.ES3.Base as ES3_BASE
            >>> from earlysign.builtin.group_sequential import schema as GST
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

        allocation_ratios: Optional[Dict[str, float]] = None
        control_arm_name = "control"
        treatment_arm_name = "treatment"

        if isinstance(arms, ES3_BASE.TwoArmComparison):
            p_c = float(props[arms.control_arm_name])
            p_t = float(props[arms.treatment_arm_name])
            allocation_ratios = arms.allocation_ratios
            if not allocation_ratios:
                allocation_ratios = {arms.treatment_arm_name: 1.0}
            control_arm_name = arms.control_arm_name
            treatment_arm_name = arms.treatment_arm_name
        elif isinstance(arms, ES3_BASE.MultiArmComparison):
            control_arm_name = arms.control_arm_name
            treatment_arm_names = arms.treatment_arm_names
            # Pick first treatment arm as primary for drift solving if not specified or just use first
            treatment_arm_name = treatment_arm_names[0]
            p_c = float(props[control_arm_name])
            p_t = float(props[treatment_arm_name])
            allocation_ratios = arms.allocation_ratios
        else:
            raise ValueError(
                f"method_from_task_spec (Binomial) requires TwoArmComparison or MultiArmComparison, but got {type(arms).__name__}."
            )

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
            rng_seed=self._model.config.rng_seed if self._model else 42,
            allocation_ratios=allocation_ratios,
            control_arm_name=control_arm_name,
            treatment_arm_name=treatment_arm_name,
            method=v_params.method,
            method_config=v_params.method_config,
        )

        # 7. Attach Adaptation Spec if SSR method is requested
        if v_params.ssr_method:
            use_weighted = True
            if v_params.ssr_method == "hsiao_2019":
                use_weighted = False

            ssr_spec = GST.SampleSizeReestimationSpec(
                method=GST.Method.CONDITIONAL_POWER,
                target_power=float(target_power),
                n_range=[0, 1000000],  # Default wide range
                use_weighted_statistic=use_weighted,
            )

            method_spec.adaptation = GST.AdaptationSpec(
                sample_size_reestimation=ssr_spec
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
        method: Literal["simulation", "numerical_integration"] = "simulation",
        method_config: Optional[
            Union[SimulationConfig, NumericalIntegrationConfig]
        ] = None,
        rng_seed: int = 42,
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
        method : Literal["simulation", "numerical_integration"]
            Method for boundary solving.
        method_config : Optional[Union[SimulationConfig, NumericalIntegrationConfig]]
            Configuration object.

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

        # 2. Solve High-level Design
        design = self.solve_design(
            info_times=info_times,
            alpha=alpha,
            power=power,
            efficacy_spending=spending_fn,
            rng_seed=rng_seed,
            method=method,
            method_config=method_config,
        )

        # 3. Calculate Sample Size (n_max)
        drift = design.drift

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
                        spending_fn=GST.SpendingFunction(
                            family=spending_fn.name if spending_fn else "obrien_fleming"
                        ),
                        budget=alpha,
                        sided=GST.Sided.ONE,
                        statistical_model=GST.CanonicalGaussianModel(),
                    ),
                    timer=GST.SampleSizeTimer(
                        unit=GST.Unit.INDIVIDUALS,
                        max_sample_size={
                            "control": n_max // 2,
                            "treatment": n_max - (n_max // 2),
                        },
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
        rng_seed: int = 42,
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

        # 2. Solve High-level Design
        design = self.solve_design(
            info_times=info_times,
            alpha=alpha,
            power=power,
            efficacy_spending=spending_fn,
            rng_seed=rng_seed,
        )

        # 3. Calculate Sample Size (n_max)
        drift = design.drift

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
                        spending_fn=GST.SpendingFunction(
                            family=spending_fn.name if spending_fn else "obrien_fleming"
                        ),
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
