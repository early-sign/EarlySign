"""
Group Sequential Trial Design API.

This module provides a programmatic interface for group sequential trial design,
allowing you to compute designs, boundaries, and operating characteristics without
using the interactive UI.

The API provides clear, descriptive method names:
- compute_design_with_fixed_timing: Compute boundaries for predetermined times
- compute_design_with_target_power: Find required sample size for target power
- optimize_expected_sample_size: Minimize ASN given constraints
- search_optimal_design_parameters: Search over design space
- find_minimum_detectable_effect: Find MDE given budget constraint

Examples
--------
>>> from earlysign.stats.design.gst.api import GSTDesign
>>>
>>> # Fixed timing design
>>> design_api = GSTDesign()
>>> result = design_api.compute_design_with_fixed_timing(
...     alpha=0.025,
...     power=0.90,
...     n_analyses=3,
...     p_control=0.10,
...     effect_size=0.02,
...     info_times=[0.33, 0.67, 1.0],
...     n_per_analysis=500
... )  # doctest: +SKIP
>>>
>>> # Get boundaries
>>> boundaries = result.lab.boundaries  # doctest: +SKIP
>>>
>>> # Run simulations for power
>>> result.lab.run_simulations()  # doctest: +SKIP
>>> power_summary = result.lab.get_power_summary()  # doctest: +SKIP
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.objectives import (
    BalancedDesign,
    DesignObjective,
    MaximizePower,
    MinimizeASN,
)
from earlysign.stats.applications.design.group_sequential.initial_design.workflows.optimize_timing.optimizer import (
    DesignOptimizer,
)
from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
from earlysign.stats.design.gst.common.lab import DesignLab
from earlysign.stats.design.gst.common.types import InformationSpacing, SpendingFunction


@dataclass
class DesignResult:
    """Result container for a group sequential design.

    Attributes
    ----------
    spec : ProportionsDesignSpec
        The design specification used.
    lab : DesignLab
        The design lab with computed boundaries and simulation results.
    extra : dict
        Additional metadata or results.
    """

    spec: ProportionsDesignSpec
    lab: DesignLab
    extra: dict[str, Any]


class GSTDesign:
    """Programmatic API for group sequential trial design.

    This class provides methods to compute group sequential designs,
    boundaries, and operating characteristics without the interactive UI.
    All methods return a DesignResult containing the spec and lab with
    computed boundaries.

    Examples
    --------
    >>> design_api = GSTDesign()  # doctest: +SKIP
    """

    def __init__(self) -> None:
        """Initialize the GST Design API."""
        pass

    def compute_design_with_fixed_timing(
        self,
        *,
        alpha: float = 0.025,
        power: float = 0.90,
        n_analyses: int = 3,
        spending_func: Literal["obrien_fleming", "pocock", "hsd"] = "obrien_fleming",
        p_control: float = 0.10,
        effect_size: float = 0.02,
        info_times: Sequence[float] = (0.33, 0.67, 1.0),
        n_per_analysis: int = 500,
    ) -> DesignResult:
        """Compute boundaries for fixed information times.

        This mode is useful when you know exactly when interim analyses will
        occur (e.g., at predetermined calendar dates or patient counts).

        Parameters
        ----------
        alpha : float, default=0.025
            One-sided significance level.
        power : float, default=0.90
            Target power (1-β).
        n_analyses : int, default=3
            Number of interim analyses.
        spending_func : {"obrien_fleming", "pocock", "hsd"}, default="obrien_fleming"
            Alpha spending function.
        p_control : float, default=0.10
            Control group proportion (for proportions endpoint).
        effect_size : float, default=0.02
            Treatment effect (absolute difference for proportions).
        info_times : Sequence[float], default=(0.33, 0.67, 1.0)
            Information fractions at each analysis.
        n_per_analysis : int, default=500
            Sample size per group at each analysis.

        Returns
        -------
        DesignResult
            Design result with computed boundaries.

        Examples
        --------
        >>> api = GSTDesign()  # doctest: +SKIP
        >>> result = api.compute_design_with_fixed_timing(
        ...     alpha=0.025,
        ...     power=0.90,
        ...     info_times=[0.5, 1.0],
        ...     n_per_analysis=1000
        ... )  # doctest: +SKIP
        """
        spec = ProportionsDesignSpec()
        spec.test.alpha = alpha
        spec.test.power = power
        spec.sequential.n_analyses = n_analyses
        spec.boundary.spending_function = SpendingFunction(spending_func)
        spec.effect.p_control = p_control
        spec.effect.effect_size = effect_size
        spec.sequential.info_times = list(info_times)
        spec.sequential.info_spacing = InformationSpacing.CUSTOM
        spec.sample_size.n_per_analysis = n_per_analysis

        lab = DesignLab(spec)
        lab.compute_boundaries()

        return DesignResult(spec=spec, lab=lab, extra={})

    def compute_design_with_target_power(
        self,
        *,
        alpha: float = 0.025,
        target_power: float = 0.90,
        n_analyses: int = 3,
        spending_func: Literal["obrien_fleming", "pocock", "hsd"] = "obrien_fleming",
        p_control: float = 0.10,
        effect_size: float = 0.02,
    ) -> DesignResult:
        """Compute required sample size for target power.

        This mode finds the minimum sample size needed to achieve the target
        power under the specified effect size and spending function.

        Parameters
        ----------
        alpha : float, default=0.025
            One-sided significance level.
        target_power : float, default=0.90
            Target power (1-β) to achieve.
        n_analyses : int, default=3
            Number of interim analyses.
        spending_func : {"obrien_fleming", "pocock", "hsd"}, default="obrien_fleming"
            Alpha spending function.
        p_control : float, default=0.10
            Control group proportion.
        effect_size : float, default=0.02
            Treatment effect (absolute difference).

        Returns
        -------
        DesignResult
            Design result with computed boundaries and required sample size.

        Examples
        --------
        >>> api = GSTDesign()  # doctest: +SKIP
        >>> result = api.compute_design_with_target_power(
        ...     target_power=0.85,
        ...     effect_size=0.03
        ... )  # doctest: +SKIP
        >>> print(result.spec.sample_size.n_per_analysis)  # doctest: +SKIP
        """
        spec = ProportionsDesignSpec()
        spec.test.alpha = alpha
        spec.test.power = target_power
        spec.sequential.n_analyses = n_analyses
        spec.boundary.spending_function = SpendingFunction(spending_func)
        spec.effect.p_control = p_control
        spec.effect.effect_size = effect_size

        lab = DesignLab(spec)
        lab.compute_boundaries()

        return DesignResult(spec=spec, lab=lab, extra={})

    def optimize_expected_sample_size(
        self,
        *,
        alpha: float = 0.025,
        target_power: float = 0.90,
        n_analyses: int = 3,
        spending_func: Literal["obrien_fleming", "pocock", "hsd"] = "obrien_fleming",
        p_control: float = 0.10,
        effect_size: float = 0.02,
        max_n_total: int = 3000,
    ) -> DesignResult:
        """Optimize information times to minimize expected sample number (ASN).

        Given a maximum total sample size constraint and target power, this mode
        finds the optimal information timing that minimizes the expected sample
        size under the alternative hypothesis.

        Returns
        -------
        DesignResult
            Design result with optimized information times and boundaries.
        """
        spec = ProportionsDesignSpec()
        spec.test.alpha = alpha
        spec.test.power = target_power
        spec.sequential.n_analyses = n_analyses
        spec.boundary.spending_function = SpendingFunction(spending_func)
        spec.effect.p_control = p_control
        spec.effect.effect_size = effect_size

        # Run optimization
        objective: DesignObjective = MinimizeASN(
            planned_max_n=max_n_total,
            target_power=target_power,
        )
        optimizer = DesignOptimizer(spec, objective)
        optimal_times = optimizer.optimize_info_times(n_analyses)

        # Update spec with optimal times
        spec.sequential.info_times = optimal_times.tolist()
        spec.sequential.info_spacing = InformationSpacing.CUSTOM

        # Set n_per_analysis based on max_n_total
        n_per_group_total = max_n_total // 2
        spec.sample_size.n_per_analysis = n_per_group_total // n_analyses

        lab = DesignLab(spec)
        lab.compute_boundaries()

        return DesignResult(
            spec=spec,
            lab=lab,
            extra={"optimal_info_times": optimal_times.tolist()},
        )

    def search_optimal_design_parameters(
        self,
        *,
        alpha: float = 0.025,
        power: float = 0.90,
        n_analyses: int = 3,
        spending_func: Literal["obrien_fleming", "pocock", "hsd"] = "obrien_fleming",
        p_control: float = 0.10,
        effect_size: float = 0.02,
        criterion: Literal[
            "minimize_asn", "maximize_power", "balanced"
        ] = "minimize_asn",
        alpha_range: tuple[float, float] = (0.01, 0.05),
        k_range: tuple[int, int] = (2, 6),
    ) -> DesignResult:
        """Search for optimal design across multiple parameters.

        This mode performs a comprehensive search over design space,
        exploring different combinations of alpha, number of analyses,
        and information timing to find the best design according to
        the specified criterion.

        Parameters
        ----------
        alpha : float, default=0.025
            Initial alpha value (will be varied in search).
        power : float, default=0.90
            Target power.
        n_analyses : int, default=3
            Initial number of analyses (will be varied in search).
        spending_func : {"obrien_fleming", "pocock", "hsd"}, default="obrien_fleming"
            Alpha spending function.
        p_control : float, default=0.10
            Control group proportion.
        effect_size : float, default=0.02
            Treatment effect (absolute difference).
        criterion : {"minimize_asn", "maximize_power", "balanced"}, default="minimize_asn"
            Optimization criterion.
        alpha_range : tuple[float, float], default=(0.01, 0.05)
            Range of alpha values to search.
        k_range : tuple[int, int], default=(2, 6)
            Range of number of analyses to search.

        Returns
        -------
        DesignResult
            Design result with optimal parameters.

        Examples
        --------
        >>> api = GSTDesign()  # doctest: +SKIP
        >>> result = api.search_optimal_design_parameters(
        ...     criterion="minimize_asn",
        ...     alpha_range=(0.01, 0.05),
        ...     k_range=(2, 5)
        ... )  # doctest: +SKIP
        """
        spec = ProportionsDesignSpec()
        spec.test.alpha = alpha
        spec.test.power = power
        spec.sequential.n_analyses = n_analyses
        spec.boundary.spending_function = SpendingFunction(spending_func)
        spec.effect.p_control = p_control
        spec.effect.effect_size = effect_size

        # Choose objective
        if criterion == "minimize_asn":
            objective_inst: DesignObjective = MinimizeASN(
                planned_max_n=10000, target_power=power
            )
        elif criterion == "maximize_power":
            objective_inst = MaximizePower(planned_max_n=10000)
        else:
            objective_inst = BalancedDesign()

        optimizer = DesignOptimizer(spec, objective_inst)
        optimized_spec = optimizer.optimize_comprehensive()

        lab = DesignLab(optimized_spec)
        lab.compute_boundaries()

        from typing import cast

        return DesignResult(
            spec=cast(ProportionsDesignSpec, optimized_spec),
            lab=lab,
            extra={
                "criterion": criterion,
                "alpha_range": alpha_range,
                "k_range": k_range,
            },
        )

    def find_minimum_detectable_effect(
        self,
        *,
        alpha: float = 0.025,
        power: float = 0.90,
        n_analyses: int = 3,
        spending_func: Literal["obrien_fleming", "pocock", "hsd"] = "obrien_fleming",
        p_control: float = 0.10,
        n_max: int = 3000,
        mde_search_range: tuple[float, float] = (0.001, 0.10),
        tolerance: float = 0.0001,
        max_iterations: int = 20,
    ) -> DesignResult:
        """Find minimum detectable effect given maximum sample size.

        This mode performs a binary search to find the smallest effect size
        that can be detected with the specified power, given a maximum
        sample size constraint.

        Parameters
        ----------
        alpha : float, default=0.025
            One-sided significance level.
        power : float, default=0.90
            Target power (1-β).
        n_analyses : int, default=3
            Number of interim analyses.
        spending_func : {"obrien_fleming", "pocock", "hsd"}, default="obrien_fleming"
            Alpha spending function.
        p_control : float, default=0.10
            Control group proportion.
        n_max : int, default=3000
            Maximum total sample size constraint.
        mde_search_range : tuple[float, float], default=(0.001, 0.10)
            Range of effect sizes to search (min, max).
        tolerance : float, default=0.0001
            Convergence tolerance for binary search.
        max_iterations : int, default=20
            Maximum number of search iterations.

        Returns
        -------
        DesignResult
            Design result with minimum detectable effect.
            The found MDE is stored in `spec.effect.effect_size`.

        Examples
        --------
        >>> api = GSTDesign()  # doctest: +SKIP
        >>> result = api.find_minimum_detectable_effect(
        ...     n_max=2000,
        ...     power=0.85
        ... )  # doctest: +SKIP
        >>> print(f"MDE: {result.spec.effect.effect_size:.5f}")  # doctest: +SKIP
        """
        # Binary search for minimum MDE
        mde_low, mde_high = mde_search_range
        iteration = 0
        best_mde = None
        best_lab = None
        best_spec = None

        while iteration < max_iterations and (mde_high - mde_low) > tolerance:
            mde_test = (mde_low + mde_high) / 2.0

            spec = ProportionsDesignSpec()
            spec.test.alpha = alpha
            spec.test.power = power
            spec.sequential.n_analyses = n_analyses
            spec.boundary.spending_function = SpendingFunction(spending_func)
            spec.effect.p_control = p_control
            spec.effect.effect_size = mde_test
            spec.sample_size.n_per_analysis = n_max // n_analyses

            try:
                lab = DesignLab(spec)
                lab.compute_boundaries()
                lab.run_simulations()

                # Get achieved power
                power_summary = lab.get_power_summary()
                achieved_power_str = str(power_summary.get("Overall Power (H1)", "0.0"))
                achieved_power = (
                    float(achieved_power_str.replace("%", "").strip()) / 100.0
                )

                # Check feasibility
                if lab.boundaries is not None:
                    info_fractions = lab.boundaries["info_times"]
                    actual_max_n = (
                        info_fractions[-1]
                        * spec.sample_size.n_per_analysis
                        * n_analyses
                    )
                else:
                    actual_max_n = float("inf")

                feasible = actual_max_n <= n_max

                # Binary search logic
                if feasible and achieved_power >= power:
                    # Can achieve power with this MDE, try smaller
                    mde_high = mde_test
                    best_mde = mde_test
                    best_lab = lab
                    best_spec = spec
                else:
                    # Cannot achieve power, need larger MDE
                    mde_low = mde_test

            except Exception:
                # If computation fails, assume not feasible
                mde_low = mde_test

            iteration += 1

        if best_mde is None or best_lab is None or best_spec is None:
            raise ValueError(
                f"Could not find feasible MDE in range {mde_search_range} "
                f"with n_max={n_max} and power={power}"
            )

        return DesignResult(
            spec=best_spec,
            lab=best_lab,
            extra={
                "mde": best_mde,
                "iterations": iteration,
                "n_max": n_max,
            },
        )

    def optimize_expected_sample_size_with_n_analyses(
        self,
        *,
        alpha: float = 0.025,
        target_power: float = 0.90,
        min_n_analyses: int = 2,
        max_n_analyses: int,
        spending_func: Literal["obrien_fleming", "pocock", "hsd"] = "obrien_fleming",
        p_control: float = 0.10,
        effect_size: float = 0.02,
        max_n_total: int = 3000,
        n_samples: int = 100,
    ) -> DesignResult:
        """Optimize both number of analyses and info times by random sampling.

        Returns
        -------
        DesignResult
            Design result with optimized n_analyses and info_times.
        """
        spec = ProportionsDesignSpec()
        spec.test.alpha = alpha
        spec.test.power = target_power
        spec.boundary.spending_function = SpendingFunction(spending_func)
        spec.effect.p_control = p_control
        spec.effect.effect_size = effect_size

        objective: DesignObjective = MinimizeASN(
            planned_max_n=max_n_total,
            target_power=target_power,
        )
        optimizer = DesignOptimizer(spec, objective)
        n_analyses, info_times = optimizer.optimize_info_times_and_n_analyses(
            max_k=max_n_analyses, min_k=min_n_analyses, n_samples=n_samples
        )
        spec.sequential.n_analyses = n_analyses
        spec.sequential.info_times = info_times.tolist()
        spec.sequential.info_spacing = InformationSpacing.CUSTOM
        n_per_group_total = max_n_total // 2
        spec.sample_size.n_per_analysis = n_per_group_total // n_analyses
        lab = DesignLab(spec)
        lab.compute_boundaries()
        return DesignResult(
            spec=spec,
            lab=lab,
            extra={
                "optimal_info_times": info_times.tolist(),
                "optimal_n_analyses": n_analyses,
            },
        )
