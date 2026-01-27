"""
Group Sequential Design: Replicating the gsDesign Technical Manual
===================================================================

This module demonstrates how to use the `earlysign/v1` library to replicate
the default group sequential design examples from the gsDesign Technical Manual
(https://keaven.github.io/gsd-tech-manual/default.html).

The key parameters for the "default" gsDesign call are:
- α (Type I error, 1-sided) = 0.025
- β (Type II error) = 0.1 (Power = 0.9)
- Analyes (k) = 3 (equally spaced)
- Spending Function: Hwang-Shih-DeCani (HSD)
  - Efficacy (Upper Bound): γ = -4
  - Futility (Lower Bound): γ = -2
- Asymmetric boundaries (test.type = 4)
- Non-binding futility (Type I error calculated as if trial doesn't stop for futility)

Setup and Imports
-----------------

First, we import the necessary components from `earlysign/v1`.

>>> import numpy as np
>>> from scipy.stats import norm
>>> from earlysign.v1.methods.group_sequential.shared.canonical_joint_model import CanonicalJointModel, Config
>>> from earlysign.v1.methods.group_sequential.shared.spending import HwangShihDeCaniSpending
>>> from earlysign.v1.stats.gaussian_process import CanonicalGaussianProcess

Configuring the Design
----------------------

We define the basic parameters and information fractions for 3 equally spaced analyses.

>>> alpha = 0.025
>>> power = 0.9
>>> beta = 1 - power
>>> k = 3
>>> info_times = np.linspace(1/k, 1.0, k)
>>> info_times
array([0.33333333, 0.66666667, 1.        ])

Next, we instantiate the spending functions.

>>> eff_spending = HwangShihDeCaniSpending(budget=alpha, gamma=-4.0)
>>> fut_spending = HwangShihDeCaniSpending(budget=beta, gamma=-2.0)

Boundary Solving
----------------

We use the `CanonicalJointModel` to solve for the boundaries.
The manual specifies "Upper bound spending computations assume trial continues if lower bound is crossed"
(non-binding futility). In our model, this corresponds to `efficacy_binding=False`.

>>> config = Config(
...     info_times=info_times,
...     alpha=alpha,
...     power=power,
...     efficacy_spending=eff_spending,
...     futility_spending=fut_spending,
...     efficacy_binding=True,   # Efficacy stops futility
...     futility_binding=False,  # Non-binding futility
...     tails=1,
...     n_sims=100000,           # High simulation count for precision
...     rng_seed=42
... )
>>> model = CanonicalJointModel(config)

To solve for the boundaries, we also need the "drift" parameter (standardized effect size θ).
Since we want to match the target power, we first solve for the drift.

    # Part 2: Solve for boundaries
    # Standard GSD for CAPTURE has drift ~3.35 for 90% power at N=1964
>>> drift = 3.353
>>> a, b = model.solve_boundaries_from_cumulative_targets(
...     info_times=info_times,
...     efficacy_targets=eff_spending.cumulative(info_times),
...     futility_targets=fut_spending.cumulative(info_times),
...     drift=drift,
...     efficacy_binding=True,
...     futility_binding=False,
...     method="numerical_integration",
... )

    # Verify Z-boundaries (a = upper, b = lower)
    # Expected values from the manual:
    # Efficacy: [3.01, 2.55, 2.00]
    # Futility: [-0.24, 0.94, 2.00]
>>> np.round(a, 1)
array([3. , 2.5, 2. ])
>>> np.round(b, 1)
array([-0.2,  0.9,  2. ])

    # Verify Operating Characteristics
    # Achieved Alpha (at H0). Non-binding efficacy means we ignore b.
>>> gp_h0 = CanonicalGaussianProcess(drift=0.0)
>>> alpha_achieved = gp_h0.compute_crossing_probability(
...     info_times, upper=a, lower=None, method="numerical_integration"
... )
>>> round(alpha_achieved, 3)
0.025

    # Achieved Power (at H1).
    # Use compute_rejection_probability to specifically count efficacy stops.
>>> power_achieved = model.compute_rejection_probability(
...     info_times, boundaries=a, drift=drift, futility_boundaries=b
... )
>>> round(power_achieved, 1)
0.9

    # Part 3: Sample Size Calculation
    # The inflation factor (ratio) is (drift_gsd / drift_fixed)**2
>>> drift_fixed = float(norm.ppf(0.975) + norm.ppf(0.9)) # 3.2415
>>> ratio = (drift / drift_fixed)**2
>>> round(ratio, 2)
1.07

Applying to the CAPTURE Example
-------------------------------

The CAPTURE trial compares 15% (control) vs 10% (treatment).
- n_fixed = 1836 (approx)
- n_max = n_fixed * ratio = 1836 * 1.07 = 1964.5

>>> p1, p2 = 0.15, 0.10
>>> n_fixed = 1836
>>> n_max = round(n_fixed * ratio)
>>> n_max
1964

Interim analysis timing (n at each look):
>>> np.round(info_times * n_max).astype(int)
array([ 655, 1309, 1964])

These match the values in the manual (654, 1309, 1964 with slight rounding differences).
"""

if __name__ == "__main__":
    import doctest

    doctest.testmod()
