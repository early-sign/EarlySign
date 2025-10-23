"""
Doctest-based tests for GST Operating Characteristics.

This module contains doctest examples that verify the correctness of
group sequential test (GST) operating characteristics calculations
for two-proportion tests.

Examples
--------
Basic GST operating characteristics with p0=0.2, effect_size=0.1:

>>> import numpy as np
>>> from earlysign.stats.design.gst.common.scenarios import (
...     TwoProportionsCalculator,
...     run_scenario_a,
...     run_scenario_b,
... )
>>> np.random.seed(42)
>>>
>>> # Setup parameters
>>> alpha = 0.05
>>> power = 0.8
>>> p0 = 0.2
>>> effect_size_target = 0.1
>>> p1_target = p0 + effect_size_target
>>>
>>> # Initialize calculator
>>> calculator = TwoProportionsCalculator(p_control=p0)
>>>
>>> # Calculate fixed design sample size
>>> n_per_group = calculator.calculate_sample_size(effect_size_target, alpha, power)
>>> n_total = n_per_group * 2
>>> print(f"Fixed design - Per group: {n_per_group}, Total: {n_total}")
Fixed design - Per group: 291, Total: 582
>>>
>>> # Define effect size range
>>> effect_sizes = np.linspace(0.02, 0.35, 50)
>>> n_interim_list = [1, 2]
>>>
>>> # Run Scenario A: Fixed Max N
>>> result_a = run_scenario_a(
...     calculator=calculator,
...     target_effect=effect_size_target,
...     effect_sizes=effect_sizes,
...     alpha=alpha,
...     power=power,
...     n_interim_list=n_interim_list,
... )
>>>
>>> # Verify Scenario A results structure
>>> result_a.fixed_n
582
>>> len(result_a.gst_results)
2
>>> list(result_a.gst_results.keys())
[1, 2]
>>>
>>> # Check GST with 1 interim (2 looks total)
>>> result_a.gst_results[1].n_looks
2
>>> result_a.gst_results[1].n_per_analysis
146
>>> len(result_a.gst_results[1].results)
50
>>>
>>> # Check GST with 2 interims (3 looks total)
>>> result_a.gst_results[2].n_looks
3
>>> result_a.gst_results[2].n_per_analysis
97
>>>
>>> # Verify power increases with effect size
>>> powers_1 = [r.power for r in result_a.gst_results[1].results]
>>> powers_1[0] < powers_1[10] < powers_1[20]
True
>>>
>>> # Run Scenario B: Fixed Power at target
>>> result_b = run_scenario_b(
...     calculator=calculator,
...     target_effect=effect_size_target,
...     effect_sizes=effect_sizes,
...     alpha=alpha,
...     power=power,
...     n_interim_list=n_interim_list,
...     inflation_factor=1.15,
... )
>>>
>>> # Verify Scenario B results structure
>>> result_b.fixed_n
582
>>> result_b.target_power
0.8
>>> len(result_b.gst_results)
2
>>>
>>> # Check that GST designs have inflated sample size
>>> result_b.gst_results[1].n_per_analysis > result_a.gst_results[1].n_per_analysis
True
>>> result_b.gst_results[2].n_per_analysis > result_a.gst_results[2].n_per_analysis
True

Realistic CTR scenario with p0=0.005, p1=0.007:

>>> import numpy as np
>>> from earlysign.stats.design.gst.common.scenarios import (
...     TwoProportionsCalculator,
...     run_scenario_a,
...     run_scenario_b,
... )
>>> np.random.seed(42)
>>>
>>> # Realistic click-through rate scenario
>>> p0 = 0.005  # 0.5% baseline
>>> p1 = 0.007  # 0.7% target
>>> effect_size_target = p1 - p0
>>> alpha = 0.05
>>> power = 0.80
>>>
>>> print(f"Control CTR: {p0*100:.2f}%")
Control CTR: 0.50%
>>> print(f"Treatment CTR: {p1*100:.2f}%")
Treatment CTR: 0.70%
>>> print(f"Effect size: {effect_size_target*100:.2f} pp")
Effect size: 0.20 pp
>>> print(f"Relative lift: {(p1/p0 - 1)*100:.0f}%")
Relative lift: 40%
>>>
>>> calculator = TwoProportionsCalculator(p_control=p0)
>>>
>>> # Calculate fixed design sample size (expect large N due to low baseline)
>>> n_per_group = calculator.calculate_sample_size(effect_size_target, alpha, power)
>>> n_total = n_per_group * 2
>>> print(f"Fixed design - Per group: {n_per_group:,}, Total: {n_total:,}")
Fixed design - Per group: 23,402, Total: 46,804
>>>
>>> # Verify large sample size is needed for low baseline CTR
>>> n_per_group > 10000
True
>>>
>>> # Define effect size range
>>> effect_sizes = np.linspace(0.0001, 0.005, 30)
>>> n_interim_list = [1, 2]
>>>
>>> # Run Scenario A: Fixed Max N
>>> result_a = run_scenario_a(
...     calculator=calculator,
...     target_effect=effect_size_target,
...     effect_sizes=effect_sizes,
...     alpha=alpha,
...     power=power,
...     n_interim_list=n_interim_list,
... )
>>>
>>> # Verify structure
>>> len(result_a.gst_results)
2
>>> result_a.fixed_n
46804
>>>
>>> # Check GST results
>>> result_a.gst_results[1].n_looks
2
>>> result_a.gst_results[2].n_looks
3
>>>
>>> # Run Scenario B: Fixed Power at target
>>> result_b = run_scenario_b(
...     calculator=calculator,
...     target_effect=effect_size_target,
...     effect_sizes=effect_sizes,
...     alpha=alpha,
...     power=power,
...     n_interim_list=n_interim_list,
...     inflation_factor=1.15,
... )
>>>
>>> # Verify structure
>>> len(result_b.gst_results)
2
>>> result_b.fixed_n
46804
>>> result_b.target_power
0.8

Sample size calculation verification:

>>> import numpy as np
>>> from earlysign.stats.design.gst.common.scenarios import TwoProportionsCalculator
>>>
>>> # Test various scenarios
>>> calc_20 = TwoProportionsCalculator(p_control=0.2)
>>> n_20 = calc_20.calculate_sample_size(effect_size=0.1, alpha=0.05, power=0.8)
>>> n_20
291
>>>
>>> calc_50 = TwoProportionsCalculator(p_control=0.5)
>>> n_50 = calc_50.calculate_sample_size(effect_size=0.1, alpha=0.05, power=0.8)
>>> n_50
385
>>>
>>> # Lower effect size requires more samples
>>> n_small = calc_20.calculate_sample_size(effect_size=0.05, alpha=0.05, power=0.8)
>>> n_small > n_20
True
>>>
>>> # Higher power requires more samples
>>> n_high_power = calc_20.calculate_sample_size(effect_size=0.1, alpha=0.05, power=0.9)
>>> n_high_power > n_20
True

Operating characteristics curve properties:

>>> import numpy as np
>>> from earlysign.stats.design.gst.common.scenarios import (
...     TwoProportionsCalculator,
...     run_scenario_a,
... )
>>> np.random.seed(42)
>>>
>>> calculator = TwoProportionsCalculator(p_control=0.2)
>>> effect_sizes = np.linspace(0.02, 0.35, 20)
>>>
>>> result = run_scenario_a(
...     calculator=calculator,
...     target_effect=0.1,
...     effect_sizes=effect_sizes,
...     alpha=0.05,
...     power=0.8,
...     n_interim_list=[1],
... )
>>>
>>> # Verify that GST results have curves computed for all effect sizes
>>> len(result.gst_results[1].results) == len(effect_sizes)
True
>>>
>>> # Verify that power increases with effect size
>>> powers = [r.power for r in result.gst_results[1].results]
>>> # Power should be monotonically increasing (with small tolerance)
>>> all(powers[i] <= powers[i+1] + 0.001 for i in range(len(powers)-1))
True
>>>
>>> # Expected sample size should decrease as effect increases (for GST)
>>> # At least for large effects
>>> esn = [r.expected_sample_size for r in result.gst_results[1].results]
>>> # For large effects, ESN should be smaller than max N
>>> max_n = result.gst_results[1].results[0].max_sample_size
>>> bool(esn[-1] < max_n)
True
"""

import doctest


def test_doctest():
    """Run all doctests in this module."""
    results = doctest.testmod(verbose=True)
    assert results.failed == 0, f"{results.failed} doctest(s) failed"


if __name__ == "__main__":
    # Run doctests when executed directly
    doctest.testmod(verbose=True)
