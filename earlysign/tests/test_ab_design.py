"""
Doctest coverage for the binomial GST design helper and OC plotter.

>>> import matplotlib
>>> matplotlib.use("Agg")
>>> from earlysign.methods.group_sequential.design.initial_design.scenarios.binomial import (
...     create_binomial_design,
... )
>>> from earlysign.methods.group_sequential.report.plot_oc_curve import (
...     OCCurvePlotter,
... )
>>> interface = create_binomial_design(
...     alpha=0.05,
...     delta=0.05,
...     power=0.8,
...     p_control=0.2,
...     effect_sizes=[0.02, 0.04, 0.06],
...     n_sim=20,
...     seed=7,
... )
>>> comparison = interface.compare_interim(k=2)

>>> len(comparison["oc_results"])
3
>>> comparison["plot_error"] is None
True
>>> plotter = OCCurvePlotter(figsize=(4, 3))
>>> ax = plotter.plot_oc_curve(
...     comparison["oc_results"],
...     target_effect=0.05,
...     null_value=0.2,
... )
>>> ax.get_ylabel()
'Expected Sample Size (ESS)'
"""

import doctest


def test_doctest() -> None:
    """Run the doctests defined in this module."""

    results = doctest.testmod(verbose=True)
    assert results.failed == 0, f"{results.failed} doctest(s) failed"


if __name__ == "__main__":
    doctest.testmod(verbose=True)
