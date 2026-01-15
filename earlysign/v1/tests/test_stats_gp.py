"""
Doctests for Gaussian Process simulation.

This module provides tests for general Gaussian Process sampling and specialized
canonical GP properties.

--- Setup ---
>>> import numpy as np
>>> from earlysign.v1.stats.gaussian_process import GaussianProcess, CanonicalGaussianProcess

--- Test: Basic Sampling ---
>>> # Simple white noise GP
>>> gp = GaussianProcess(dims=1)
>>> t = np.array([0.1, 0.5, 1.0])
>>> samples = gp.sample(t, n_sims=100)
>>> samples.shape
(100, 3)
>>> # Check that variance is approx 1 at each point
>>> np.allclose(np.var(samples, axis=0), 1.0, atol=0.5)
True

--- Test: Canonical GP Properties ---
>>> # Canonical GP should have Cov(Zi, Zj) = sqrt(min/max)
>>> gp = CanonicalGaussianProcess(drift=0.0)
>>> t = np.array([0.5, 1.0])
>>> samples = gp.sample(t, n_sims=10000)
>>> # Theoretically Cov(Z_0.5, Z_1.0) = sqrt(0.5/1.0) = 1/sqrt(2) approx 0.707
>>> # Since Z are standardized (Var=1), Correlation = Covariance
>>> corr = np.corrcoef(samples, rowvar=False)[0, 1]
>>> np.allclose(corr, 1.0 / np.sqrt(2), atol=0.05)
True

--- Test: Stopping Rule ---
>>> gp = GaussianProcess(dims=1)
>>> samples = np.array([
...     [0.0, 2.0, 0.0],  # Stops at look 2 (idx 1)
...     [0.0, 0.0, 0.0],  # Never stops
...     [3.0, 0.0, 0.0],  # Stops at look 1 (idx 0)
... ])
>>> upper = np.array([2.5, 1.5, 1.5])
>>> stopped, stop_looks = gp.apply_stopping_rule(samples, upper=upper)
>>> stopped.tolist()
[True, False, True]
>>> stop_looks.tolist()
[2, 3, 1]

--- Test: Boundary Solving ---
>>> # Solve for z such that P(Z > z) = 0.025 (should be approx 1.96)
>>> gp = CanonicalGaussianProcess(drift=0.0, rng=np.random.default_rng(42))
>>> t = np.array([1.0])
>>> samples = gp.sample(t, n_sims=50000)
>>> ever_stopped = np.zeros(50000, dtype=bool)
>>> b = gp.solve_boundary_step(0, samples, ever_stopped, 0.025, side="upper")
>>> np.allclose(b, 1.96, atol=0.05)
True

--- Test: Multivariate Sampling ---
>>> def mean_func(t):
...     # Return (k, 2)
...     return np.column_stack([t, 2*t])
>>> def cov_func(t1, t2):
...     # t1: (k1, 1), t2: (1, k2)
...     match = (t1 == t2)  # (k1, k2)
...     return match[..., np.newaxis, np.newaxis] * np.eye(2)
>>> gp = GaussianProcess(mean_func=mean_func, cov_func=cov_func, dims=2)
>>> t = np.array([0.5, 1.0])
>>> samples = gp.sample(t, n_sims=10)
>>> samples.shape
(10, 2, 2)
"""
