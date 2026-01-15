"""
Doctests for Canonical Joint Model.

This module provides tests for instantiating and solving boundaries for the
Canonical Joint Model from ES3 protocol specifications.

--- Setup ---
>>> import numpy as np
>>> import earlysign.schema.ES3.GST as GST
>>> from earlysign.v1.methods.group_sequential.canonical_dist import CanonicalJointModel, Config
>>> from earlysign.v1.methods.group_sequential.spending import OBFSpending

--- Test: Model from Spec Basic ---
>>> info_times = [0.5, 1.0]
>>> spec = GST.Protocol(
...     name="Test Protocol",
...     task=GST.TaskSpec(
...         kind="group_sequential",
...         arms=["C", "T"],
...         response_type=GST.ResponseType.BINARY,
...         efficacy=GST.EfficacyRequirement(alpha=0.025),
...         futility=GST.FutilityRequirement(power=0.9),
...         hypotheses=GST.HypothesisSpec(
...             h_null="H0", h_alt="H1",
...             test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
...             target_effect=GST.BinaryEffectSize(proportions={"C": 0.1, "T": 0.15})
...         )
...     ),
...     method=GST.MethodSpec(
...         kind="group_sequential",
...         efficacy=GST.StoppingRule(
...             schedule=GST.ScheduleSpec(unit=GST.Unit.INFORMATION_FRACTION, interim_points=info_times),
...             boundary=GST.SpendingBoundary(
...                 kind="spending",
...                 boundary_scale=GST.BoundaryScale.Z_SCORE,
...                 binding=True,
...                 reference_model=GST.BinaryModel(
...                     kind="binary", test_statistic=GST.TestStatistic.Z, 
...                     link_function=GST.LinkFunction.IDENTITY, use_canonical_joint_distribution=True
...                 ),
...                 spending_function=GST.SpendingFunctionSpec(type="obrien_fleming")
...             )
...         )
...     )
... )
>>> model = CanonicalJointModel.from_spec(spec, n_sims=5000)
>>> model.config.alpha
0.025
>>> np.allclose(model.config.info_times, [0.5, 1.0])
True
>>> model.config.efficacy_spending is not None
True

--- Test: Dual Boundary Solving (Binding) ---
>>> info_times_arr = np.array([0.5, 1.0])
>>> config = Config(
...     info_times=info_times_arr, alpha=0.025, power=0.9,
...     efficacy_spending=OBFSpending(alpha=0.025),
...     futility_spending=OBFSpending(alpha=0.1),
...     binding_futility=True, n_sims=5000, rng_seed=42
... )
>>> model = CanonicalJointModel(config)
>>> a, b = model.solve_boundaries(drift=3.24)
>>> len(a) == 2 and len(b) == 2
True
>>> bool(a[0] > a[1])  # OBF characteristic
True
>>> bool(b[0] < b[1])  # Futility characteristic
True
>>> bool(np.isclose(a[1], b[1], atol=0.1))  # Terminal match
True

--- Test: Binding vs Non-binding ---
>>> eff_sf = OBFSpending(alpha=0.025)
>>> fut_sf = OBFSpending(alpha=0.1)
>>> config_bind = Config(
...     info_times=info_times_arr, alpha=0.025, power=0.9,
...     efficacy_spending=eff_sf, futility_spending=fut_sf,
...     binding_futility=True, n_sims=10000, rng_seed=42
... )
>>> a_bind, _ = CanonicalJointModel(config_bind).solve_boundaries(drift=3.24)
>>> config_nonbind = Config(
...     info_times=info_times_arr, alpha=0.025, power=0.9,
...     efficacy_spending=eff_sf, futility_spending=fut_sf,
...     binding_futility=False, n_sims=10000, rng_seed=42
... )
>>> a_nonbind, _ = CanonicalJointModel(config_nonbind).solve_boundaries(drift=3.24)
>>> bool(a_nonbind[0] >= a_bind[0])
True

--- Test: Efficacy Only ---
>>> spec_eff = GST.Protocol(
...     name="Eff Only",
...     task=GST.TaskSpec(
...         kind="group_sequential",
...         arms=["C", "T"],
...         response_type=GST.ResponseType.BINARY,
...         efficacy=GST.EfficacyRequirement(alpha=0.05),
...         hypotheses=GST.HypothesisSpec(
...             h_null="H0", h_alt="H1",
...             test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
...             target_effect=GST.BinaryEffectSize(proportions={"C": 0.1, "T": 0.15})
...         )
...     ),
...     method=GST.MethodSpec(
...         kind="group_sequential",
...         efficacy=GST.StoppingRule(
...             schedule=GST.ScheduleSpec(unit=GST.Unit.INFORMATION_FRACTION, interim_points=info_times),
...             boundary=GST.SpendingBoundary(
...                 kind="spending",
...                 boundary_scale=GST.BoundaryScale.Z_SCORE,
...                 binding=True,
...                 reference_model=GST.BinaryModel(
...                     kind="binary", test_statistic=GST.TestStatistic.Z, 
...                     link_function=GST.LinkFunction.IDENTITY, use_canonical_joint_distribution=True
...                 ),
...                 spending_function=GST.SpendingFunctionSpec(type="obrien_fleming")
...             )
...         )
...     )
... )
>>> model_eff = CanonicalJointModel.from_spec(spec_eff)
>>> model_eff.config.alpha
0.05
>>> a_eff, b_eff = model_eff.solve_boundaries()
>>> a_eff is not None and b_eff is None
True

--- Test: Futility Only ---
>>> spec_fut = GST.Protocol(
...     name="Fut Only",
...     task=GST.TaskSpec(
...         kind="group_sequential",
...         arms=["C", "T"],
...         response_type=GST.ResponseType.BINARY,
...         futility=GST.FutilityRequirement(power=0.8),
...         hypotheses=GST.HypothesisSpec(
...             h_null="H0", h_alt="H1",
...             test_logic=GST.SuperiorityHypothesis(superiority_margin=0.0),
...             target_effect=GST.BinaryEffectSize(proportions={"C": 0.1, "T": 0.15})
...         )
...     ),
...     method=GST.MethodSpec(
...         kind="group_sequential",
...         futility=GST.StoppingRule(
...             schedule=GST.ScheduleSpec(unit=GST.Unit.INFORMATION_FRACTION, interim_points=info_times),
...             boundary=GST.SpendingBoundary(
...                 kind="spending",
...                 boundary_scale=GST.BoundaryScale.Z_SCORE,
...                 binding=True,
...                 reference_model=GST.BinaryModel(
...                     kind="binary", test_statistic=GST.TestStatistic.Z, 
...                     link_function=GST.LinkFunction.IDENTITY, use_canonical_joint_distribution=True
...                 ),
...                 spending_function=GST.SpendingFunctionSpec(type="obrien_fleming")
...             )
...         )
...     )
... )
>>> model_fut = CanonicalJointModel.from_spec(spec_fut)
>>> model_fut.config.power
0.8
>>> a_fut, b_fut = model_fut.solve_boundaries(drift=2.48)
>>> a_fut is None and b_fut is not None
True
"""
