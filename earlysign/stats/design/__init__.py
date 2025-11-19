"""Group Sequential Trial (GST) Design Module.

This module provides comprehensive tools for designing and analyzing group sequential trials.

Key Features
------------
- **Boundary Calculation**: Compute critical boundaries using spending functions
  (O'Brien-Fleming, Pocock, HSD)
- **Effect Size Calculation**: Support for two-sample proportions, time-to-event,
  and two-sample means tests
- **Power Simulation**: Estimate power and operating characteristics
- **Design Optimization**: Minimize ASN, maximize power, or balanced objectives

Module Structure
----------------
earlysign/stats/design/
├── gst/
│   ├── common/        # GST-specific shared components
│   │   ├── boundaries.py    # Boundary value calculation
│   │   ├── config.py        # Design specification dataclasses
│   │   ├── lab.py           # Main orchestrator (DesignLab)
│   │   ├── simulation.py    # Simulation engine
│   │   └── types.py         # Type definitions (Enums)
│   ├── optimization.py      # Design optimization
│   ├── fixed_timing/        # Fixed timing design mode
│   ├── optimize_asn/        # ASN optimization design mode
│   ├── optimize_design/     # Multi-objective optimization design mode
│   └── fixed_power/         # Fixed power design mode

Shared scheme utilities now live in ``earlysign/stats/essentials/schemes/``::

    essentials/schemes/
    ├── protocols.py       # Protocols for effect size calculators
    ├── survival/          # Time-to-event dataclasses + calculators
    ├── two_means/         # Continuous-outcome dataclasses + calculators
    └── two_proportions/   # Binary-outcome dataclasses + calculators

Basic Usage
-----------
Two-Sample Proportions Sequential Design::

    from earlysign.stats.design.gst.common.config import ProportionsDesignSpec
    from earlysign.stats.design.gst.common.lab import DesignLab

    # Create design specification
    spec = ProportionsDesignSpec()
    spec.test.alpha = 0.025
    spec.test.power = 0.90
    spec.sequential.n_analyses = 3
    spec.effect.p_control = 0.10
    spec.effect.effect_size = 0.02
    spec.sample_size.n_per_analysis = 500

    # Run analysis
    lab = DesignLab(spec)
    lab.compute_boundaries()
    lab.run_simulations()

    # Display results
    summary = lab.get_summary()
    power_summary = lab.get_power_summary()

Design Optimization::

    from earlysign.integration.design.group_sequential.initial_design.workflows.optimize_timing import (
        DesignOptimizer,
        MinimizeASN,
    )

    spec = ProportionsDesignSpec()
    objective = MinimizeASN(planned_max_n=3000, target_power=0.90)
    optimizer = DesignOptimizer(spec, objective)
    optimal_spec = optimizer.optimize_comprehensive()

Extensibility
-------------
To add a new test type:

1. Add new effect and sample size dataclasses (and calculators) under ``essentials/schemes/<scheme>/``
2. Add a new ``DesignSpec`` subclass in ``config.py``
3. Implement the scheme-specific ``EffectSizeCalculator`` using the shared protocol
4. Update ``DesignLab._create_effect_calculator`` to reference the new calculator

To add a new optimization objective:

Create a DesignObjective subclass in optimize_timing/nelder_mead.py::

    from earlysign.integration.design.group_sequential.initial_design.workflows.optimize_timing import DesignObjective

    class CustomObjective(DesignObjective):
        def evaluate(self, spec, lab):
            # Custom objective implementation
            pass

        def get_constraints(self, spec):
            return {}

Notes
-----
This module uses approximation methods based on group sequential trial theory:

- Normal approximations
- Independence of boundaries assumption
- Simplified ASN optimization approach

For rigorous statistical guarantees, theoretical validation and simulation-based
verification should be performed.

References
----------
.. [1] Jennison, C., & Turnbull, B. W. (2000). Group Sequential Methods with
       Applications to Clinical Trials
.. [2] Lan, K. K. G., & DeMets, D. L. (1983). Discrete sequential boundaries
       for clinical trials
.. [3] Hwang, I. K., Shih, W. J., & De Cani, J. S. (1990). Group sequential
       designs using a family of type I error probability spending functions
"""
