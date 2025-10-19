"""
Core functional components for Group Sequential Testing.

This submodule contains pure functions implementing the functional design
system documented in docs/source/methods/gst.md. These functions are:

- Pure (no side effects)
- Stateless (no Ledger dependencies)
- Composable (can be chained and combined)
- Backend-agnostic (work with numpy arrays)

For Ledger-integrated operators, see the parent `operators/` module.

Modules
-------
spending
    Alpha and beta spending functions (α(t), β(t))
boundaries
    Boundary calculation from design specifications (Design function)
information
    Information time calculation functions (ChooseT.* functions)
performance
    Operating characteristics calculation (Performance function)
sensitivity
    Minimum detectable effect calculation (Sensitivity function)
power_curves
    Power curve and power surface functions
design_schema
    Type definitions and schema validation
conversions
    Scale conversions and utility functions

Examples
--------
>>> from earlysign.stats.common.group_sequential.essentials import spending
>>> alpha_spent = spending.obf_spending(t=0.5, alpha=0.05)
>>> round(alpha_spent, 6)
0.005575

>>> from earlysign.stats.common.group_sequential.essentials import boundaries
>>> # Full functional design workflow coming soon
"""

# Submodules available for import:
# - boundaries
# - conversions
# - design_schema
# - information
# - spending
#
# Users should import specific modules they need:
# from earlysign.stats.common.group_sequential.essentials import spending
# from earlysign.stats.common.group_sequential.essentials.boundaries import resolve_boundary_from_design
