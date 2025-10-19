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
    Operating characteristics (Performance, PowerCurve_GST, Sensitivity)
fitting
    Fit spending functions to boundaries (FitSpending)
design_schema
    Type definitions and schema validation
conversions
    Scale conversions and utility functions

See Also
--------
For detailed examples, see the individual module docstrings:
- spending.py: Alpha/beta spending function examples
- performance.py: Power curve and operating characteristics examples
- boundaries.py: Boundary calculation examples
"""
