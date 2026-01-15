"""Design utilities for group sequential testing.

The subpackage hosts helpers for constructing initial designs, schemas, and
ledger records used throughout the group sequential workflow.

Examples
--------
>>> from earlysign.v0.methods.group_sequential.design import initial_design
>>> hasattr(initial_design, "constants")
True
"""

# Namespace export to make ``design.initial_design`` available for imports.
from earlysign.v0.methods.group_sequential.design import initial_design  # noqa: F401
