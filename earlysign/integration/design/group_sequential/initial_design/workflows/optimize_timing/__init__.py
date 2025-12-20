"""Compatibility shim for timing optimisation workflows."""

from earlysign.methods.group_sequential.design.initial_design.workflows import (
    optimize_timing,
)

minimize_asn = optimize_timing.minimize_asn
