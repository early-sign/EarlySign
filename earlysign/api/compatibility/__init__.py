"""Compatibility layer for popular statistical packages.

This layer provides the APIs (e.g., similar function signatures and result data structures) compatible with popular statistical libraries.
The goal is to make it easy for users familiar with those libraries to adopt earlysign.core implementations with minimal changes to call sites.
They aim to produce comparable results for typical inputs, but they are reimplementations solely derived from the codebase of this package:
implementations are independently authored (no reuse of upstream source) to avoid the license issues.
"""
