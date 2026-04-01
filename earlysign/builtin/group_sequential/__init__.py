"""
Group Sequential Design Method Architecture.

This module follows a strict "Core-Adapter-Engine-Design-Reporting" architecture:

1.  **Core** (`core/`): The Mathematical Brain.
    *   **Responsibility**: Pure mathematics, Z-scale statistics, Boundary solving, Spending functions.
    *   **Components**: `model.py`, `policy.py`, `spending.py`.
    *   **Dependencies**: Generic math libraries (scipy, numpy). No dependency on `engine` or `design`.

2.  **Adapter** (`adapters/`): The Domain Bridge.
    *   **Responsibility**: Converts real-world data (Binomial, Continuous) to Z-scale (drift, information).
    *   **Components**: `binomial.py`, `continuous.py`, `base.py`.
    *   **Dependencies**: Can import from `core`.

3.  **Engine** (`engine/`): The Execution Runtime.
    *   **Responsibility**: Orchestrates the trial loop, manages state (Ledger), checks triggers.
    *   **Components**: `orchestrator.py`, `entities.py`, `ssr.py`.
    *   **Dependencies**: Imports `core` and `adapters`.

4.  **Design** (`design/`): The Planner.
    *   **Responsibility**: Generates Protocols, evaluates Operating Characteristics (OC) via simulation.
    *   **Components**: `designer.py`, `evaluator/`.
    *   **Dependencies**: Imports `core` and `adapters`.

5.  **Reporting** (`reporting/`): The Observer.
    *   **Responsibility**: Projections from Ledger to UI/Analysis objects.
    *   **Components**: `projectors.py`, `visualization.py`.
    *   **Dependencies**: Imports from `engine/entities.py`.
"""
