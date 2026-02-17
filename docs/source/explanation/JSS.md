# Introduction

- Sequential procedures in statistics have flourished: early stopping protocols by interim analysis, adaptive design, etc.
- There are statistical software packages that facilitate the adoption of these procedures. However, they mainly focus on the design phase of the study, and do not provide tools for the analysis phase.
- With the advent of a variety of sequential analysis methods, it is becoming increasingly important to have a tool that can serve as a unified platform for implementing and serving sequential analysis methods.

- In this article, we articulate the desiderata of sequential analysis software and identify the design patterns that fulfill them. By doing so, we establish the guiding principles for a unified architectural implementation.

- As proof of concept and practical application, we present EarlySign, a statistical software package that provides tools for the analysis phase of sequential analysis methods.

## Related Work

Existing software packages for sequential analysis
- rpact
- MAMS
- gsDesign
- SAS sequential analysis procedures
- Ax
- PlanOut
- Confidence by Spotify
- ExpAn
- seqabpy
- ...

# High-level Design 

## Key Requirements for Statistical Software in Sequential Analysis

- Flexibility: especially in adaptive design, the analysis phase can be very flexible and may deviate from the prespecified course of occurrence. For example, the experiment may be conducted under the group sequential design for a fixed number of stages but then the design may be modified based on the conditional error principle. Such variability of sequential methods poses challenges for software that assumes a single study design.
- Auditability and Robustness to Unpredictability: Sequential methods often introduce complex dependencies between events. Tracking the history of these events and their dependencies is crucial, especially when methods are adopted that were not anticipated during the design phase. Comprehensive recording of statistical events and actions enables that the methods introduced mid-stream can have access to an accurate and complete history.


## Core Design Patterns

- Event sourcing
  - append-only, immutable
- Command Query Responsibility Segregation (CQRS)

### Practical considerations
- Entity with Snapshots

<!-- Entity = Identity + Special Projection with Snapshot -->
Entities are special types of objects that maintain identity.
Under the event sourcing pattern, entities are derivatives: they are computed from the events.
In terms of the CQRS pattern, entities can be seen as special types of projections with snapshots.
An Entity is not a pure CQRS Write Model nor a pure Read Model; it's a hybrid concept.
It represents a consistently-identifiable aggregate whose state is derived from event projections but can be cached as Snapshots.

<!-- SequentialEntity = Entity with Indexed Trajectory -->
For sequential procedures (e.g., GST), certain entities are better represented as sequential entities.
In this case, the state is indexed by a temporal/ordinal coordinate (the Index),
and the state becomes a trajectory $(S_0, S_1, \ldots, S_n)$.
This trajectory is treated as a single coherent Entity.

In implementation, sequential entity may have two snapshot strategies (`SnapshotStrategy`):
- `collective`: Snapshot of the entire trajectory $[(i_1, S_{i_1}), \ldots, (i_n, S_{i_n})]$ (all states stored in one record)
- `pointwise`: Persist only the latest point $(i, S_i)$; the trajectory is reconstructed on demand by collecting pointwise snapshots.

Entities enable snapshots: Entities are beings with identity. The consistent identity enables us to create snapshots, thereby providing computation efficiency.

Not all concepts in this framework are entities, and not all events are related to entities. However, some concepts can be extracted as entities, which are specifically supported for the sake of computational efficiency.

Some (not all) aggregates of the events can be represented as entities. 

Under the CQRS pattern, the following complications arise:
- the latest entity state should be computed by a projection
- to support snapshots, the result of the projection should be saved to the event store by a write model
- the entities are the most effective when they are used across multiple projections.
- however, the projections are assumed parallelizable. They are read operations and multiple projections may try to find the latest entity state at the same time, resulting in a race condition of updating the entity state.

To resolve this, we use the Optimistic Concurrency Control (OCC) pattern.
The OCC pattern is a concurrency control method that uses version numbers to ensure that updates are applied only if the data has not been modified by another process.

Therefore, entities have a version number and they are conditionally updated only if the version number does not exist in the entity's state snapshots.

We support two types of entities:
- (Standard) Entities
- Sequential Entities

Standard entities are entities that have a unique identity.
Sequential entities are entities that have a sequential identity.
Sequential entities are also entities but their identity is defined for the trajectory of the entity, i.e., their being carries the history of the states.
For example, in a sequential analysis, the state of the trial is a standard entity as their latest state is what is of interest, while the Z statistic (the stochastic process) is more naturally considered as a sequential entity since the entire history of their realizations are used to perform the operation of the protocol (e.g., to compute the decision boundary for the next step).
The distinction is not necessarily clear-cut, and it is up to the user to decide which entity is more appropriate for their use case.
In either case, the entities are derivatives of the events in this framework, and they can be safely recomputed if necessary.

# EarlySign Implementation

Based on the conceptual design, we implement EarlySign as a statistical software package.

The package is evolved around 
- Ledger
  - Trace system
- Write Model (Writer)
- Read Model (Projection)
- Entity State (Entity)

## Ledger
Ledger is our realization of the event source.
Its schema is shown in Table 1.

- Supports timestamp, UID, and other labels.
- Supports entity identities and version numbers.

- In implementation, we use ibis framework to support various backends. It natively supports Google BigQuery and many other backends.

- All statistical events and decisions are registered in the ledger.

- Records are immutable and append-only, ensuring auditability.

- In the vanilla core layer, we can write and read any record to and from the ledger.

## Standardized Schema: EarlySign Standard Schema (ES3)
We also provide a standardized schema for sequential analysis methods.
As a platform for implementing various sequential statistical methods, we provide a standardized schema.

## Controller

Controller is a class that aggregates commands and queries to the ledger (some of which are both, for convenience).
It is the main console of operations such as generating protocols, declaring protocols, recording observations, modifying protocols, recording offline decisions, executing ad-hoc analysis, starting trigger-based analysis, reporting results, and outputting progress reports.
