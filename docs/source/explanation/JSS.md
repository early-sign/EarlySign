# Introduction

- Sequential procedures in statistics have flourished: early stopping protocols by interim analysis, adaptive design, etc.
- There are statistical software packages that facilitate the adoption of these procedures. However, they mainly focus on the design phase of the study, and do not provide tools for the analysis phase.
- With the advent of a variety of sequential analysis methods, it is becoming increasingly important to have a tool that can serve as a unified platform for implementing and serving sequential analysis methods.

- This article identifies key requirements for statistical software in sequential analysis and outlines design patterns to address them, establishing a unified foundation for software architecture for sequential analysis.

- As proof of concept and practical application, we present EarlySign, a statistical software package that provides tools for the analysis phase of sequential analysis methods.

## Related Work

Existing software packages for sequential analysis
- rpact
- MAMS
- gsDesign
- SAS sequential analysis procedures
- ...

# High-level Design 

## Key Requirements for Statistical Software in Sequential Analysis

- Flexibility: especially in adaptive design, the analysis phase can be very flexible and may deviate from the prespecified course of occurrence. For example, the experiment may be conducted under the group sequential design for a fixed number of stages but then the design may be modified based on the conditional error principle. Such variability of sequential methods poses challenges for software that assumes a single study design.
- Auditability and Robustness to Unpredictability: Sequential methods often introduce complex dependencies between events. Tracking the history of these events and their dependencies is crucial, especially when methods are adopted that were not anticipated during the design phase. Comprehensive recording of statistical events and actions enables that the methods introduced mid-stream can have access to an accurate and complete history.


## Core Design Patterns

- Event sourcing
- Command Query Responsibility Segregation (CQRS)

### Practical considerations
Entity for snapshots:
Some (not all) aggregates of the events can be represented as entities. 
This is realized by the Optimistic Concurrency Control (OCC) pattern.

Therefore, entities have a version number and they are conditionally updated only if the version number does not exist in the entity's state snapshots.

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

- We use ibis framework to support various backends.

- All statistical events and decisions are registered in the ledger.

- Records are immutable and append-only, ensuring auditability.

- In the vanilla core layer, we can write and read any record to and from the ledger.

## Standardized Schema: EarlySign Standard Schema (ES3)
We also provide a standardized schema for sequential analysis methods.
As a platform for implementing various sequential statistical methods, we provide a standardized schema.
