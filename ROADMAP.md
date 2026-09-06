# RobotReplay Roadmap

RobotReplay is under active development. The current release establishes a
working ROS 2 navigation-diagnostics baseline; the roadmap focuses on
generalization, reproducibility, validation depth, and productization.

## Current Baseline

Implemented today:

- secure SQLite3 / MCAP ROS 2 bag ingestion,
- semantic topic resolution,
- synchronized feature extraction,
- healthy-behavior anomaly detection,
- fault-pattern statistical cross-checking,
- five ROS-specific evidence signatures,
- supported root-cause diagnosis,
- failure-chain reconstruction,
- interactive telemetry inspection,
- engineering recommendations,
- PDF incident reports,
- capability-aware abstention.

## Near-Term: Generalization and Reproducibility

### Unknown-bag robustness

- Test previously unseen ROS 2 navigation bags.
- Validate renamed topic handling.
- Exercise partial telemetry and missing-sensor cases.
- Make capability availability explicit in reports.
- Expand abstention tests for unsupported recordings.

### Reproducible execution

- Add Docker-based setup.
- Add automated smoke tests.
- Add CI for Python checks and non-ROS regression components.
- Add a documented regression command for compatible ROS environments.

### Evaluation expansion

- Add more unseen healthy missions.
- Add repeated controlled runs per failure signature.
- Report latency distributions rather than only single-run latency.
- Measure supported-diagnosis false positives and false negatives.
- Add detector ablations.

## Mid-Term: Broader Diagnostics

Candidate evidence families:

- odometry drift / localization divergence,
- planner oscillation,
- controller saturation,
- sensor timestamp skew,
- sustained compute or callback latency,
- perception stream degradation,
- network or middleware delay,
- power and actuator health,
- manipulation-specific telemetry.

Each new diagnosis should remain evidence-backed and independently testable.

## Longer-Term: Platform Direction

### Cross-robot telemetry normalization

Build a canonical semantic layer that maps heterogeneous robot telemetry into
stable diagnostic roles without depending on one robot's topic names.

### Temporal representations

Investigate sequence-aware representations for telemetry windows while keeping
the final diagnosis evidence-grounded.

### Open-set recognition

Strengthen explicit unknown-fault handling so unsupported incidents are
localized to affected subsystems without being forced into known classes.

### Causal dependency reasoning

Represent subsystem dependencies and propagation relationships as a graph to
improve failure-chain reconstruction.

### Incident retrieval

Store normalized incident fingerprints and retrieve similar historical events,
repairs, and recurrence patterns.

### Fleet diagnostics

Potential platform capabilities:

- multi-robot incident ingestion,
- incident history,
- recurrence analytics,
- fleet health summaries,
- release-to-release regression tracking,
- engineering workflow integrations,
- programmatic API access.

## Productization Principles

RobotReplay should continue to prioritize:

1. **Evidence over opaque confidence**
2. **Safe abstention over forced labels**
3. **Reproducibility over one-off demonstrations**
4. **Cross-robot compatibility over topic-name hard-coding**
5. **Useful engineering actions over generic anomaly alerts**
6. **Measured claims over marketing accuracy claims**

Roadmap items describe intended development direction and are not current
capabilities unless implemented and validated in the repository.
