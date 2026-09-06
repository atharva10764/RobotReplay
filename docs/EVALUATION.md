# RobotReplay Evaluation

This document describes the current evaluation evidence and the limits of the
claims that can be made from it.

## Evaluation Philosophy

RobotReplay separates three questions:

1. Did the system observe abnormal behavior?
2. Did statistical patterns resemble a known controlled fault?
3. Did ROS-specific evidence establish a supported diagnosis?

The final diagnosis is based on the third question.

## Controlled Recorded-Mission Regression

Current controlled recordings:

| Recording | Expected result | First supported event |
|---|---|---:|
| `healthy_003` | HEALTHY — no supported fault | — |
| `localization_jump_001` | Localization Jump | 10.400 s |
| `lidar_dropout_001` | LiDAR Dropout | 60.600 s |
| `tf_delay_001` | TF Delay / Discontinuity | 60.600 s |
| `wheel_mismatch_001` | Wheel / Encoder Direction Mismatch | 60.200 s |

Current result:

```text
5 / 5 controlled recordings produced the expected final outcome.
```

This is a small regression set, not a production accuracy estimate.

## Diagnostic Latency

For controlled fault recordings, latency is measured from the known injected
fault time to the first supported diagnostic event.

| Failure | Measured latency |
|---|---:|
| Localization Jump | 0.369 s |
| LiDAR Dropout | 0.600 s |
| TF Delay / Discontinuity | 0.600 s |
| Wheel / Encoder Direction Mismatch | 0.200 s |

Summary:

```text
Mean latency:     ~0.442 s
Maximum latency:   0.600 s
```

The `/robotreplay/fault_event` marker is used for controlled evaluation timing
only and is not available to the diagnosis path.

## Healthy-Behavior Layer

On unseen healthy mission `healthy_003`:

```text
Anomalous windows: approximately 0.73%
Final evidence diagnosis: HEALTHY / no supported fault
```

This illustrates why the generic anomaly layer is treated as a cross-check
rather than a root-cause engine.

## Supervised Fault-Pattern Benchmark

Training design:

- source missions for training: `healthy_001`, `healthy_002`,
- held-out source mission: `healthy_003`,
- window size: 20 s,
- classes: Healthy plus four controlled fault families,
- training samples: 600,
- held-out test samples: 200,
- model: Random Forest.

Measured held-out telemetry-augmentation benchmark:

| Metric | Result |
|---|---:|
| Accuracy | 100% |
| Macro precision | 100% |
| Macro recall | 100% |
| Macro F1 | 100% |
| Healthy false-positive rate | 0% |

This benchmark measures separability of the controlled augmented classes on the
held-out source mission. It is **not** evidence of 100% real-world diagnostic
accuracy.

## Real Recorded-Bag Classifier Check

The classifier was also checked on five recorded controlled missions and
returned the expected class in all five cases.

This remains a small controlled evaluation.

## Feature-Coverage Gate

The supervised classifier expects 34 features and requires at least 60%
compatible feature coverage.

When coverage is insufficient, it abstains.

This prevents an unfamiliar or partial recording from being assigned a
statistical class using too little compatible information.

## LiDAR Stream Integrity Development Evidence

The public BCubed / ACOLYTE ROS 2 data was used while developing the LiDAR
Stream Integrity Failure signature.

The resulting evidence pattern uses:

- non-monotonic scan timestamps,
- exact repeated scans,
- low scan usability.

The detector does not hard-code a specific robot, topic name, or scan rate.

Because those recordings informed detector development, results on the same
recordings are classified as **external development evidence**, not independent
blind validation.

## Security Regression

Current upload-validation evidence:

```text
Input-security suite:    9 / 9 PASS
Resource-security suite: 3 / 3 PASS
SQLite3 positive upload: PASS
MCAP positive upload:    PASS
```

These are regression tests for implemented controls and do not constitute a
security certification.

## What the Current Evidence Supports

The current evidence supports claims that RobotReplay:

- can ingest supported ROS 2 bag formats,
- can reproduce five controlled mission outcomes,
- can identify the first supported event with measured latency in four
  controlled fault recordings,
- can keep generic anomaly scoring separate from evidence-based diagnosis,
- can abstain when classifier feature coverage is insufficient,
- has a working external-data-derived LiDAR integrity signature,
- has tested input-security controls.

## What the Current Evidence Does Not Support

The current evidence does **not** establish:

- arbitrary open-world fault diagnosis,
- production-wide accuracy,
- cross-robot generalization,
- robustness to every ROS 2 topic schema,
- fleet-scale performance,
- formal security certification.

## Next Evaluation Priorities

The next evaluation phase should add:

- previously unseen ROS 2 bags,
- renamed-topic recordings,
- partial telemetry cases,
- multiple healthy environments,
- repeated runs per fault family,
- independent external datasets,
- additional robot platforms,
- latency distributions,
- false-positive / false-negative analysis,
- detector ablations.
