#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


RESULTS = Path("data/results")

CASES = [
    {
        "scenario": "Healthy Mission",
        "ground_truth": "Healthy",
        "incident": "healthy_003_incident.json",
        "fault_time": None,
    },
    {
        "scenario": "Localization Jump",
        "ground_truth": "Localization Jump",
        "incident": "localization_jump_001_incident.json",
        "fault_time": 10.031,
    },
    {
        "scenario": "LiDAR Dropout",
        "ground_truth": "LiDAR Dropout",
        "incident": "lidar_dropout_001_incident.json",
        "fault_time": 60.000,
    },
    {
        "scenario": "TF Delay / Discontinuity",
        "ground_truth": "TF Delay / Discontinuity",
        "incident": "tf_delay_001_incident.json",
        "fault_time": 60.000,
    },
    {
        "scenario": "Wheel / Encoder Mismatch",
        "ground_truth": "Wheel / Encoder Direction Mismatch",
        "incident": "wheel_mismatch_001_incident.json",
        "fault_time": 60.000,
    },
]


rows = []

for case in CASES:

    path = RESULTS / case["incident"]

    with open(path) as f:
        report = json.load(f)

    detected = report["root_cause"]
    status = report["mission_status"]
    event_time = report.get("first_abnormal_time_s")
    evidence = report.get("evidence_strength", 0.0)

    if case["ground_truth"] == "Healthy":

        correct = (
            status == "HEALTHY"
            and detected == "No supported fault detected"
        )

        latency = None

    else:

        correct = detected == case["ground_truth"]

        latency = (
            event_time - case["fault_time"]
            if event_time is not None
            else None
        )

    rows.append({
        "Scenario": case["scenario"],
        "Ground Truth": case["ground_truth"],
        "RobotReplay Diagnosis": detected,
        "Correct": correct,
        "Fault Time (s)": case["fault_time"],
        "Diagnostic Event (s)": event_time,
        "Diagnostic Latency (s)":
            round(latency, 3)
            if latency is not None
            else None,
        "Evidence Strength":
            round(evidence, 3),
    })


df = pd.DataFrame(rows)

output = RESULTS / "mvp_evaluation.csv"
df.to_csv(output, index=False)

faults = df[df["Ground Truth"] != "Healthy"]

fault_accuracy = faults["Correct"].mean()
healthy_correct = bool(
    df[df["Ground Truth"] == "Healthy"]["Correct"].iloc[0]
)

mean_latency = faults[
    "Diagnostic Latency (s)"
].mean()

max_latency = faults[
    "Diagnostic Latency (s)"
].max()


print("\n=== RobotReplay MVP Evaluation ===\n")

print(df.to_string(index=False))

print("\n----------------------------------------")
print(
    f"Controlled fault classes correctly diagnosed: "
    f"{faults['Correct'].sum()} / {len(faults)}"
)

print(
    f"Unseen healthy mission correctly rejected: "
    f"{'YES' if healthy_correct else 'NO'}"
)

print(
    f"Mean diagnostic latency: "
    f"{mean_latency:.3f} s"
)

print(
    f"Maximum diagnostic latency: "
    f"{max_latency:.3f} s"
)

print(
    "\nNote: Results represent the current controlled "
    "MVP validation set, not production-scale accuracy."
)

print(f"\nSaved: {output}")
print("\nMVP evaluation: PASS")
