# RobotReplay Reproducible Example

`localization_jump_rosbag.zip` is a compact ROS 2 recording included to exercise
the complete upload and diagnosis workflow.

## Run

From the repository root:

```bash
streamlit run dashboard/app.py
```

Then:

1. Open **Analyze Recording**.
2. Upload `examples/localization_jump_rosbag.zip`.
3. Run the analysis.
4. Inspect the incident summary, evidence, telemetry, failure timeline,
   recommended actions, and PDF report.

Expected result:

```text
Mission status:          ANOMALOUS
Most likely cause:       Localization Jump
First diagnostic event:  10.400 s
Evidence strength:       1.00
```

The **Sample Incidents** workflow can also be explored without uploading a
recording.

This example is intended for reproducibility and interface testing. It should
not be interpreted as evidence of general cross-robot diagnostic accuracy.
