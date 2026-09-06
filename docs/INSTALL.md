# RobotReplay Installation

RobotReplay is an evidence-based incident-investigation system for ROS 2
autonomous robot recordings.

## Validated Baseline Environment

- ROS 2 Humble
- Python 3.10.12
- Linux x86_64
- SQLite3 and MCAP ROS bag storage

RobotReplay requires ROS 2 Python libraries supplied by the ROS installation.
These are intentionally not installed from PyPI.

Required ROS Python modules:

- `rclpy`
- `rosbag2_py`
- `rosidl_runtime_py`

## 1. Source ROS 2 Humble

```bash
source /opt/ros/humble/setup.bash
```

Verify:

```bash
echo "$ROS_DISTRO"
```

Expected:

```text
humble
```

## 2. Create the Python Environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
source /opt/ros/humble/setup.bash
```

Verify Python:

```bash
python3 --version
```

Validated version:

```text
Python 3.10.12
```

## 3. Install Python Dependencies

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e . --no-deps
```

Do not install `rclpy`, `rosbag2_py`, or `rosidl_runtime_py` from PyPI. They are
supplied by ROS 2 Humble.

Validated Python runtime packages:

```text
numpy         2.2.6
pandas        2.3.3
scipy         1.15.3
scikit-learn  1.7.2
joblib        1.5.3
PyYAML        6.0.3
plotly        6.9.0
streamlit     1.61.1
reportlab     4.4.9
```

The current RobotReplay machine-learning artifacts were validated with
`scikit-learn==1.7.2`.

## 4. Verify ROS Integration

```bash
python3 - <<'PY'
import rclpy
import rosbag2_py
import rosidl_runtime_py

print("rclpy:             PASS")
print("rosbag2_py:        PASS")
print("rosidl_runtime_py: PASS")
PY
```

All three imports must succeed before RobotReplay analyzes ROS 2 bags.

## 5. Verify Python Runtime

```bash
python3 - <<'PY'
import numpy
import pandas
import scipy
import sklearn
import joblib
import yaml
import plotly
import streamlit
import reportlab

print("numpy:        ", numpy.__version__)
print("pandas:       ", pandas.__version__)
print("scipy:        ", scipy.__version__)
print("scikit-learn: ", sklearn.__version__)
print("joblib:       ", joblib.__version__)
print("PyYAML:       ", yaml.__version__)
print("plotly:       ", plotly.__version__)
print("streamlit:    ", streamlit.__version__)
print("reportlab:    ", reportlab.Version)
PY
```

## 6. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Open the local Streamlit address shown in the terminal.

The product workflow is:

```text
Analyze Recording / Sample Incidents
        ↓
Incident Summary
        ↓
What Happened
        ↓
Evidence + Telemetry
        ↓
Failure Timeline
        ↓
Recommended Actions
        ↓
PDF Incident Investigation Report
```

## 7. Analyze a Local ROS 2 Bag from the CLI

Use any compatible ROS 2 bag directory available on your machine:

```bash
robotreplay /path/to/ros2_bag_directory
```

or:

```bash
python3 -m robotreplay.analyze /path/to/ros2_bag_directory
```

Large raw controlled recordings are intentionally excluded from version
control.

## 8. Validate a Zipped ROS 2 Bag

RobotReplay accepts ZIP archives containing exactly one ROS 2 bag.

Supported bag storage backends:

- SQLite3 (`.db3`)
- MCAP (`.mcap`)

Example:

```bash
python3 scripts/validate_upload.py /path/to/recording.zip
```

Validate the included reproducible example:

```bash
python3 scripts/validate_upload.py examples/localization_jump_rosbag.zip
```

The current compressed upload limit is 250 MiB.

## 9. Run Upload Security Regressions

```bash
python3 scripts/test_upload_security.py
python3 scripts/test_upload_resource_security.py
```

Current baseline result:

```text
Input-security cases:      9 / 9 PASS
Resource-security cases:   3 / 3 PASS
Valid SQLite3 upload:      PASS
Valid MCAP upload:         PASS
```

These tests describe RobotReplay's implemented regression suite and are not a
general security certification.

## 10. Supported Evidence Signatures

1. Localization Jump
2. LiDAR Dropout
3. LiDAR Stream Integrity Failure
4. TF Delay / Discontinuity
5. Wheel / Encoder Direction Mismatch

RobotReplay may abstain from statistical cross-checks when a recording does not
contain sufficient feature coverage.

Unsupported incidents are not forced into a known failure signature.

## 11. Run Baseline Verification

With the Python environment active and ROS sourced:

```bash
bash scripts/verify_release.sh
```

The script verifies controlled mission evidence, upload-security regression,
positive storage-backend validation, runtime imports, tracked-file hygiene, and
release hashes.

## 12. Evaluation Scope

The controlled baseline regression includes:

- one unseen healthy recording,
- Localization Jump,
- LiDAR Dropout,
- TF Delay / Discontinuity,
- Wheel / Encoder Direction Mismatch.

The supervised classifier's 100% metric is scoped to a 200-sample controlled
held-out telemetry-augmentation benchmark and is not a claim of production-wide
accuracy.

The LiDAR Stream Integrity Failure signature was developed using the public
BCubed / ACOLYTE dataset. Results on those same development recordings are
treated as development-set evidence rather than independent validation.

RobotReplay should not be described as providing arbitrary open-world diagnosis
or production-wide accuracy guarantees.
