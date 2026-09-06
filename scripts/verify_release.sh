#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd "$SCRIPT_DIR/.."
    pwd
)"

cd "$PROJECT_ROOT"

RELEASE_DIR="evaluation/baseline_release"
MANIFEST="$RELEASE_DIR/RELEASE_MANIFEST.txt"
HASHES="$RELEASE_DIR/release_sha256.txt"

mkdir -p "$RELEASE_DIR"

echo "============================================================"
echo "ROBOTREPLAY BASELINE RELEASE VERIFICATION"
echo "============================================================"

echo
echo "[1/8] Checking required evaluation evidence..."

required_files=(
    "$RELEASE_DIR/healthy_003.txt"
    "$RELEASE_DIR/localization_jump_001.txt"
    "$RELEASE_DIR/lidar_dropout_001.txt"
    "$RELEASE_DIR/tf_delay_001.txt"
    "$RELEASE_DIR/wheel_mismatch_001.txt"
    "$RELEASE_DIR/upload_security.txt"
    "$RELEASE_DIR/upload_resource_security.txt"
    "$RELEASE_DIR/upload_validation/positive_sqlite_upload.txt"
    "$RELEASE_DIR/upload_validation/positive_mcap_upload.txt"
)

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "FAIL: Missing $file"
        exit 1
    fi
    echo "PASS: $file"
done

echo
echo "[2/8] Verifying controlled mission outcomes..."

grep -q "Mission status: HEALTHY" "$RELEASE_DIR/healthy_003.txt"
grep -q "Most likely cause: No supported fault detected" "$RELEASE_DIR/healthy_003.txt"
grep -q "Most likely cause: Localization Jump" "$RELEASE_DIR/localization_jump_001.txt"
grep -q "First diagnostic event: 10.400 s" "$RELEASE_DIR/localization_jump_001.txt"
grep -q "Most likely cause: LiDAR Dropout" "$RELEASE_DIR/lidar_dropout_001.txt"
grep -q "First diagnostic event: 60.600 s" "$RELEASE_DIR/lidar_dropout_001.txt"
grep -q "Most likely cause: TF Delay / Discontinuity" "$RELEASE_DIR/tf_delay_001.txt"
grep -q "First diagnostic event: 60.600 s" "$RELEASE_DIR/tf_delay_001.txt"
grep -q "Most likely cause: Wheel / Encoder Direction Mismatch" "$RELEASE_DIR/wheel_mismatch_001.txt"
grep -q "First diagnostic event: 60.200 s" "$RELEASE_DIR/wheel_mismatch_001.txt"

echo "Controlled mission outcomes: PASS"

echo
echo "[3/8] Verifying upload security..."

grep -q "Security tests passed: 9 / 9" "$RELEASE_DIR/upload_security.txt"
grep -q "Upload security regression: PASS" "$RELEASE_DIR/upload_security.txt"
grep -q "Resource security tests passed: 3 / 3" "$RELEASE_DIR/upload_resource_security.txt"
grep -q "Resource security regression: PASS" "$RELEASE_DIR/upload_resource_security.txt"

echo "Upload security: PASS"

echo
echo "[4/8] Verifying positive upload validation..."

grep -qi "PASS" "$RELEASE_DIR/upload_validation/positive_sqlite_upload.txt"
grep -qi "PASS" "$RELEASE_DIR/upload_validation/positive_mcap_upload.txt"

echo "SQLite3 / MCAP positive upload validation: PASS"

echo
echo "[5/8] Verifying dashboard and ROS imports..."

python3 -m py_compile dashboard/app.py

python3 - <<'PY'
import reportlab
import streamlit
import plotly
import robotreplay
import rclpy
import rosbag2_py
import rosidl_runtime_py

print("Dashboard / PDF / ROS import check: PASS")
PY

echo
echo "[6/8] Checking tracked repository hygiene..."

forbidden="$(
    git ls-files \
    | grep -Ei \
        '(^|/)(\.venv|external_validation|scenarios|backup_before_)|\.db3$|\.mcap$|__pycache__|_before_.*\.py$|\.zip$' \
    | grep -v '^examples/localization_jump_rosbag\.zip$' \
    || true
)"

if [[ -n "$forbidden" ]]; then
    echo "FAIL: Forbidden tracked content:"
    echo "$forbidden"
    exit 1
fi

echo "Tracked repository hygiene: PASS"

echo
echo "[7/8] Writing release manifest..."

python3 - <<'PY' > "$MANIFEST"
from datetime import datetime, timezone
from importlib.metadata import version
import os
import platform

packages = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "PyYAML",
    "plotly",
    "streamlit",
    "reportlab",
]

try:
    package_version = version("robotreplay")
except Exception:
    package_version = "not-installed"

print("=" * 64)
print("ROBOTREPLAY BASELINE RELEASE MANIFEST")
print("=" * 64)
print()
print("Release:                v1.0.0")
print("Package version:        " + package_version)
print("Generated UTC:          " + datetime.now(timezone.utc).isoformat())
print()
print("PROJECT STATUS")
print("--------------")
print("Active development")
print()
print("RUNTIME")
print("-------")
print("ROS_DISTRO:             " + os.environ.get("ROS_DISTRO", "not-set"))
print("ROS_VERSION:            " + os.environ.get("ROS_VERSION", "not-set"))
print("ROS_PYTHON_VERSION:     " + os.environ.get("ROS_PYTHON_VERSION", "not-set"))
print("Python:                 " + platform.python_version())
print("Platform:               " + platform.platform())
print()
print("PYTHON PACKAGES")
print("---------------")

for package in packages:
    try:
        print(f"{package:<24}{version(package)}")
    except Exception:
        print(f"{package:<24}NOT INSTALLED")

print()
print("CONTROLLED BASELINE REGRESSION")
print("------------------------------")
print("healthy_003              HEALTHY / no supported fault")
print("localization_jump_001    Localization Jump / 10.400 s")
print("lidar_dropout_001        LiDAR Dropout / 60.600 s")
print("tf_delay_001             TF Delay / Discontinuity / 60.600 s")
print("wheel_mismatch_001       Wheel / Encoder Direction Mismatch / 60.200 s")
print()
print("Controlled recordings:   5 / 5 expected outcomes")
print()
print("UPLOAD VALIDATION")
print("-----------------")
print("Input-security suite:    9 / 9 PASS")
print("Resource-security suite: 3 / 3 PASS")
print("SQLite3 upload:          PASS")
print("MCAP upload:             PASS")
print()
print("SUPPORTED EVIDENCE SIGNATURES")
print("-----------------------------")
print("1. Localization Jump")
print("2. LiDAR Dropout")
print("3. LiDAR Stream Integrity Failure")
print("4. TF Delay / Discontinuity")
print("5. Wheel / Encoder Direction Mismatch")
print()
print("EVALUATION SCOPE")
print("----------------")
print(
    "The supervised classifier achieved 100% on the 200-sample "
    "controlled held-out telemetry-augmentation benchmark."
)
print("This is not a production-world accuracy claim.")
print(
    "BCubed / ACOLYTE results used while developing the LiDAR "
    "Stream Integrity Failure signature are external development "
    "evidence, not independent blind validation."
)
print()
print("STATUS")
print("------")
print("Baseline release verification: PASS")
PY

echo "Release manifest written: $MANIFEST"

echo
echo "[8/8] Writing SHA-256 release hashes..."

sha256sum \
    README.md \
    requirements.txt \
    pyproject.toml \
    LICENSE \
    ROADMAP.md \
    SECURITY.md \
    docs/INSTALL.md \
    docs/ARCHITECTURE.md \
    docs/EVALUATION.md \
    dashboard/app.py \
    models/fault_classifier.joblib \
    models/healthy_isolation_forest.joblib \
    robotreplay/config.py \
    robotreplay/services/analyzer.py \
    robotreplay/ingestion/upload.py \
    robotreplay/ingestion/validation.py \
    robotreplay/features/extract_features.py \
    robotreplay/detectors/localization_jump.py \
    robotreplay/detectors/lidar_dropout.py \
    robotreplay/detectors/lidar_integrity.py \
    robotreplay/detectors/tf_delay.py \
    robotreplay/detectors/wheel_mismatch.py \
    > "$HASHES"

echo "SHA-256 manifest written: $HASHES"

echo
echo "============================================================"
echo "ROBOTREPLAY BASELINE RELEASE VERIFICATION: PASS"
echo "============================================================"
