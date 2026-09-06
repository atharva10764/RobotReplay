from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# HEALTHY BASELINES
# ---------------------------------------------------------
#
# IMPORTANT:
# healthy_003 is intentionally NOT included here.
#
# healthy_001 + healthy_002 = baseline / training missions
# healthy_003               = independent healthy validation
#
# This prevents validation leakage.
# ---------------------------------------------------------

DEFAULT_HEALTHY_BASELINES = [
    PROJECT_ROOT / "data/features/healthy_001.csv",
    PROJECT_ROOT / "data/features/healthy_002.csv",
]


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "healthy_isolation_forest.joblib"
)


DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)


DEFAULT_RUNTIME_DIR = (
    PROJECT_ROOT
    / "data"
    / "runtime"
)


DEFAULT_SAMPLE_PERIOD = 0.2


# ---------------------------------------------------------
# ROSBAG STORAGE BACKENDS
# ---------------------------------------------------------
#
# RobotReplay supports ROS 2 rosbag2 recordings stored
# using:
#
#   sqlite3  -> .db3
#   mcap     -> .mcap
#
# Storage compatibility is independent of the diagnostic
# models. Supporting MCAP does not retrain or modify the
# trained ML models.
# ---------------------------------------------------------

SUPPORTED_STORAGE_IDS = {
    "sqlite3",
    "mcap",
}


# ---------------------------------------------------------
# VALIDATED MVP EVIDENCE SIGNATURES
# ---------------------------------------------------------
#
# These are fault families for which the current MVP has an
# explicit ROS / robotics evidence detector.
#
# LiDAR Stream Integrity Failure was developed after an
# external public ROS 2 dataset exposed an additional
# failure mechanism. Results on that same dataset after
# development are development-set evidence, not independent
# validation of the new detector.
# ---------------------------------------------------------

SUPPORTED_FAULTS = [
    "Localization Jump",
    "LiDAR Dropout",
    "LiDAR Stream Integrity Failure",
    "TF Delay / Discontinuity",
    "Wheel / Encoder Direction Mismatch",
]
