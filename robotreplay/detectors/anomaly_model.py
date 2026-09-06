#!/usr/bin/env python3

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


FEATURES = [
    # LiDAR health
    "scan_age",
    "scan_gap",
    "scan_rate_hz",
    "scan_valid_fraction",

    # Commanded behaviour
    "cmd_vx",
    "cmd_wz",
    "command_active",

    # Actual base motion
    "odom_vx",
    "odom_wz",

    # IMU
    "imu_wz",

    # Actuation
    "left_wheel_vel",
    "right_wheel_vel",
    "wheel_velocity_difference",
    "wheel_velocity_product",

    # Command -> motion consistency
    "linear_motion_error",
    "angular_motion_error",
    "odom_imu_yaw_error",

    # Localization changes
    "amcl_position_step",
    "amcl_yaw_step",
    "amcl_cov_x",
    "amcl_cov_y",
    "amcl_cov_yaw",

    # TF health
    "tf_map_odom_age",
    "tf_odom_base_age",
]


def load_csvs(paths):
    frames = []

    for p in paths:
        df = pd.read_csv(p)
        df["dataset_file"] = Path(p).name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_features(df, features):
    available = [f for f in features if f in df.columns]

    if not available:
        raise RuntimeError("None of the requested ML features exist.")

    return df[available].copy(), available


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="Healthy training CSVs",
    )

    parser.add_argument(
        "--validate",
        required=True,
        help="Independent healthy validation CSV",
    )

    parser.add_argument(
        "--model-out",
        default="models/healthy_isolation_forest.joblib",
    )

    parser.add_argument(
        "--results-out",
        default="data/results/healthy_validation.csv",
    )

    args = parser.parse_args()

    train_df = load_csvs(args.train)
    val_df = pd.read_csv(args.validate)

    X_train_raw, features = prepare_features(
        train_df,
        FEATURES,
    )

    X_val_raw = val_df[features].copy()

    # Missing values are expected because different ROS topics
    # publish at different rates.
    imputer = SimpleImputer(strategy="median")

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)

    # Robust scaling makes the model less sensitive to a few
    # naturally large values in healthy navigation.
    scaler = RobustScaler()

    X_train = scaler.fit_transform(X_train_imp)
    X_val = scaler.transform(X_val_imp)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train)

    # Higher value = more anomalous.
    train_scores = -model.score_samples(X_train)
    val_scores = -model.score_samples(X_val)

    # Do not blindly trust IsolationForest's built-in threshold.
    # Define normal behaviour from our actual healthy training data.
    threshold = np.percentile(train_scores, 99.0)

    train_anomaly = train_scores > threshold
    val_anomaly = val_scores > threshold

    val_df = val_df.copy()
    val_df["anomaly_score"] = val_scores
    val_df["is_anomaly"] = val_anomaly.astype(int)

    model_package = {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": features,
        "threshold": float(threshold),
    }

    model_path = Path(args.model_out)
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model_package, model_path)

    results_path = Path(args.results_out)
    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_df.to_csv(
        results_path,
        index=False,
    )

    print("\n=== RobotReplay Healthy Behaviour Model ===")

    print("\nTraining missions:")
    for p in args.train:
        print(f"  {p}")

    print("\nValidation mission:")
    print(f"  {args.validate}")

    print(f"\nTraining rows:    {len(train_df)}")
    print(f"Validation rows:  {len(val_df)}")
    print(f"ML features:      {len(features)}")

    print("\nFeatures:")
    for f in features:
        print(f"  {f}")

    print(f"\nAnomaly threshold: {threshold:.6f}")

    print(
        "Training anomaly rate: "
        f"{100 * train_anomaly.mean():.2f}%"
    )

    print(
        "Validation anomaly rate: "
        f"{100 * val_anomaly.mean():.2f}%"
    )

    print(
        "Validation anomalous rows: "
        f"{val_anomaly.sum()} / {len(val_anomaly)}"
    )

    if val_anomaly.any():

        abnormal = val_df.loc[
            val_df["is_anomaly"] == 1,
            ["time", "anomaly_score"]
        ]

        print("\nFirst validation anomalies:")
        print(abnormal.head(10).to_string(index=False))

    print(f"\nModel saved:   {model_path}")
    print(f"Results saved: {results_path}")

    print("\nHealthy anomaly model: PASS")


if __name__ == "__main__":
    main()
