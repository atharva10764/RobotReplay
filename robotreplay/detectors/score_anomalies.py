#!/usr/bin/env python3

import argparse
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument(
        "--model",
        default="models/healthy_isolation_forest.joblib"
    )
    parser.add_argument(
        "--output",
        default=None
    )
    args = parser.parse_args()

    package = joblib.load(args.model)

    model = package["model"]
    imputer = package["imputer"]
    scaler = package["scaler"]
    features = package["features"]
    threshold = package["threshold"]

    df = pd.read_csv(args.csv)

    missing_features = [
        f for f in features
        if f not in df.columns
    ]

    if missing_features:
        raise RuntimeError(
            f"Missing required features: {missing_features}"
        )

    X_raw = df[features].copy()
    X_imp = imputer.transform(X_raw)
    X = scaler.transform(X_imp)

    scores = -model.score_samples(X)
    anomalies = scores > threshold

    result = df.copy()
    result["anomaly_score"] = scores
    result["is_anomaly"] = anomalies.astype(int)

    print("\n=== RobotReplay Anomaly Analysis ===")
    print(f"Rows:              {len(result)}")
    print(f"Threshold:         {threshold:.6f}")
    print(
        f"Anomalous rows:    "
        f"{anomalies.sum()} / {len(result)}"
    )
    print(
        f"Anomaly rate:      "
        f"{100 * anomalies.mean():.2f}%"
    )

    if anomalies.any():
        abnormal = result.loc[anomalies]

        first = abnormal.iloc[0]
        peak = result.loc[result["anomaly_score"].idxmax()]

        print(
            f"First anomaly:     "
            f"{first['time']:.3f} s "
            f"(score={first['anomaly_score']:.6f})"
        )

        print(
            f"Peak anomaly:      "
            f"{peak['time']:.3f} s "
            f"(score={peak['anomaly_score']:.6f})"
        )

        print("\nFirst anomalous windows:")
        print(
            abnormal[
                ["time", "anomaly_score"]
            ]
            .head(15)
            .to_string(index=False)
        )

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")

    print("\nAnomaly analysis: PASS")


if __name__ == "__main__":
    main()
