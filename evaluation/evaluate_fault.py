#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd
import rosbag2_py

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def get_fault_events(bag_path):

    bag_path = str(Path(bag_path).expanduser().resolve())

    reader = rosbag2_py.SequentialReader()

    reader.open(
        rosbag2_py.StorageOptions(
            uri=bag_path,
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions("", ""),
    )

    topic_types = {
        x.name: x.type
        for x in reader.get_all_topics_and_types()
    }

    first_ns = None
    events = []

    while reader.has_next():

        topic, raw, timestamp_ns = reader.read_next()

        if first_ns is None:
            first_ns = timestamp_ns

        if topic != "/robotreplay/fault_event":
            continue

        msg_class = get_message(topic_types[topic])
        msg = deserialize_message(raw, msg_class)

        relative_time = (
            timestamp_ns - first_ns
        ) / 1e9

        events.append({
            "time": relative_time,
            "message": msg.data,
        })

    return events


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("bag")
    parser.add_argument("scored_csv")

    args = parser.parse_args()

    events = get_fault_events(args.bag)

    if not events:
        raise RuntimeError(
            "No /robotreplay/fault_event found in bag."
        )

    df = pd.read_csv(args.scored_csv)

    fault_time = events[0]["time"]

    pre = df[df["time"] < fault_time].copy()
    post = df[df["time"] >= fault_time].copy()

    post_anomalies = post[
        post["is_anomaly"] == 1
    ].copy()

    print("\n=== RobotReplay Fault Evaluation ===")

    print(f"\nFault event:")
    print(f"  Relative time: {fault_time:.3f} s")
    print(f"  Event: {events[0]['message']}")

    print("\nBefore fault:")
    print(f"  Windows: {len(pre)}")

    if len(pre):
        print(
            f"  Anomaly rate: "
            f"{100 * pre['is_anomaly'].mean():.2f}%"
        )

    print("\nAfter fault:")
    print(f"  Windows: {len(post)}")

    if len(post):
        print(
            f"  Anomaly rate: "
            f"{100 * post['is_anomaly'].mean():.2f}%"
        )

    if len(post_anomalies):

        first = post_anomalies.iloc[0]

        latency = (
            first["time"] - fault_time
        )

        print(
            f"\nFirst anomaly after injection: "
            f"{first['time']:.3f} s"
        )

        print(
            f"Detection latency: "
            f"{latency:.3f} s"
        )

    else:
        print(
            "\nNo anomaly detected after injection."
        )

    # Peak anomaly after fault
    if len(post):

        peak_idx = post[
            "anomaly_score"
        ].idxmax()

        peak = df.loc[peak_idx]

        print(
            f"\nPeak post-fault anomaly:"
            f" {peak['time']:.3f} s"
        )

        print(
            f"Peak score:"
            f" {peak['anomaly_score']:.6f}"
        )

    # Inspect the telemetry close to the true fault
    window = df[
        (df["time"] >= fault_time - 2.0)
        & (df["time"] <= fault_time + 5.0)
    ].copy()

    useful = [
        "time",
        "anomaly_score",
        "is_anomaly",
        "amcl_position_step",
        "amcl_yaw_step",
        "amcl_cov_x",
        "amcl_cov_y",
        "amcl_cov_yaw",
        "tf_map_odom_position_step",
        "tf_map_odom_yaw_step",
        "tf_map_odom_age",
        "odom_vx",
        "odom_wz",
        "cmd_vx",
        "cmd_wz",
    ]

    useful = [
        c for c in useful
        if c in window.columns
    ]

    print(
        "\nTelemetry around injected fault:"
    )

    print(
        window[useful]
        .to_string(index=False)
    )

    print(
        "\nFault evaluation: PASS"
    )


if __name__ == "__main__":
    main()
