#!/usr/bin/env python3

import argparse
from pathlib import Path

import rosbag2_py

from rclpy.serialization import serialize_message
from std_msgs.msg import String


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--duration", type=float, default=8.0)

    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if output_path.exists():
        raise RuntimeError(
            f"Output already exists: {output_path}"
        )

    reader = rosbag2_py.SequentialReader()

    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(input_path),
            storage_id="sqlite3"
        ),
        rosbag2_py.ConverterOptions("", "")
    )

    writer = rosbag2_py.SequentialWriter()

    writer.open(
        rosbag2_py.StorageOptions(
            uri=str(output_path),
            storage_id="sqlite3"
        ),
        rosbag2_py.ConverterOptions("", "")
    )

    topics = reader.get_all_topics_and_types()

    for topic in topics:
        writer.create_topic(topic)

    event_topic = rosbag2_py.TopicMetadata(
        name="/robotreplay/fault_event",
        type="std_msgs/msg/String",
        serialization_format="cdr",
        offered_qos_profiles=""
    )

    writer.create_topic(event_topic)

    first_ns = None
    event_written = False

    dropped_scans = 0
    copied_messages = 0

    while reader.has_next():

        topic, data, timestamp_ns = reader.read_next()

        if first_ns is None:
            first_ns = timestamp_ns

        relative_time = (
            timestamp_ns - first_ns
        ) / 1e9

        fault_end = args.start + args.duration

        # Insert exact ground-truth fault marker.
        if (
            not event_written
            and relative_time >= args.start
        ):
            event = String()
            event.data = (
                f"LIDAR_DROPOUT;"
                f"duration={args.duration:.3f};"
                f"relative_start={args.start:.3f}"
            )

            event_timestamp = (
                first_ns
                + int(args.start * 1e9)
            )

            writer.write(
                "/robotreplay/fault_event",
                serialize_message(event),
                event_timestamp
            )

            event_written = True

        # Remove scan messages only inside fault window.
        if (
            topic == "/scan"
            and args.start <= relative_time < fault_end
        ):
            dropped_scans += 1
            continue

        writer.write(
            topic,
            data,
            timestamp_ns
        )

        copied_messages += 1

    print("\n=== RobotReplay LiDAR Dropout Injection ===")
    print(f"Input:          {input_path}")
    print(f"Output:         {output_path}")
    print(f"Fault start:    {args.start:.3f} s")
    print(f"Fault duration: {args.duration:.3f} s")
    print(f"Dropped scans:  {dropped_scans}")
    print(f"Copied msgs:    {copied_messages}")
    print(f"Ground truth:   {'YES' if event_written else 'NO'}")
    print("\nLiDAR fault injection: PASS")


if __name__ == "__main__":
    main()
