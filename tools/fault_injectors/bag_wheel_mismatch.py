#!/usr/bin/env python3

import argparse
from pathlib import Path

import rosbag2_py

from rclpy.serialization import (
    deserialize_message,
    serialize_message,
)

from sensor_msgs.msg import JointState
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
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions("", ""),
    )

    writer = rosbag2_py.SequentialWriter()

    writer.open(
        rosbag2_py.StorageOptions(
            uri=str(output_path),
            storage_id="sqlite3",
        ),
        rosbag2_py.ConverterOptions("", ""),
    )

    topics = reader.get_all_topics_and_types()

    for topic in topics:
        writer.create_topic(topic)

    event_topic = rosbag2_py.TopicMetadata(
        name="/robotreplay/fault_event",
        type="std_msgs/msg/String",
        serialization_format="cdr",
        offered_qos_profiles="",
    )

    writer.create_topic(event_topic)

    first_ns = None
    event_written = False
    modified_messages = 0

    fault_end = args.start + args.duration

    while reader.has_next():

        topic, raw, timestamp_ns = reader.read_next()

        if first_ns is None:
            first_ns = timestamp_ns

        relative_time = (
            timestamp_ns - first_ns
        ) / 1e9

        # Ground-truth marker
        if (
            not event_written
            and relative_time >= args.start
        ):
            event = String()

            event.data = (
                f"WHEEL_ENCODER_MISMATCH;"
                f"wheel=right;"
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
                event_timestamp,
            )

            event_written = True

        # Reverse right-wheel feedback during fault window
        if (
            topic == "/joint_states"
            and args.start <= relative_time < fault_end
        ):

            msg = deserialize_message(
                raw,
                JointState,
            )

            names = list(msg.name)
            velocities = list(msg.velocity)

            changed = False

            for i, name in enumerate(names):

                if (
                    "wheel" in name.lower()
                    and "right" in name.lower()
                    and i < len(velocities)
                ):
                    velocities[i] *= -1.0
                    changed = True

            if changed:

                msg.velocity = velocities
                raw = serialize_message(msg)
                modified_messages += 1

        writer.write(
            topic,
            raw,
            timestamp_ns,
        )

    print("\n=== RobotReplay Wheel Fault Injection ===")
    print(f"Input:                {input_path}")
    print(f"Output:               {output_path}")
    print(f"Fault start:          {args.start:.3f} s")
    print(f"Fault duration:       {args.duration:.3f} s")
    print(f"Modified joint msgs:  {modified_messages}")
    print(
        f"Ground truth marker:  "
        f"{'YES' if event_written else 'NO'}"
    )

    print("\nWheel fault injection: PASS")


if __name__ == "__main__":
    main()
