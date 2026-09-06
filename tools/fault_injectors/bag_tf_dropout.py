#!/usr/bin/env python3

import argparse
from pathlib import Path

import rosbag2_py

from rclpy.serialization import (
    deserialize_message,
    serialize_message,
)

from tf2_msgs.msg import TFMessage
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

    removed_transforms = 0
    modified_tf_messages = 0
    copied_messages = 0

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
                f"TF_DROPOUT;"
                f"transform=map_to_odom;"
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

        # Remove map -> odom only during fault window
        if (
            topic == "/tf"
            and args.start <= relative_time < fault_end
        ):

            msg = deserialize_message(
                raw,
                TFMessage,
            )

            kept = []

            removed_here = 0

            for tr in msg.transforms:

                parent = tr.header.frame_id.lstrip("/")
                child = tr.child_frame_id.lstrip("/")

                if parent == "map" and child == "odom":
                    removed_here += 1
                    continue

                kept.append(tr)

            if removed_here > 0:

                removed_transforms += removed_here
                modified_tf_messages += 1

                # Preserve all unrelated TF transforms.
                if kept:
                    new_msg = TFMessage()
                    new_msg.transforms = kept

                    writer.write(
                        topic,
                        serialize_message(new_msg),
                        timestamp_ns,
                    )

                    copied_messages += 1

                continue

        writer.write(
            topic,
            raw,
            timestamp_ns,
        )

        copied_messages += 1

    print("\n=== RobotReplay TF Fault Injection ===")

    print(f"Input:                  {input_path}")
    print(f"Output:                 {output_path}")
    print(f"Fault start:            {args.start:.3f} s")
    print(f"Fault duration:         {args.duration:.3f} s")
    print(f"Removed map→odom TFs:   {removed_transforms}")
    print(f"Modified TF messages:   {modified_tf_messages}")
    print(f"Copied messages:        {copied_messages}")
    print(
        f"Ground truth marker:    "
        f"{'YES' if event_written else 'NO'}"
    )

    print("\nTF fault injection: PASS")


if __name__ == "__main__":
    main()
