#!/usr/bin/env python3

import sys
from collections import Counter
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_bag(bag_path: str):
    bag_path = str(Path(bag_path).expanduser().resolve())

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions("", "")

    reader.open(storage_options, converter_options)

    topic_metadata = reader.get_all_topics_and_types()
    topic_types = {
        topic.name: topic.type
        for topic in topic_metadata
    }

    print("\n=== RobotReplay Bag Reader ===")
    print(f"Bag: {bag_path}\n")

    print("Topics discovered:")
    for topic, msg_type in sorted(topic_types.items()):
        print(f"  {topic:<40} {msg_type}")

    counts = Counter()
    first_timestamp = None
    last_timestamp = None
    decoded_topics = set()

    print("\nMessage decoding check:")

    while reader.has_next():
        topic, raw_data, timestamp = reader.read_next()

        if first_timestamp is None:
            first_timestamp = timestamp

        last_timestamp = timestamp
        counts[topic] += 1

        # Prove that ROS messages can actually be deserialized.
        if topic not in decoded_topics:
            msg_class = get_message(topic_types[topic])
            msg = deserialize_message(raw_data, msg_class)

            print(
                f"  {topic:<40} "
                f"decoded as {type(msg).__name__}"
            )

            decoded_topics.add(topic)

    print("\nMessage counts:")
    for topic, count in sorted(counts.items()):
        print(f"  {topic:<40} {count}")

    if first_timestamp is not None:
        duration = (last_timestamp - first_timestamp) / 1e9
        print(f"\nDuration from messages: {duration:.3f} s")

    print(f"Total messages: {sum(counts.values())}")
    print("\nBag ingestion: PASS")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python3 bag_reader.py <rosbag_directory>"
        )
        sys.exit(1)

    read_bag(sys.argv[1])
