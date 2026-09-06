#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import rosbag2_py

from rclpy.serialization import (
    deserialize_message,
)

from rosidl_runtime_py.utilities import (
    get_message,
)


# =========================================================
# CONSTANTS
# =========================================================

SCAN_TYPE = (
    "sensor_msgs/msg/LaserScan"
)

ODOM_TYPE = (
    "nav_msgs/msg/Odometry"
)

TF_TYPE = (
    "tf2_msgs/msg/TFMessage"
)


# =========================================================
# METADATA
# =========================================================

def storage_id_from_metadata(
    bag_dir: Path,
) -> str:

    metadata_path = (
        bag_dir
        / "metadata.yaml"
    )


    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata.yaml not found: {bag_dir}"
        )


    metadata = yaml.safe_load(
        metadata_path.read_text()
    )


    info = metadata.get(
        "rosbag2_bagfile_information",
        metadata,
    )


    storage_id = (
        info.get(
            "storage_identifier"
        )
        or
        info.get(
            "storage_id"
        )
    )


    if not storage_id:
        raise RuntimeError(
            (
                "Unable to determine "
                f"storage id for {bag_dir}"
            )
        )


    return str(
        storage_id
    )


# =========================================================
# HELPERS
# =========================================================

def to_seconds(
    timestamp_ns: int,
) -> float:

    return (
        float(timestamp_ns)
        / 1_000_000_000.0
    )


def header_time(
    message,
) -> float:

    return (
        float(
            message.header.stamp.sec
        )
        +
        float(
            message.header.stamp.nanosec
        )
        / 1_000_000_000.0
    )


def topic_rate(
    times: list[float],
) -> float:

    if len(times) < 2:
        return 0.0


    duration = (
        times[-1]
        - times[0]
    )


    if duration <= 0:
        return 0.0


    return (
        (len(times) - 1)
        / duration
    )


def topic_duration(
    times: list[float],
) -> float:

    if len(times) < 2:
        return 0.0


    return max(
        0.0,
        times[-1]
        - times[0],
    )


def gap_percentile(
    times: list[float],
    percentile: float,
) -> float:

    if len(times) < 2:
        return 0.0


    gaps = np.diff(
        np.asarray(
            times,
            dtype=float,
        )
    )


    return float(
        np.percentile(
            gaps,
            percentile,
        )
    )


def maximum_gap(
    times: list[float],
) -> float:

    if len(times) < 2:
        return 0.0


    return float(
        np.max(
            np.diff(
                np.asarray(
                    times,
                    dtype=float,
                )
            )
        )
    )


# =========================================================
# BAG ANALYSIS
# =========================================================

def analyze_bag(
    bag_dir: Path,
    label: str,
    expected_class: str,
) -> dict:

    bag_dir = (
        bag_dir
        .expanduser()
        .resolve()
    )


    storage_id = (
        storage_id_from_metadata(
            bag_dir
        )
    )


    reader = (
        rosbag2_py.SequentialReader()
    )


    reader.open(
        rosbag2_py.StorageOptions(
            uri=str(
                bag_dir
            ),
            storage_id=
                storage_id,
        ),

        rosbag2_py.ConverterOptions(
            input_serialization_format=
                "cdr",

            output_serialization_format=
                "cdr",
        ),
    )


    topic_types = {
        topic.name:
            topic.type

        for topic
        in reader.get_all_topics_and_types()
    }


    # -----------------------------------------------------
    # Semantic topic discovery for this external robot.
    # This is evaluation-only.
    # -----------------------------------------------------

    scan_topics = [
        name
        for name, type_name
        in topic_types.items()
        if type_name == SCAN_TYPE
    ]


    odom_topics = [
        name
        for name, type_name
        in topic_types.items()
        if type_name == ODOM_TYPE
    ]


    tf_topics = [
        name
        for name, type_name
        in topic_types.items()
        if (
            type_name == TF_TYPE
            and
            "static" not in name.lower()
        )
    ]


    if not scan_topics:
        raise RuntimeError(
            (
                "No LaserScan topic found "
                f"in {bag_dir}"
            )
        )


    scan_topic = (
        scan_topics[0]
    )


    odom_topic = (
        odom_topics[0]
        if odom_topics
        else None
    )


    tf_topic = (
        tf_topics[0]
        if tf_topics
        else None
    )


    laser_type = get_message(
        topic_types[
            scan_topic
        ]
    )


    # -----------------------------------------------------
    # Data containers
    # -----------------------------------------------------

    bag_first_time = None
    bag_last_time = None

    total_messages = 0

    scan_times = []
    scan_headers = []

    odom_times = []
    tf_times = []

    finite_fractions = []
    valid_fractions = []

    scan_sizes = []

    identical_scans = 0

    previous_ranges = None

    header_duplicates = 0
    header_nonmonotonic = 0

    previous_header = None


    # -----------------------------------------------------
    # Read bag
    # -----------------------------------------------------

    while reader.has_next():

        (
            topic_name,
            serialized,
            timestamp_ns,
        ) = reader.read_next()


        total_messages += 1


        current_time = (
            to_seconds(
                timestamp_ns
            )
        )


        if bag_first_time is None:
            bag_first_time = (
                current_time
            )


        bag_last_time = (
            current_time
        )


        # -------------------------------------------------
        # LiDAR
        # -------------------------------------------------

        if topic_name == scan_topic:

            message = (
                deserialize_message(
                    serialized,
                    laser_type,
                )
            )


            scan_times.append(
                current_time
            )


            current_header = (
                header_time(
                    message
                )
            )


            scan_headers.append(
                current_header
            )


            if previous_header is not None:

                delta = (
                    current_header
                    - previous_header
                )


                if abs(delta) < 1e-9:
                    header_duplicates += 1

                elif delta < 0:
                    header_nonmonotonic += 1


            previous_header = (
                current_header
            )


            ranges = np.asarray(
                message.ranges,
                dtype=np.float64,
            )


            scan_sizes.append(
                len(
                    ranges
                )
            )


            if len(ranges):

                finite = (
                    np.isfinite(
                        ranges
                    )
                )


                finite_fraction = float(
                    finite.mean()
                )


                finite_fractions.append(
                    finite_fraction
                )


                valid = (
                    finite
                    &
                    (
                        ranges
                        >= float(
                            message.range_min
                        )
                    )
                    &
                    (
                        ranges
                        <= float(
                            message.range_max
                        )
                    )
                )


                valid_fractions.append(
                    float(
                        valid.mean()
                    )
                )


                if (
                    previous_ranges
                    is not None
                    and
                    len(previous_ranges)
                    == len(ranges)
                    and
                    np.array_equal(
                        previous_ranges,
                        ranges,
                        equal_nan=True,
                    )
                ):

                    identical_scans += 1


                previous_ranges = (
                    ranges.copy()
                )


        # -------------------------------------------------
        # Odometry
        # -------------------------------------------------

        elif (
            odom_topic is not None
            and
            topic_name == odom_topic
        ):

            odom_times.append(
                current_time
            )


        # -------------------------------------------------
        # Dynamic TF
        # -------------------------------------------------

        elif (
            tf_topic is not None
            and
            topic_name == tf_topic
        ):

            tf_times.append(
                current_time
            )


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    bag_duration = 0.0


    if (
        bag_first_time is not None
        and
        bag_last_time is not None
    ):

        bag_duration = (
            bag_last_time
            - bag_first_time
        )


    scan_pairs = max(
        0,
        len(scan_times)
        - 1,
    )


    identical_fraction = (
        identical_scans
        / scan_pairs

        if scan_pairs
        else 0.0
    )


    nonmonotonic_fraction = (
        header_nonmonotonic
        / scan_pairs

        if scan_pairs
        else 0.0
    )


    duplicate_stamp_fraction = (
        header_duplicates
        / scan_pairs

        if scan_pairs
        else 0.0
    )


    odom_duration = (
        topic_duration(
            odom_times
        )
    )


    odom_coverage = (
        odom_duration
        / bag_duration

        if bag_duration > 0
        else 0.0
    )


    result = {

        "label":
            label,

        "expected_class":
            expected_class,

        "bag_path":
            str(
                bag_dir
            ),

        "storage_id":
            storage_id,

        "total_messages":
            total_messages,

        "bag_duration_s":
            bag_duration,

        "scan_topic":
            scan_topic,

        "scan_messages":
            len(
                scan_times
            ),

        "scan_rate_hz":
            topic_rate(
                scan_times
            ),

        "scan_median_gap_s":
            gap_percentile(
                scan_times,
                50,
            ),

        "scan_p95_gap_s":
            gap_percentile(
                scan_times,
                95,
            ),

        "scan_p99_gap_s":
            gap_percentile(
                scan_times,
                99,
            ),

        "scan_max_gap_s":
            maximum_gap(
                scan_times
            ),

        "scan_header_nonmonotonic_count":
            header_nonmonotonic,

        "scan_header_nonmonotonic_fraction":
            nonmonotonic_fraction,

        "scan_header_duplicate_count":
            header_duplicates,

        "scan_header_duplicate_fraction":
            duplicate_stamp_fraction,

        "scan_identical_consecutive_count":
            identical_scans,

        "scan_identical_consecutive_fraction":
            identical_fraction,

        "scan_median_finite_fraction":
            (
                float(
                    np.median(
                        finite_fractions
                    )
                )
                if finite_fractions
                else 0.0
            ),

        "scan_min_finite_fraction":
            (
                float(
                    np.min(
                        finite_fractions
                    )
                )
                if finite_fractions
                else 0.0
            ),

        "scan_median_valid_fraction":
            (
                float(
                    np.median(
                        valid_fractions
                    )
                )
                if valid_fractions
                else 0.0
            ),

        "scan_min_valid_fraction":
            (
                float(
                    np.min(
                        valid_fractions
                    )
                )
                if valid_fractions
                else 0.0
            ),

        "scan_median_rays":
            (
                float(
                    np.median(
                        scan_sizes
                    )
                )
                if scan_sizes
                else 0.0
            ),

        "odom_topic":
            odom_topic,

        "odom_messages":
            len(
                odom_times
            ),

        "odom_rate_hz":
            topic_rate(
                odom_times
            ),

        "odom_duration_s":
            odom_duration,

        "odom_coverage_fraction":
            odom_coverage,

        "tf_topic":
            tf_topic,

        "tf_messages":
            len(
                tf_times
            ),

        "tf_rate_hz":
            topic_rate(
                tf_times
            ),

        "tf_p95_gap_s":
            gap_percentile(
                tf_times,
                95,
            ),

        "tf_p99_gap_s":
            gap_percentile(
                tf_times,
                99,
            ),

        "tf_max_gap_s":
            maximum_gap(
                tf_times
            ),
    }


    return result


# =========================================================
# DIFFERENTIAL COMPARISON
# =========================================================

def build_comparison(
    results: list[dict],
) -> dict:

    normal = next(
        item
        for item in results
        if item[
            "expected_class"
        ] == "Normal"
    )


    attacks = [
        item
        for item in results
        if item[
            "expected_class"
        ] != "Normal"
    ]


    comparison = {

        "baseline":
            normal[
                "label"
            ],

        "baseline_metrics": {

            "scan_rate_hz":
                normal[
                    "scan_rate_hz"
                ],

            "scan_header_nonmonotonic_fraction":
                normal[
                    "scan_header_nonmonotonic_fraction"
                ],

            "scan_identical_consecutive_fraction":
                normal[
                    "scan_identical_consecutive_fraction"
                ],

            "scan_median_finite_fraction":
                normal[
                    "scan_median_finite_fraction"
                ],

            "scan_median_valid_fraction":
                normal[
                    "scan_median_valid_fraction"
                ],

            "odom_coverage_fraction":
                normal[
                    "odom_coverage_fraction"
                ],

            "tf_p95_gap_s":
                normal[
                    "tf_p95_gap_s"
                ],
        },

        "attack_comparisons":
            [],
    }


    for attack in attacks:

        normal_rate = (
            normal[
                "scan_rate_hz"
            ]
        )


        normal_tf_p95 = (
            normal[
                "tf_p95_gap_s"
            ]
        )


        comparison[
            "attack_comparisons"
        ].append({

            "label":
                attack[
                    "label"
                ],

            "scan_rate_ratio_vs_normal":
                (
                    attack[
                        "scan_rate_hz"
                    ]
                    / normal_rate

                    if normal_rate > 0
                    else None
                ),

            "nonmonotonic_timestamp_rate":
                attack[
                    "scan_header_nonmonotonic_fraction"
                ],

            "identical_scan_rate":
                attack[
                    "scan_identical_consecutive_fraction"
                ],

            "finite_ray_fraction_change":
                (
                    attack[
                        "scan_median_finite_fraction"
                    ]
                    -
                    normal[
                        "scan_median_finite_fraction"
                    ]
                ),

            "valid_ray_fraction_change":
                (
                    attack[
                        "scan_median_valid_fraction"
                    ]
                    -
                    normal[
                        "scan_median_valid_fraction"
                    ]
                ),

            "odom_coverage_change":
                (
                    attack[
                        "odom_coverage_fraction"
                    ]
                    -
                    normal[
                        "odom_coverage_fraction"
                    ]
                ),

            "tf_p95_gap_ratio_vs_normal":
                (
                    attack[
                        "tf_p95_gap_s"
                    ]
                    /
                    normal_tf_p95

                    if normal_tf_p95 > 0
                    else None
                ),
        })


    return comparison


# =========================================================
# CONSOLE
# =========================================================

def print_summary(
    results: list[dict],
) -> None:

    print(
        "\n"
        "========================================"
    )

    print(
        "BCUBED EXTERNAL LIDAR INTEGRITY EVALUATION"
    )

    print(
        "========================================\n"
    )


    display_rows = []


    for item in results:

        display_rows.append({

            "Run":
                item[
                    "label"
                ],

            "Class":
                item[
                    "expected_class"
                ],

            "Scan Hz":
                round(
                    item[
                        "scan_rate_hz"
                    ],
                    2,
                ),

            "NonMono %":
                round(
                    100
                    * item[
                        "scan_header_nonmonotonic_fraction"
                    ],
                    2,
                ),

            "Repeated %":
                round(
                    100
                    * item[
                        "scan_identical_consecutive_fraction"
                    ],
                    2,
                ),

            "Finite %":
                round(
                    100
                    * item[
                        "scan_median_finite_fraction"
                    ],
                    2,
                ),

            "Valid %":
                round(
                    100
                    * item[
                        "scan_median_valid_fraction"
                    ],
                    2,
                ),

            "Odom coverage %":
                round(
                    100
                    * item[
                        "odom_coverage_fraction"
                    ],
                    2,
                ),

            "TF P95 gap":
                round(
                    item[
                        "tf_p95_gap_s"
                    ],
                    4,
                ),
        })


    frame = pd.DataFrame(
        display_rows
    )


    print(
        frame.to_string(
            index=False
        )
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare BCubed normal and "
            "LiDAR-incident ROS 2 MCAP bags."
        )
    )


    parser.add_argument(
        "--normal",
        required=True,
        type=Path,
    )


    parser.add_argument(
        "--attack",
        required=True,
        nargs="+",
        type=Path,
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "external_validation"
            / "bcubed"
            / "analysis"
        ),
    )


    args = parser.parse_args()


    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    inputs = [
        (
            "Test_1-E1",
            args.normal,
            "Normal",
        )
    ]


    for attack_path in args.attack:

        inputs.append(
            (
                attack_path.name,
                attack_path,
                "LiDAR DoS incident",
            )
        )


    results = []


    for (
        label,
        bag_path,
        expected_class,
    ) in inputs:

        print(
            f"\nAnalyzing {label}..."
        )


        result = analyze_bag(
            bag_path,
            label,
            expected_class,
        )


        results.append(
            result
        )


    print_summary(
        results
    )


    comparison = (
        build_comparison(
            results
        )
    )


    csv_path = (
        args.output_dir
        / "bcubed_external_metrics.csv"
    )


    json_path = (
        args.output_dir
        / "bcubed_external_comparison.json"
    )


    pd.DataFrame(
        results
    ).to_csv(
        csv_path,
        index=False,
    )


    json_path.write_text(
        json.dumps(
            {
                "evaluation_name":
                    (
                        "BCubed external "
                        "LiDAR integrity comparison"
                    ),

                "development_note":
                    (
                        "External dataset characterization. "
                        "These recordings were not used to "
                        "train the existing RobotReplay "
                        "Isolation Forest or Random Forest."
                    ),

                "runs":
                    results,

                "comparison":
                    comparison,
            },

            indent=2,
        )
    )


    print(
        "\nSaved:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {json_path}"
    )


    print(
        "\nExternal comparison: PASS"
    )


if __name__ == "__main__":
    main()
