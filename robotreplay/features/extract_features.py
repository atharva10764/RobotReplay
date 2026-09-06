#!/usr/bin/env python3

from __future__ import annotations

import argparse

from pathlib import Path


import numpy as np
import pandas as pd
import rosbag2_py


from rclpy.serialization import (
    deserialize_message,
)

from rosidl_runtime_py.utilities import (
    get_message,
)


from robotreplay.features.extract_features_core import (
    extract_features as _extract_core,
)

from robotreplay.ingestion.validation import (
    inspect_bag_directory,
    resolve_semantic_topics,
)


# =========================================================
# LIDAR INTEGRITY CONFIGURATION
# =========================================================

LIDAR_INTEGRITY_WINDOW_SECONDS = 1.0

LOW_QUALITY_SCAN_THRESHOLD = 0.25


# =========================================================
# HELPERS
# =========================================================

def _header_stamp_seconds(
    message,
) -> float | None:

    try:

        stamp = (
            message.header.stamp
        )


        return (
            float(
                stamp.sec
            )
            +
            float(
                stamp.nanosec
            )
            /
            1_000_000_000.0
        )


    except Exception:

        return None


def _exact_scan_match(
    previous_ranges: np.ndarray,
    current_ranges: np.ndarray,
) -> bool:

    if (
        len(
            previous_ranges
        )
        !=
        len(
            current_ranges
        )
    ):

        return False


    try:

        return bool(
            np.array_equal(
                previous_ranges,
                current_ranges,
                equal_nan=True,
            )
        )


    except TypeError:

        previous_safe = (
            np.nan_to_num(
                previous_ranges,
                nan=9.87654321e307,
            )
        )


        current_safe = (
            np.nan_to_num(
                current_ranges,
                nan=9.87654321e307,
            )
        )


        return bool(
            np.array_equal(
                previous_safe,
                current_safe,
            )
        )


# =========================================================
# RAW LIDAR INTEGRITY EXTRACTION
# =========================================================

def _extract_lidar_integrity(
    bag_path: Path,
) -> pd.DataFrame:

    metadata = (
        inspect_bag_directory(
            bag_path
        )
    )


    reader = (
        rosbag2_py
        .SequentialReader()
    )


    reader.open(

        rosbag2_py.StorageOptions(
            uri=str(
                bag_path
            ),
            storage_id=
                metadata.storage_id,
        ),

        rosbag2_py.ConverterOptions(
            "",
            "",
        ),
    )


    topic_types = {

        topic.name:
            topic.type

        for topic
        in reader.get_all_topics_and_types()
    }


    semantic_topics = (
        resolve_semantic_topics(
            topic_types
        )
    )


    scan_topic = (
        semantic_topics.get(
            "scan"
        )
    )


    if scan_topic is None:

        return pd.DataFrame()


    scan_type = (
        topic_types.get(
            scan_topic
        )
    )


    if (
        scan_type
        !=
        "sensor_msgs/msg/LaserScan"
    ):

        return pd.DataFrame()


    message_class = (
        get_message(
            scan_type
        )
    )


    first_ns = None

    previous_header_time = None

    previous_ranges = None


    rows = []


    while reader.has_next():

        (
            topic,
            raw,
            timestamp_ns,
        ) = reader.read_next()


        # Keep mission time aligned with the complete bag,
        # not with the first LiDAR message.
        if first_ns is None:

            first_ns = (
                timestamp_ns
            )


        if topic != scan_topic:

            continue


        mission_time = (
            timestamp_ns
            -
            first_ns
        ) / 1e9


        message = (
            deserialize_message(
                raw,
                message_class,
            )
        )


        ranges = np.asarray(
            message.ranges,
            dtype=np.float64,
        )


        # -------------------------------------------------
        # Return quality
        # -------------------------------------------------

        if len(
            ranges
        ) > 0:

            finite_mask = (
                np.isfinite(
                    ranges
                )
            )


            finite_fraction = float(
                finite_mask.mean()
            )


            valid_mask = (
                finite_mask
                &
                (
                    ranges
                    >=
                    float(
                        message.range_min
                    )
                )
                &
                (
                    ranges
                    <=
                    float(
                        message.range_max
                    )
                )
            )


            valid_fraction = float(
                valid_mask.mean()
            )


        else:

            # An empty LaserScan message is itself
            # unusable telemetry.
            finite_fraction = 0.0

            valid_fraction = 0.0


        severely_degraded = float(
            valid_fraction
            <=
            LOW_QUALITY_SCAN_THRESHOLD
        )


        # -------------------------------------------------
        # Header timestamp integrity
        # -------------------------------------------------

        header_time = (
            _header_stamp_seconds(
                message
            )
        )


        header_delta = (
            np.nan
        )


        nonmonotonic = (
            0.0
        )


        if (
            header_time is not None
            and
            previous_header_time
            is not None
        ):

            header_delta = (
                header_time
                -
                previous_header_time
            )


            if (
                header_delta
                < -1e-9
            ):

                nonmonotonic = (
                    1.0
                )


        if header_time is not None:

            previous_header_time = (
                header_time
            )


        # -------------------------------------------------
        # Exact consecutive payload repetition
        # -------------------------------------------------

        repeated = (
            0.0
        )


        if previous_ranges is not None:

            repeated = float(
                _exact_scan_match(
                    previous_ranges,
                    ranges,
                )
            )


        previous_ranges = (
            ranges.copy()
        )


        rows.append({

            "time":
                float(
                    mission_time
                ),

            "scan_header_delta_s":
                (
                    float(
                        header_delta
                    )
                    if np.isfinite(
                        header_delta
                    )
                    else np.nan
                ),

            "scan_header_nonmonotonic":
                nonmonotonic,

            "scan_payload_repeat":
                repeated,

            "scan_finite_fraction":
                finite_fraction,

            "scan_integrity_valid_fraction":
                valid_fraction,

            "scan_severely_degraded":
                severely_degraded,
        })


    frame = pd.DataFrame(
        rows
    )


    if frame.empty:

        return frame


    frame.sort_values(
        "time",
        inplace=True,
    )


    frame.reset_index(
        drop=True,
        inplace=True,
    )


    # =====================================================
    # ROLLING ONE-SECOND INTEGRITY WINDOWS
    # =====================================================

    work = frame.copy()


    work[
        "_rolling_time"
    ] = pd.to_timedelta(
        work[
            "time"
        ],
        unit="s",
    )


    work.set_index(
        "_rolling_time",
        inplace=True,
    )


    rolling = work.rolling(
        f"{LIDAR_INTEGRITY_WINDOW_SECONDS}s",
        min_periods=1,
    )


    # Number of actual LaserScan messages represented by
    # each rolling evidence window.
    frame[
        "scan_window_message_count_1s"
    ] = (
        rolling[
            "scan_integrity_valid_fraction"
        ]
        .count()
        .to_numpy(
            dtype=float
        )
    )


    # Timestamp ordering
    frame[
        "scan_header_nonmonotonic_fraction_1s"
    ] = (
        rolling[
            "scan_header_nonmonotonic"
        ]
        .mean()
        .to_numpy(
            dtype=float
        )
    )


    # Repeated exact payloads
    frame[
        "scan_payload_repeat_fraction_1s"
    ] = (
        rolling[
            "scan_payload_repeat"
        ]
        .mean()
        .to_numpy(
            dtype=float
        )
    )


    # Keep rolling means for diagnostics/backward
    # compatibility with the first development version.
    frame[
        "scan_finite_fraction_1s"
    ] = (
        rolling[
            "scan_finite_fraction"
        ]
        .mean()
        .to_numpy(
            dtype=float
        )
    )


    frame[
        "scan_valid_fraction_1s"
    ] = (
        rolling[
            "scan_integrity_valid_fraction"
        ]
        .mean()
        .to_numpy(
            dtype=float
        )
    )


    # Median quality is more robust when valid and corrupt
    # messages alternate.
    frame[
        "scan_finite_median_1s"
    ] = (
        rolling[
            "scan_finite_fraction"
        ]
        .median()
        .to_numpy(
            dtype=float
        )
    )


    frame[
        "scan_valid_median_1s"
    ] = (
        rolling[
            "scan_integrity_valid_fraction"
        ]
        .median()
        .to_numpy(
            dtype=float
        )
    )


    # Fraction of actual LaserScan messages in the window
    # that contain <=25% usable in-range returns.
    frame[
        "scan_low_quality_fraction_1s"
    ] = (
        rolling[
            "scan_severely_degraded"
        ]
        .mean()
        .to_numpy(
            dtype=float
        )
    )


    return frame


# =========================================================
# PUBLIC FEATURE EXTRACTOR
# =========================================================

def extract_features(
    bag_path,
    output_csv,
    sample_period=0.2,
    label="unknown",
):

    bag_path = Path(
        bag_path
    ).expanduser().resolve()


    output_csv = Path(
        output_csv
    ).expanduser().resolve()


    # -----------------------------------------------------
    # Existing validated canonical extractor.
    # -----------------------------------------------------

    base = _extract_core(

        bag_path,

        output_csv,

        sample_period=
            sample_period,

        label=
            label,
    )


    # -----------------------------------------------------
    # LiDAR stream-integrity enrichment.
    # -----------------------------------------------------

    integrity = (
        _extract_lidar_integrity(
            bag_path
        )
    )


    if integrity.empty:

        enriched = (
            base.copy()
        )


        print(
            "\nLiDAR integrity enrichment:"
        )


        print(
            "  LaserScan integrity telemetry "
            "not available."
        )


    else:

        integrity_columns = [

            "time",

            "scan_header_delta_s",

            "scan_header_nonmonotonic",

            "scan_payload_repeat",

            "scan_finite_fraction",

            "scan_integrity_valid_fraction",

            "scan_severely_degraded",

            "scan_window_message_count_1s",

            "scan_header_nonmonotonic_fraction_1s",

            "scan_payload_repeat_fraction_1s",

            "scan_finite_fraction_1s",

            "scan_valid_fraction_1s",

            "scan_finite_median_1s",

            "scan_valid_median_1s",

            "scan_low_quality_fraction_1s",
        ]


        enriched = pd.merge_asof(

            base.sort_values(
                "time"
            ),

            integrity[
                integrity_columns
            ].sort_values(
                "time"
            ),

            on="time",

            direction="backward",
        )


        print(
            "\nLiDAR integrity enrichment:"
        )


        print(
            f"  raw scan messages: "
            f"{len(integrity)}"
        )


        print(
            "  peak non-monotonic fraction "
            "(1 s): "
            f"{100 * integrity['scan_header_nonmonotonic_fraction_1s'].max():.2f}%"
        )


        print(
            "  peak repeated-scan fraction "
            "(1 s): "
            f"{100 * integrity['scan_payload_repeat_fraction_1s'].max():.2f}%"
        )


        print(
            "  peak degraded-scan fraction "
            "(1 s): "
            f"{100 * integrity['scan_low_quality_fraction_1s'].max():.2f}%"
        )


        print(
            "  minimum median finite-return "
            "fraction (1 s): "
            f"{100 * integrity['scan_finite_median_1s'].min():.2f}%"
        )


        print(
            "  minimum median valid-return "
            "fraction (1 s): "
            f"{100 * integrity['scan_valid_median_1s'].min():.2f}%"
        )


        print(
            "  maximum scan messages/window: "
            f"{int(integrity['scan_window_message_count_1s'].max())}"
        )


    # -----------------------------------------------------
    # Persist enriched features.
    # -----------------------------------------------------

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    enriched.to_csv(
        output_csv,
        index=False,
    )


    print(
        f"\nEnriched feature columns: "
        f"{len(enriched.columns)}"
    )


    print(
        "LiDAR integrity enrichment: PASS"
    )


    return enriched


# =========================================================
# CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract RobotReplay telemetry features "
            "with LiDAR stream-integrity evidence."
        )
    )


    parser.add_argument(
        "bag",
        help="ROS 2 bag directory",
    )


    parser.add_argument(
        "output_csv",
        help="Output feature CSV",
    )


    parser.add_argument(
        "--sample-period",
        type=float,
        default=0.2,
    )


    parser.add_argument(
        "--label",
        default="unknown",
    )


    args = (
        parser.parse_args()
    )


    extract_features(

        args.bag,

        args.output_csv,

        sample_period=
            args.sample_period,

        label=
            args.label,
    )


if __name__ == "__main__":

    main()
