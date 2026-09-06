#!/usr/bin/env python3

from __future__ import annotations


import pandas as pd


from robotreplay.detectors.common import (
    no_detection,
    unavailable,
)


NAME = (
    "LiDAR Stream Integrity Failure"
)


# =========================================================
# EVIDENCE REQUIREMENTS
# =========================================================
#
# The signature is intentionally conjunctive.
#
# A recording is NOT classified from scan rate, robot name,
# topic name, or one unusual LaserScan property.
#
# A valid event requires:
#
#   1. enough real LaserScan messages to form evidence,
#   2. timestamp ordering degradation,
#   3. exact consecutive payload repetition,
#   4. a majority of scans having severely degraded usable
#      range content.
#
# BCubed Test-3 was used as development evidence for this
# signature. Post-development performance on those same
# recordings must not be described as independent
# validation.
# =========================================================

MIN_WINDOW_SCAN_MESSAGES = 5

NONMONOTONIC_FRACTION_THRESHOLD = 0.10

REPEATED_SCAN_FRACTION_THRESHOLD = 0.20

LOW_QUALITY_SCAN_FRACTION_THRESHOLD = 0.50


# This is used only to describe downstream propagation.
# It does not trigger the LiDAR diagnosis itself.
DOWNSTREAM_STALE_THRESHOLD_S = 1.0


# =========================================================
# HELPERS
# =========================================================

def _numeric(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in frame.columns:

        return pd.Series(
            index=frame.index,
            dtype=float,
        )


    return pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    )


def _safe_max(
    values: pd.Series,
) -> float | None:

    values = (
        values.dropna()
    )


    if values.empty:

        return None


    return float(
        values.max()
    )


def _safe_min(
    values: pd.Series,
) -> float | None:

    values = (
        values.dropna()
    )


    if values.empty:

        return None


    return float(
        values.min()
    )


def _downstream_staleness(
    mission: pd.DataFrame,
    detection_time: float,
) -> dict:

    later = mission[
        mission[
            "time"
        ]
        >=
        detection_time
    ]


    result = {

        "observed":
            False,

        "first_event_s":
            None,

        "peak_odom_age_s":
            None,

        "peak_map_odom_age_s":
            None,

        "peak_odom_base_age_s":
            None,
    }


    candidate_times = []


    for (
        column,
        output_name,
    ) in [

        (
            "odom_age",
            "peak_odom_age_s",
        ),

        (
            "tf_map_odom_age",
            "peak_map_odom_age_s",
        ),

        (
            "tf_odom_base_age",
            "peak_odom_base_age_s",
        ),

    ]:

        if column not in later.columns:

            continue


        values = pd.to_numeric(
            later[
                column
            ],
            errors="coerce",
        )


        finite = (
            values.dropna()
        )


        if finite.empty:

            continue


        result[
            output_name
        ] = round(
            float(
                finite.max()
            ),
            3,
        )


        stale_mask = (
            values
            >=
            DOWNSTREAM_STALE_THRESHOLD_S
        )


        if not stale_mask.any():

            continue


        stale_rows = later[
            stale_mask.fillna(
                False
            )
        ]


        if stale_rows.empty:

            continue


        candidate_times.append(
            float(
                stale_rows.iloc[
                    0
                ][
                    "time"
                ]
            )
        )


    if candidate_times:

        result[
            "observed"
        ] = True


        result[
            "first_event_s"
        ] = round(
            min(
                candidate_times
            ),
            3,
        )


    return result


# =========================================================
# DETECTOR
# =========================================================

def detect_lidar_integrity(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
) -> dict:

    # Current V2 signature relies on direct internal
    # consistency of each LaserScan stream rather than a
    # cross-robot numeric healthy baseline.
    del healthy


    required = {

        "time",

        "scan_window_message_count_1s",

        "scan_header_nonmonotonic_fraction_1s",

        "scan_payload_repeat_fraction_1s",

        "scan_low_quality_fraction_1s",

        "scan_finite_median_1s",

        "scan_valid_median_1s",
    }


    missing = sorted(
        required
        -
        set(
            mission.columns
        )
    )


    if missing:

        return unavailable(
            NAME,
            (
                "Missing LiDAR integrity features: "
                +
                ", ".join(
                    missing
                )
            ),
        )


    time = _numeric(
        mission,
        "time",
    )


    scan_count = _numeric(
        mission,
        "scan_window_message_count_1s",
    )


    nonmonotonic = _numeric(
        mission,
        "scan_header_nonmonotonic_fraction_1s",
    )


    repeated = _numeric(
        mission,
        "scan_payload_repeat_fraction_1s",
    )


    low_quality = _numeric(
        mission,
        "scan_low_quality_fraction_1s",
    )


    finite_median = _numeric(
        mission,
        "scan_finite_median_1s",
    )


    valid_median = _numeric(
        mission,
        "scan_valid_median_1s",
    )


    # =====================================================
    # CONJUNCTIVE EVENT
    # =====================================================

    enough_scans = (
        scan_count
        >=
        MIN_WINDOW_SCAN_MESSAGES
    )


    timestamp_integrity_failure = (
        nonmonotonic
        >=
        NONMONOTONIC_FRACTION_THRESHOLD
    )


    repeated_payload_failure = (
        repeated
        >=
        REPEATED_SCAN_FRACTION_THRESHOLD
    )


    severe_quality_failure = (
        low_quality
        >=
        LOW_QUALITY_SCAN_FRACTION_THRESHOLD
    )


    event_mask = (

        enough_scans

        &
        timestamp_integrity_failure

        &
        repeated_payload_failure

        &
        severe_quality_failure
    )


    candidates = mission[
        event_mask.fillna(
            False
        )
    ]


    # =====================================================
    # MISSION SUMMARY
    # =====================================================

    peak_nonmonotonic = (
        _safe_max(
            nonmonotonic
        )
    )


    peak_repeated = (
        _safe_max(
            repeated
        )
    )


    peak_low_quality = (
        _safe_max(
            low_quality
        )
    )


    minimum_finite_median = (
        _safe_min(
            finite_median
        )
    )


    minimum_valid_median = (
        _safe_min(
            valid_median
        )
    )


    maximum_scan_count = (
        _safe_max(
            scan_count
        )
    )


    summary_evidence = {

        "minimum_window_scan_messages":
            MIN_WINDOW_SCAN_MESSAGES,

        "nonmonotonic_fraction_threshold":
            NONMONOTONIC_FRACTION_THRESHOLD,

        "repeated_scan_fraction_threshold":
            REPEATED_SCAN_FRACTION_THRESHOLD,

        "low_quality_scan_fraction_threshold":
            LOW_QUALITY_SCAN_FRACTION_THRESHOLD,

        "peak_nonmonotonic_fraction_1s":
            (
                round(
                    peak_nonmonotonic,
                    4,
                )
                if peak_nonmonotonic
                is not None
                else None
            ),

        "peak_repeated_scan_fraction_1s":
            (
                round(
                    peak_repeated,
                    4,
                )
                if peak_repeated
                is not None
                else None
            ),

        "peak_low_quality_scan_fraction_1s":
            (
                round(
                    peak_low_quality,
                    4,
                )
                if peak_low_quality
                is not None
                else None
            ),

        "minimum_finite_median_1s":
            (
                round(
                    minimum_finite_median,
                    4,
                )
                if minimum_finite_median
                is not None
                else None
            ),

        "minimum_valid_median_1s":
            (
                round(
                    minimum_valid_median,
                    4,
                )
                if minimum_valid_median
                is not None
                else None
            ),

        "maximum_scan_messages_per_window":
            (
                int(
                    maximum_scan_count
                )
                if maximum_scan_count
                is not None
                else None
            ),

        "candidate_window_count":
            int(
                event_mask.fillna(
                    False
                ).sum()
            ),
    }


    # =====================================================
    # NO DETECTION
    # =====================================================

    if candidates.empty:

        return no_detection(
            NAME,
            summary_evidence,
        )


    # =====================================================
    # FIRST SUPPORTED EVENT
    # =====================================================

    first = (
        candidates.iloc[
            0
        ]
    )


    detection_time = float(
        first[
            "time"
        ]
    )


    scans_at_detection = int(
        first[
            "scan_window_message_count_1s"
        ]
    )


    nonmonotonic_at_detection = float(
        first[
            "scan_header_nonmonotonic_fraction_1s"
        ]
    )


    repeated_at_detection = float(
        first[
            "scan_payload_repeat_fraction_1s"
        ]
    )


    degraded_at_detection = float(
        first[
            "scan_low_quality_fraction_1s"
        ]
    )


    finite_median_at_detection = float(
        first[
            "scan_finite_median_1s"
        ]
    )


    valid_median_at_detection = float(
        first[
            "scan_valid_median_1s"
        ]
    )


    downstream = (
        _downstream_staleness(
            mission,
            detection_time,
        )
    )


    # =====================================================
    # EVIDENCE STRENGTH
    # =====================================================
    #
    # Heuristic evidence score, NOT an ML probability.
    #
    # The three independent LiDAR-integrity conditions are
    # already simultaneously satisfied before this code is
    # reached.
    # =====================================================

    evidence_strength = (
        0.95
    )


    if downstream[
        "observed"
    ]:

        evidence_strength = (
            1.0
        )


    # =====================================================
    # HUMAN-READABLE EVIDENCE
    # =====================================================

    supporting_evidence = [

        (
            f"{scans_at_detection} LaserScan messages "
            "were available in the evidence window, "
            "providing sufficient observations for "
            "stream-integrity assessment."
        ),

        (
            "LiDAR header timestamp ordering degraded "
            f"({100 * nonmonotonic_at_detection:.1f}% "
            "non-monotonic messages in the detection "
            "window)."
        ),

        (
            "Exact consecutive LiDAR payloads repeated "
            f"abnormally ({100 * repeated_at_detection:.1f}% "
            "of messages in the detection window)."
        ),

        (
            "Most LiDAR messages in the same window "
            "contained severely degraded usable range "
            f"content ({100 * degraded_at_detection:.1f}% "
            "severely degraded scans)."
        ),

        (
            "Median usable return content in the "
            "detection window fell to "
            f"{100 * valid_median_at_detection:.1f}% "
            f"(finite-return median "
            f"{100 * finite_median_at_detection:.1f}%)."
        ),
    ]


    contradicting_evidence = []


    if downstream[
        "observed"
    ]:

        downstream_time = (
            downstream[
                "first_event_s"
            ]
        )


        if (
            downstream_time
            is not None
            and
            downstream_time
            >= detection_time
        ):

            delay = (
                downstream_time
                -
                detection_time
            )


            supporting_evidence.append(
                (
                    "Navigation-state telemetry became "
                    "persistently stale after the first "
                    "LiDAR integrity event "
                    f"(approximately {delay:.3f} s later), "
                    "supporting downstream propagation."
                )
            )


        else:

            supporting_evidence.append(
                (
                    "Persistent navigation-state "
                    "staleness was also observed during "
                    "the incident."
                )
            )


    else:

        contradicting_evidence.append(
            (
                "No persistent downstream odometry or "
                "transform staleness above "
                f"{DOWNSTREAM_STALE_THRESHOLD_S:.1f} s "
                "was observed."
            )
        )


    evidence = {

        **summary_evidence,

        "scan_messages_at_detection":
            scans_at_detection,

        "nonmonotonic_fraction_at_detection":
            round(
                nonmonotonic_at_detection,
                4,
            ),

        "repeated_scan_fraction_at_detection":
            round(
                repeated_at_detection,
                4,
            ),

        "low_quality_scan_fraction_at_detection":
            round(
                degraded_at_detection,
                4,
            ),

        "finite_median_at_detection":
            round(
                finite_median_at_detection,
                4,
            ),

        "valid_median_at_detection":
            round(
                valid_median_at_detection,
                4,
            ),

        "downstream_navigation_staleness":
            downstream,
    }


    # =====================================================
    # RESULT
    # =====================================================

    return {

        "name":
            NAME,

        "score":
            round(
                float(
                    evidence_strength
                ),
                3,
            ),

        "available":
            True,

        "first_event_s":
            round(
                detection_time,
                3,
            ),

        "evidence":
            evidence,

        "supporting_evidence":
            supporting_evidence,

        "contradicting_evidence":
            contradicting_evidence,

        "failure_chain": [

            (
                "LiDAR stream integrity "
                "becomes unreliable"
            ),

            (
                "Timestamp ordering breaks, "
                "scan payloads repeat, and usable "
                "range content deteriorates"
            ),

            (
                "Laser-based state estimation receives "
                "inconsistent sensor observations"
            ),

            (
                "Odometry / localization / transform "
                "updates may subsequently become stale"
            ),

            (
                "Navigation reliability degrades"
            ),
        ],

        "recommended_checks": [

            (
                "Inspect the LiDAR driver and transport "
                "path for duplicated, replayed, or "
                "out-of-order LaserScan messages"
            ),

            (
                "Inspect LaserScan header timestamps "
                "and ROS time synchronization"
            ),

            (
                "Inspect finite and in-range LaserScan "
                "return fractions over time"
            ),

            (
                "Check LiDAR network bandwidth, DDS "
                "traffic, message rate, and compute load"
            ),

            (
                "Inspect laser odometry and downstream "
                "TF behavior immediately after the first "
                "LiDAR integrity event"
            ),
        ],
    }
