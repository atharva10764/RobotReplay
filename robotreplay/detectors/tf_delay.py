#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

from pathlib import Path

import numpy as np
import pandas as pd


from robotreplay.detectors.common import (
    healthy_values,
    no_detection,
    unavailable,
)


NAME = (
    "TF Delay / Discontinuity"
)


# =========================================================
# CONFIGURATION
# =========================================================

# A stale transform that begins only in the final portion of
# a recording can be caused by mission shutdown while the
# rosbag recorder continues running.
#
# This is NOT sufficient by itself to suppress a fault.
# Suppression also requires:
#
#   1. no transform recovery, and
#   2. another navigation-state stream becoming stale.
#
# This prevents isolated map→odom failures from being hidden.

TERMINAL_REGION_START_FRACTION = 0.85


# =========================================================
# HELPERS
# =========================================================

def _healthy_threshold(
    healthy: list[pd.DataFrame],
    column: str,
    *,
    minimum: float = 0.5,
) -> tuple[
    float | None,
    float | None,
]:

    values = healthy_values(
        healthy,
        column,
    )


    if values.empty:

        return (
            None,
            None,
        )


    healthy_max = float(
        values.max()
    )


    q995 = float(
        np.percentile(
            values,
            99.5,
        )
    )


    threshold = max(
        healthy_max
        * 1.5,

        q995
        * 2.0,

        minimum,
    )


    return (
        healthy_max,
        threshold,
    )


def _peak_numeric(
    frame: pd.DataFrame,
    column: str,
) -> float | None:

    if column not in frame.columns:

        return None


    values = pd.to_numeric(
        frame[
            column
        ],
        errors="coerce",
    ).dropna()


    if values.empty:

        return None


    return float(
        values.max()
    )


def _has_recovery(
    mission: pd.DataFrame,
    column: str,
    *,
    after_time: float,
    threshold: float,
) -> bool:

    if column not in mission.columns:

        return False


    later = mission[
        mission[
            "time"
        ]
        > after_time
    ]


    if later.empty:

        return False


    values = pd.to_numeric(
        later[
            column
        ],
        errors="coerce",
    ).dropna()


    if values.empty:

        return False


    return bool(
        (
            values
            <= threshold
        ).any()
    )


def _companion_stream_stale(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
) -> tuple[
    bool,
    dict,
]:

    evidence = {}


    companion_columns = [
        "tf_odom_base_age",
        "odom_age",
    ]


    stale = False


    for column in companion_columns:

        if column not in mission.columns:

            continue


        (
            healthy_max,
            threshold,
        ) = _healthy_threshold(
            healthy,
            column,
            minimum=0.5,
        )


        peak = _peak_numeric(
            mission,
            column,
        )


        if peak is None:

            continue


        evidence[
            f"{column}_peak_s"
        ] = round(
            peak,
            3,
        )


        if threshold is not None:

            evidence[
                f"{column}_healthy_max_s"
            ] = round(
                healthy_max,
                3,
            )


            evidence[
                f"{column}_stale_threshold_s"
            ] = round(
                threshold,
                3,
            )


            if peak > threshold:

                stale = True


        else:

            # If no healthy baseline exists for the
            # companion stream, do not use it to suppress
            # a TF diagnosis.

            evidence[
                f"{column}_baseline_available"
            ] = False


    return (
        stale,
        evidence,
    )


# =========================================================
# DETECTOR
# =========================================================

def detect_tf_delay(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
) -> dict:

    required = {
        "time",
        "tf_map_odom_age",
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
                "Missing telemetry features: "
                +
                ", ".join(
                    missing
                )
            ),
        )


    (
        healthy_max,
        threshold,
    ) = _healthy_threshold(
        healthy,
        "tf_map_odom_age",
        minimum=0.5,
    )


    if (
        healthy_max is None
        or
        threshold is None
    ):

        return unavailable(
            NAME,
            (
                "Healthy TF timing "
                "baseline is unavailable."
            ),
        )


    numeric_age = pd.to_numeric(
        mission[
            "tf_map_odom_age"
        ],
        errors="coerce",
    )


    candidates = mission[
        numeric_age
        > threshold
    ]


    baseline_evidence = {

        "healthy_max_map_odom_age_s":
            round(
                healthy_max,
                3,
            ),

        "tf_stale_threshold_s":
            round(
                threshold,
                3,
            ),
    }


    if candidates.empty:

        return no_detection(
            NAME,
            baseline_evidence,
        )


    first = candidates.iloc[
        0
    ]


    detection_time = float(
        first[
            "time"
        ]
    )


    mission_times = pd.to_numeric(
        mission[
            "time"
        ],
        errors="coerce",
    ).dropna()


    if mission_times.empty:

        return unavailable(
            NAME,
            (
                "Mission timing information "
                "is unavailable."
            ),
        )


    mission_start = float(
        mission_times.min()
    )


    mission_end = float(
        mission_times.max()
    )


    mission_duration = max(
        0.0,
        mission_end
        -
        mission_start,
    )


    if mission_duration > 0:

        onset_fraction = (
            (
                detection_time
                -
                mission_start
            )
            /
            mission_duration
        )

    else:

        onset_fraction = 0.0


    peak_map_odom = float(
        numeric_age.max()
    )


    peak_odom_base = (
        _peak_numeric(
            mission,
            "tf_odom_base_age",
        )
    )


    recovered = _has_recovery(
        mission,
        "tf_map_odom_age",
        after_time=
            detection_time,
        threshold=
            threshold,
    )


    (
        companion_stale,
        companion_evidence,
    ) = _companion_stream_stale(
        mission,
        healthy,
    )


    terminal_candidate = (
        onset_fraction
        >=
        TERMINAL_REGION_START_FRACTION
    )


    terminal_shutdown_tail = (
        terminal_candidate
        and
        not recovered
        and
        companion_stale
    )


    applicability_evidence = {

        "candidate_first_event_s":
            round(
                detection_time,
                3,
            ),

        "candidate_onset_fraction":
            round(
                onset_fraction,
                4,
            ),

        "candidate_onset_percent":
            round(
                100.0
                *
                onset_fraction,
                2,
            ),

        "terminal_region_start_percent":
            round(
                100.0
                *
                TERMINAL_REGION_START_FRACTION,
                1,
            ),

        "map_odom_recovered":
            bool(
                recovered
            ),

        "companion_navigation_stream_stale":
            bool(
                companion_stale
            ),

        "terminal_shutdown_tail_suppressed":
            bool(
                terminal_shutdown_tail
            ),

        **companion_evidence,
    }


    # -----------------------------------------------------
    # Suppress probable mission-shutdown / recording tail.
    #
    # Example:
    #
    # navigation stack ends near the end of mission
    # + map→odom stops
    # + odom/base stream also stops
    # + rosbag continues recording briefly
    #
    # This is not sufficient evidence for an isolated TF
    # failure.
    # -----------------------------------------------------

    if terminal_shutdown_tail:

        return no_detection(
            NAME,
            {

                **baseline_evidence,

                **applicability_evidence,

                "candidate_peak_map_odom_age_s":
                    round(
                        peak_map_odom,
                        3,
                    ),

                "suppression_reason":
                    (
                        "TF staleness began only in the "
                        "terminal portion of the recording, "
                        "did not recover, and another "
                        "navigation-state stream also became "
                        "stale. This is treated as probable "
                        "mission-shutdown / recording-tail "
                        "behavior rather than an isolated "
                        "TF fault."
                    ),
            },
        )


    # -----------------------------------------------------
    # Evidence score
    # -----------------------------------------------------

    ratio = (
        peak_map_odom
        /
        max(
            threshold,
            1e-9,
        )
    )


    evidence_strength = min(
        1.0,
        (
            0.55
            +
            0.08
            *
            min(
                ratio,
                6.0,
            )
        ),
    )


    evidence = {

        **baseline_evidence,

        **applicability_evidence,

        "map_odom_age_at_detection_s":
            round(
                float(
                    first[
                        "tf_map_odom_age"
                    ]
                ),
                3,
            ),

        "peak_map_odom_age_s":
            round(
                peak_map_odom,
                3,
            ),

        "peak_odom_base_age_s":
            (
                round(
                    peak_odom_base,
                    3,
                )
                if peak_odom_base
                is not None
                else None
            ),
    }


    supporting_evidence = [

        (
            f"map→odom age reached "
            f"{peak_map_odom:.3f} s versus a healthy "
            f"maximum of {healthy_max:.3f} s."
        ),
    ]


    if recovered:

        supporting_evidence.append(
            (
                "The transform stream later recovered, "
                "supporting a transient TF interruption "
                "rather than a terminal recording tail."
            )
        )


    if (
        peak_odom_base is not None
        and
        peak_odom_base
        <= threshold
    ):

        supporting_evidence.append(
            (
                f"odom→base remained fresh "
                f"(peak age {peak_odom_base:.3f} s), "
                "localizing the stale transform behavior "
                "to the map→odom layer."
            )
        )


    elif (
        companion_stale
        and
        not terminal_candidate
    ):

        supporting_evidence.append(
            (
                "Additional navigation-state telemetry "
                "also became stale while substantial "
                "mission time remained, supporting a "
                "mid-mission navigation/transform "
                "subsystem interruption."
            )
        )


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
            [],

        "failure_chain": [

            (
                "map→odom transform "
                "stops updating"
            ),

            (
                "Localization transform "
                "becomes stale"
            ),

            (
                "Navigation pose transforms "
                "become unreliable"
            ),

            (
                "Planning or control may "
                "pause, fail, or recover"
            ),
        ],

        "recommended_checks": [

            (
                "Inspect the localization / "
                "map→odom transform publisher"
            ),

            (
                "Inspect transform timestamps "
                "and publication continuity"
            ),

            (
                "Verify ROS time "
                "synchronization"
            ),

            (
                "Check map, odom, and base "
                "frame configuration"
            ),
        ],
    }


# =========================================================
# CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--healthy",
        nargs="+",
        required=True,
    )


    parser.add_argument(
        "--mission",
        required=True,
    )


    parser.add_argument(
        "--output",
        default=(
            "data/results/"
            "tf_delay_diagnosis.json"
        ),
    )


    args = parser.parse_args()


    healthy = [

        pd.read_csv(
            path
        )

        for path
        in args.healthy
    ]


    mission = pd.read_csv(
        args.mission
    )


    result = detect_tf_delay(
        mission,
        healthy,
    )


    print(
        "\n=== RobotReplay TF Diagnosis ==="
    )


    if not result.get(
        "available",
        True,
    ):

        print(
            "\nTF diagnosis unavailable."
        )


        print(
            result.get(
                "reason",
                "",
            )
        )

        return


    if result[
        "score"
    ] <= 0:

        print(
            "\nNo supported map→odom TF "
            "delay/discontinuity detected."
        )


        evidence = result.get(
            "evidence",
            {}
        )


        if evidence.get(
            "terminal_shutdown_tail_suppressed"
        ):

            print(
                "Terminal shutdown / "
                "recording tail suppressed."
            )


        return


    print(
        "\nMOST LIKELY ROOT CAUSE"
    )


    print(
        NAME
    )


    print(
        f"\nFirst diagnostic event: "
        f"{result['first_event_s']:.3f} s"
    )


    print(
        f"Evidence strength:      "
        f"{result['score']:.3f}"
    )


    output = Path(
        args.output
    )


    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output.write_text(
        json.dumps(
            result,
            indent=2,
        )
    )


    print(
        f"\nSaved diagnosis: "
        f"{output}"
    )


    print(
        "TF diagnosis: PASS"
    )


if __name__ == "__main__":

    main()
