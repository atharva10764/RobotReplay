#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from robotreplay.detectors.common import (
    healthy_values,
    no_detection,
    robust_threshold,
    unavailable,
)


NAME = "Localization Jump"


def detect_localization_jump(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
    correlation_window_s: float = 0.5,
) -> dict:

    required = {
        "time",
        "amcl_position_step",
        "tf_map_odom_position_step",
    }

    missing = sorted(
        required
        - set(
            mission.columns
        )
    )

    if missing:

        return unavailable(
            NAME,
            (
                "Missing telemetry features: "
                + ", ".join(missing)
            ),
        )


    amcl_baseline = healthy_values(
        healthy,
        "amcl_position_step",
    )

    tf_baseline = healthy_values(
        healthy,
        "tf_map_odom_position_step",
    )


    if (
        amcl_baseline.empty
        or tf_baseline.empty
    ):

        return unavailable(
            NAME,
            "Healthy localization baseline is unavailable.",
        )


    amcl_threshold = robust_threshold(
        amcl_baseline
    )

    tf_threshold = robust_threshold(
        tf_baseline
    )


    baseline_evidence = {
        "healthy_amcl_threshold_m":
            round(
                amcl_threshold,
                3,
            ),

        "healthy_tf_step_threshold_m":
            round(
                tf_threshold,
                3,
            ),

        "correlation_window_s":
            round(
                correlation_window_s,
                3,
            ),
    }


    amcl_events = mission.loc[
        mission["amcl_position_step"]
        > amcl_threshold,
        [
            "time",
            "amcl_position_step",
        ],
    ].dropna()


    tf_events = mission.loc[
        mission["tf_map_odom_position_step"]
        > tf_threshold,
        [
            "time",
            "tf_map_odom_position_step",
        ],
    ].dropna()


    if (
        amcl_events.empty
        or tf_events.empty
    ):

        return no_detection(
            NAME,
            baseline_evidence,
        )


    matched_events = []


    for _, amcl_event in amcl_events.iterrows():

        separation = (
            tf_events["time"]
            - float(
                amcl_event[
                    "time"
                ]
            )
        ).abs()


        closest_index = separation.idxmin()

        closest_separation = float(
            separation.loc[
                closest_index
            ]
        )


        if (
            closest_separation
            <= correlation_window_s
        ):

            tf_event = tf_events.loc[
                closest_index
            ]

            detection_time = max(
                float(
                    amcl_event["time"]
                ),
                float(
                    tf_event["time"]
                ),
            )


            matched_events.append(
                (
                    detection_time,
                    amcl_event,
                    tf_event,
                    closest_separation,
                )
            )


    if not matched_events:

        return no_detection(
            NAME,
            baseline_evidence,
        )


    matched_events.sort(
        key=lambda item: item[0]
    )


    (
        detection_time,
        amcl_event,
        tf_event,
        separation,
    ) = matched_events[0]


    amcl_step = float(
        amcl_event[
            "amcl_position_step"
        ]
    )

    tf_step = float(
        tf_event[
            "tf_map_odom_position_step"
        ]
    )


    amcl_ratio = (
        amcl_step
        / amcl_threshold
    )

    tf_ratio = (
        tf_step
        / tf_threshold
    )


    evidence_strength = min(
        1.0,
        (
            0.50
            + 0.15
            * min(
                amcl_ratio,
                2.0,
            )
            + 0.15
            * min(
                tf_ratio,
                2.0,
            )
        ),
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

        "evidence": {

            **baseline_evidence,

            "amcl_position_step_m":
                round(
                    amcl_step,
                    3,
                ),

            "map_odom_position_step_m":
                round(
                    tf_step,
                    3,
                ),

            "evidence_time_separation_s":
                round(
                    separation,
                    3,
                ),
        },

        "supporting_evidence": [

            (
                f"AMCL position changed "
                f"{amcl_step:.3f} m, above the healthy "
                f"threshold of {amcl_threshold:.3f} m."
            ),

            (
                f"map→odom changed "
                f"{tf_step:.3f} m within "
                f"{separation:.3f} s of the AMCL "
                f"discontinuity."
            ),
        ],

        "contradicting_evidence":
            [],

        "failure_chain": [

            "Localization estimate changes abruptly",

            "map→odom correction changes abruptly",

            (
                "Robot pose used by Nav2 "
                "becomes inconsistent"
            ),

            (
                "Navigation controller reacts "
                "to changed pose"
            ),
        ],

        "recommended_checks": [

            (
                "Check AMCL initialization "
                "and pose resets"
            ),

            (
                "Inspect localization sensor "
                "consistency"
            ),

            (
                "Inspect unexpected "
                "/initialpose publications"
            ),

            (
                "Verify map-to-odom "
                "localization updates"
            ),
        ],
    }


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
            "localization_diagnosis.json"
        ),
    )

    args = parser.parse_args()


    healthy = [
        pd.read_csv(path)
        for path
        in args.healthy
    ]

    mission = pd.read_csv(
        args.mission
    )


    result = detect_localization_jump(
        mission,
        healthy,
    )


    print(
        "\n=== RobotReplay "
        "Localization Diagnosis ==="
    )


    if result["score"] <= 0:

        print(
            "\nNo localization-jump "
            "signature detected."
        )

        return


    print(
        "\nMOST LIKELY ROOT CAUSE"
    )

    print(
        "Localization Jump"
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
        f"\nSaved diagnosis: {output}"
    )

    print(
        "Localization diagnosis: PASS"
    )


if __name__ == "__main__":
    main()
