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


NAME = "LiDAR Dropout"


def detect_lidar_dropout(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
) -> dict:

    required = {
        "time",
        "scan_age",
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


    scan_age = healthy_values(
        healthy,
        "scan_age",
    )


    if scan_age.empty:

        return unavailable(
            NAME,
            (
                "Healthy LiDAR timing "
                "baseline is unavailable."
            ),
        )


    healthy_max = float(
        scan_age.max()
    )

    healthy_q995 = float(
        np.percentile(
            scan_age,
            99.5,
        )
    )


    threshold = max(

        healthy_max * 1.5,

        healthy_q995 * 2.0,

        0.5,
    )


    candidates = mission[
        mission["scan_age"]
        > threshold
    ]


    baseline_evidence = {

        "healthy_max_scan_age_s":
            round(
                healthy_max,
                3,
            ),

        "scan_stale_threshold_s":
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


    first = candidates.iloc[0]


    detection_time = float(
        first["time"]
    )


    peak_age = float(
        mission[
            "scan_age"
        ].max()
    )


    max_gap = None

    if (
        "scan_gap"
        in mission.columns
    ):

        max_gap = float(
            mission[
                "scan_gap"
            ].max()
        )


    recovery_rows = mission[
        (
            mission["time"]
            > detection_time
        )
        &
        (
            mission["scan_age"]
            <= threshold
        )
    ]


    recovery_time = None

    if not recovery_rows.empty:

        recovery_time = float(
            recovery_rows.iloc[0][
                "time"
            ]
        )


    ratio = (
        peak_age
        / max(
            threshold,
            1e-9,
        )
    )


    evidence_strength = min(
        1.0,
        (
            0.55
            + 0.08
            * min(
                ratio,
                6.0,
            )
        ),
    )


    evidence = {

        **baseline_evidence,

        "scan_age_at_detection_s":
            round(
                float(
                    first[
                        "scan_age"
                    ]
                ),
                3,
            ),

        "peak_scan_age_s":
            round(
                peak_age,
                3,
            ),

        "maximum_scan_gap_s":
            (
                round(
                    max_gap,
                    3,
                )
                if max_gap is not None
                else None
            ),

        "recovery_time_s":
            (
                round(
                    recovery_time,
                    3,
                )
                if recovery_time is not None
                else None
            ),
    }


    supporting_evidence = [

        (
            f"LiDAR scan age reached "
            f"{peak_age:.3f} s versus a healthy "
            f"maximum of {healthy_max:.3f} s."
        ),
    ]


    if max_gap is not None:

        supporting_evidence.append(

            (
                f"Observed scan gap reached "
                f"{max_gap:.3f} s."
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
                "LiDAR scan stream "
                "becomes stale"
            ),

            (
                "Navigation receives "
                "outdated obstacle data"
            ),

            (
                "Local costmap can no "
                "longer be refreshed reliably"
            ),

            (
                "Controller may slow, stop, "
                "or trigger recovery"
            ),
        ],

        "recommended_checks": [

            (
                "Check /scan publisher "
                "health and frequency"
            ),

            (
                "Inspect LiDAR driver or "
                "transport interruptions"
            ),

            (
                "Verify sensor power "
                "and communication"
            ),

            (
                "Check Nav2 observation-source "
                "timeout settings"
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
            "lidar_dropout_diagnosis.json"
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


    result = detect_lidar_dropout(
        mission,
        healthy,
    )


    print(
        "\n=== RobotReplay "
        "LiDAR Diagnosis ==="
    )


    if result["score"] <= 0:

        print(
            "\nNo LiDAR dropout detected."
        )

        return


    print(
        "\nMOST LIKELY ROOT CAUSE"
    )

    print(
        "LiDAR Dropout"
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
        "LiDAR diagnosis: PASS"
    )


if __name__ == "__main__":
    main()
