#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from robotreplay.detectors.common import (
    no_detection,
    unavailable,
)


NAME = (
    "Wheel / Encoder Direction Mismatch"
)


def fit_wheel_model(
    healthy: list[pd.DataFrame],
    wheel_column: str,
):

    rows = []


    for df in healthy:

        required = {
            "cmd_vx",
            "cmd_wz",
            wheel_column,
        }


        if not required.issubset(
            df.columns
        ):

            continue


        subset = df[
            [
                "cmd_vx",
                "cmd_wz",
                wheel_column,
            ]
        ].dropna()


        subset = subset[
            (
                subset[
                    "cmd_vx"
                ].abs()
                > 0.02
            )
            |
            (
                subset[
                    "cmd_wz"
                ].abs()
                > 0.02
            )
        ]


        rows.append(
            subset
        )


    if not rows:

        return None


    data = pd.concat(
        rows,
        ignore_index=True,
    )


    if len(data) < 20:

        return None


    X = np.column_stack(
        [
            data[
                "cmd_vx"
            ].to_numpy(
                float
            ),

            data[
                "cmd_wz"
            ].to_numpy(
                float
            ),

            np.ones(
                len(data)
            ),
        ]
    )


    y = data[
        wheel_column
    ].to_numpy(
        float
    )


    beta, *_ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )


    predicted = (
        X
        @ beta
    )


    residual = np.abs(
        y
        - predicted
    )


    residual_threshold = max(

        float(
            np.percentile(
                residual,
                99.0,
            )
        )
        * 1.25,

        0.5,
    )


    predicted_magnitude_floor = max(

        float(
            np.percentile(
                np.abs(y),
                10.0,
            )
        ),

        0.2,
    )


    return (
        beta,
        residual_threshold,
        predicted_magnitude_floor,
    )


def detect_wheel_mismatch(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
) -> dict:

    required = {
        "time",
        "cmd_vx",
        "cmd_wz",
        "left_wheel_vel",
        "right_wheel_vel",
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


    left_model = fit_wheel_model(
        healthy,
        "left_wheel_vel",
    )


    right_model = fit_wheel_model(
        healthy,
        "right_wheel_vel",
    )


    if (
        left_model is None
        or right_model is None
    ):

        return unavailable(
            NAME,
            (
                "Healthy command-to-wheel "
                "baseline is unavailable."
            ),
        )


    (
        left_beta,
        left_residual_threshold,
        left_magnitude_floor,
    ) = left_model


    (
        right_beta,
        right_residual_threshold,
        right_magnitude_floor,
    ) = right_model


    df = mission.copy()


    X = np.column_stack(
        [
            df[
                "cmd_vx"
            ].fillna(
                0
            ).to_numpy(
                float
            ),

            df[
                "cmd_wz"
            ].fillna(
                0
            ).to_numpy(
                float
            ),

            np.ones(
                len(df)
            ),
        ]
    )


    df[
        "expected_left_wheel_vel"
    ] = (
        X
        @ left_beta
    )


    df[
        "expected_right_wheel_vel"
    ] = (
        X
        @ right_beta
    )


    df[
        "left_wheel_residual"
    ] = (
        df[
            "left_wheel_vel"
        ]
        - df[
            "expected_left_wheel_vel"
        ]
    ).abs()


    df[
        "right_wheel_residual"
    ] = (
        df[
            "right_wheel_vel"
        ]
        - df[
            "expected_right_wheel_vel"
        ]
    ).abs()


    active_motion = (

        (
            df[
                "cmd_vx"
            ].abs()
            > 0.05
        )
        |
        (
            df[
                "cmd_wz"
            ].abs()
            > 0.10
        )
    )


    left_sign_bad = (

        (
            df[
                "expected_left_wheel_vel"
            ].abs()
            > left_magnitude_floor
        )
        &
        (
            df[
                "left_wheel_vel"
            ].abs()
            > left_magnitude_floor
        )
        &
        (
            np.sign(
                df[
                    "expected_left_wheel_vel"
                ]
            )
            !=
            np.sign(
                df[
                    "left_wheel_vel"
                ]
            )
        )
    )


    right_sign_bad = (

        (
            df[
                "expected_right_wheel_vel"
            ].abs()
            > right_magnitude_floor
        )
        &
        (
            df[
                "right_wheel_vel"
            ].abs()
            > right_magnitude_floor
        )
        &
        (
            np.sign(
                df[
                    "expected_right_wheel_vel"
                ]
            )
            !=
            np.sign(
                df[
                    "right_wheel_vel"
                ]
            )
        )
    )


    strong_residual = (

        (
            df[
                "left_wheel_residual"
            ]
            > left_residual_threshold
        )
        |
        (
            df[
                "right_wheel_residual"
            ]
            > right_residual_threshold
        )
    )


    signature = (

        active_motion
        &
        (
            left_sign_bad
            |
            right_sign_bad
        )
        &
        strong_residual
    )


    # Require persistence across at least
    # two neighboring 0.2 s windows.
    persistent = (

        signature
        &
        (
            signature.shift(
                1,
                fill_value=False,
            )
            |
            signature.shift(
                -1,
                fill_value=False,
            )
        )
    )


    candidates = df[
        persistent
    ]


    baseline_evidence = {

        "left_residual_threshold_rad_s":
            round(
                left_residual_threshold,
                3,
            ),

        "right_residual_threshold_rad_s":
            round(
                right_residual_threshold,
                3,
            ),

        "model_type":
            (
                "healthy command-to-wheel "
                "linear baseline"
            ),
    }


    if candidates.empty:

        return no_detection(
            NAME,
            baseline_evidence,
        )


    first = candidates.iloc[0]


    mismatch_windows = int(
        persistent.sum()
    )


    left_observed = float(
        first[
            "left_wheel_vel"
        ]
    )


    right_observed = float(
        first[
            "right_wheel_vel"
        ]
    )


    left_expected = float(
        first[
            "expected_left_wheel_vel"
        ]
    )


    right_expected = float(
        first[
            "expected_right_wheel_vel"
        ]
    )


    if bool(
        right_sign_bad.loc[
            first.name
        ]
    ):

        mismatched_wheel = "right"

    else:

        mismatched_wheel = "left"


    evidence_strength = min(
        1.0,
        (
            0.65
            + 0.01
            * mismatch_windows
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
                float(
                    first[
                        "time"
                    ]
                ),
                3,
            ),

        "evidence": {

            **baseline_evidence,

            "commanded_linear_velocity_mps":
                round(
                    float(
                        first[
                            "cmd_vx"
                        ]
                    ),
                    3,
                ),

            "commanded_angular_velocity_rad_s":
                round(
                    float(
                        first[
                            "cmd_wz"
                        ]
                    ),
                    3,
                ),

            "left_wheel_velocity_rad_s":
                round(
                    left_observed,
                    3,
                ),

            "right_wheel_velocity_rad_s":
                round(
                    right_observed,
                    3,
                ),

            "expected_left_wheel_velocity_rad_s":
                round(
                    left_expected,
                    3,
                ),

            "expected_right_wheel_velocity_rad_s":
                round(
                    right_expected,
                    3,
                ),

            "mismatched_wheel":
                mismatched_wheel,

            "persistent_mismatch_windows":
                mismatch_windows,
        },

        "supporting_evidence": [

            (
                f"Observed {mismatched_wheel}-wheel "
                f"direction contradicts the direction "
                f"predicted from healthy "
                f"command-to-wheel behaviour."
            ),

            (
                f"Mismatch persisted for "
                f"{mismatch_windows} synchronized "
                f"windows."
            ),
        ],

        "contradicting_evidence":
            [],

        "failure_chain": [

            (
                "Velocity command is issued"
            ),

            (
                "Wheel feedback contradicts "
                "expected differential-drive response"
            ),

            (
                "Encoder or motor direction becomes "
                "inconsistent with commanded motion"
            ),

            (
                "Odometry or physical motion can "
                "diverge from the command"
            ),

            (
                "Navigation may lose progress "
                "or trigger recovery"
            ),
        ],

        "recommended_checks": [

            (
                "Verify motor direction "
                "and encoder polarity"
            ),

            (
                "Check encoder sign convention"
            ),

            (
                "Inspect ros2_control / diff-drive "
                "wheel configuration"
            ),

            (
                "Verify URDF wheel-axis orientation"
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
            "wheel_mismatch_diagnosis.json"
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


    result = detect_wheel_mismatch(
        mission,
        healthy,
    )


    print(
        "\n=== RobotReplay "
        "Wheel Diagnosis ==="
    )


    if result["score"] <= 0:

        print(
            "\nNo wheel / encoder "
            "direction mismatch detected."
        )

        return


    print(
        "\nMOST LIKELY ROOT CAUSE"
    )

    print(
        "Wheel / Encoder Direction Mismatch"
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
        "Wheel diagnosis: PASS"
    )


if __name__ == "__main__":
    main()
