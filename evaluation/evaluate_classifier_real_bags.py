#!/usr/bin/env python3

from __future__ import annotations

import sys

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import pandas as pd


from robotreplay.ml.fault_classifier import (
    classify_mission,
)


CASES = [

    {
        "name":
            "Healthy Mission",

        "features":
            PROJECT_ROOT
            / "data/runtime/healthy_003_features.csv",

        "fallback":
            PROJECT_ROOT
            / "data/features/healthy_003.csv",

        "truth":
            "Healthy",
    },

    {
        "name":
            "Localization Jump",

        "features":
            PROJECT_ROOT
            / "data/runtime/localization_jump_001_features.csv",

        "fallback":
            PROJECT_ROOT
            / "data/features/localization_jump_001.csv",

        "truth":
            "Localization Jump",
    },

    {
        "name":
            "LiDAR Dropout",

        "features":
            PROJECT_ROOT
            / "data/runtime/lidar_dropout_001_features.csv",

        "fallback":
            PROJECT_ROOT
            / "data/features/lidar_dropout_001.csv",

        "truth":
            "LiDAR Dropout",
    },

    {
        "name":
            "TF Delay / Discontinuity",

        "features":
            PROJECT_ROOT
            / "data/runtime/tf_delay_001_features.csv",

        "fallback":
            PROJECT_ROOT
            / "data/features/tf_delay_001.csv",

        "truth":
            "TF Delay / Discontinuity",
    },

    {
        "name":
            "Wheel / Encoder Direction Mismatch",

        "features":
            PROJECT_ROOT
            / "data/runtime/wheel_mismatch_001_features.csv",

        "fallback":
            PROJECT_ROOT
            / "data/features/wheel_mismatch_001.csv",

        "truth":
            "Wheel / Encoder Direction Mismatch",
    },
]


def resolve_features(
    case: dict,
) -> Path:

    if case[
        "features"
    ].is_file():

        return case[
            "features"
        ]


    if case[
        "fallback"
    ].is_file():

        return case[
            "fallback"
        ]


    raise FileNotFoundError(
        (
            f"No features found for "
            f"{case['name']}"
        )
    )


def main() -> None:

    print(
        "\n=== RobotReplay Classifier "
        "Real-Bag Evaluation ===\n"
    )


    rows = []


    for case in CASES:

        path = resolve_features(
            case
        )


        df = pd.read_csv(
            path
        )


        result = classify_mission(
            df
        )


        prediction = result[
            "prediction"
        ]


        probability = float(
            result[
                "probability"
            ]
        )


        correct = (
            prediction
            ==
            case[
                "truth"
            ]
        )


        selected_window = (
            result.get(
                "selected_window"
            )
        )


        if selected_window:

            window_text = (
                f"{selected_window['window_start_s']:.1f}"
                f"–"
                f"{selected_window['window_end_s']:.1f} s"
            )

        else:

            window_text = (
                "—"
            )


        rows.append(
            {
                "scenario":
                    case[
                        "name"
                    ],

                "ground_truth":
                    case[
                        "truth"
                    ],

                "prediction":
                    prediction,

                "probability":
                    probability,

                "correct":
                    correct,

                "selected_window":
                    window_text,

                "windows_evaluated":
                    result[
                        "windows_evaluated"
                    ],
            }
        )


        print(
            "----------------------------------------"
        )


        print(
            f"Scenario:     "
            f"{case['name']}"
        )


        print(
            f"Ground truth: "
            f"{case['truth']}"
        )


        print(
            f"ML prediction:"
            f" {prediction}"
        )


        print(
            f"Probability:  "
            f"{100 * probability:.2f}%"
        )


        print(
            f"Correct:      "
            f"{'YES' if correct else 'NO'}"
        )


        print(
            f"Window:       "
            f"{window_text}"
        )


        print(
            "\nClass probabilities:"
        )


        for ranked in result[
            "ranked_probabilities"
        ]:

            print(
                (
                    f"  "
                    f"{ranked['class']:<38} "
                    f"{100 * ranked['probability']:6.2f}%"
                )
            )


        print()


    results = pd.DataFrame(
        rows
    )


    correct_count = int(
        results[
            "correct"
        ].sum()
    )


    total = len(
        results
    )


    accuracy = (
        correct_count
        /
        total
    )


    output = (
        PROJECT_ROOT
        / "data/results/"
        "fault_classifier_real_bag_evaluation.csv"
    )


    results.to_csv(
        output,
        index=False,
    )


    print(
        "========================================"
    )


    print(
        "REAL / CONTROLLED BAG SUMMARY"
    )


    print(
        "========================================"
    )


    print(
        results.to_string(
            index=False
        )
    )


    print(
        "\nCorrect diagnoses: "
        f"{correct_count} / {total}"
    )


    print(
        "Bag-level accuracy: "
        f"{100 * accuracy:.2f}%"
    )


    print(
        (
            "\nNote: this is a five-recording "
            "RobotReplay validation set and is "
            "reported separately from the 200-sample "
            "held-out augmentation benchmark."
        )
    )


    print(
        f"\nSaved: {output}"
    )


    print(
        "\nReal-bag classifier evaluation: PASS"
    )


if __name__ == "__main__":

    main()
