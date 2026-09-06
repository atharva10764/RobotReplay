#!/usr/bin/env python3

from __future__ import annotations

import random
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


import numpy as np
import pandas as pd


from robotreplay.ml.fault_features import (
    FEATURE_NAMES,
    summarize_fault_window,
)


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_SEED = 42

WINDOW_SECONDS = 20.0


TRAIN_SAMPLES_PER_CLASS_PER_SOURCE = 60

TEST_SAMPLES_PER_CLASS = 40


CLASSES = [

    "Healthy",

    "Localization Jump",

    "LiDAR Dropout",

    "TF Delay / Discontinuity",

    "Wheel / Encoder Direction Mismatch",
]


SOURCES = {

    "healthy_001":
        PROJECT_ROOT
        / "data/features/healthy_001.csv",

    "healthy_002":
        PROJECT_ROOT
        / "data/features/healthy_002.csv",

    "healthy_003":
        PROJECT_ROOT
        / "data/features/healthy_003.csv",
}


TRAIN_SOURCES = [

    "healthy_001",

    "healthy_002",
]


TEST_SOURCES = [

    "healthy_003",
]


OUTPUT = (
    PROJECT_ROOT
    / "data/ml/fault_classifier_dataset.csv"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


random.seed(
    RANDOM_SEED
)


# =========================================================
# WINDOW SELECTION
# =========================================================

def choose_window(
    df: pd.DataFrame,
    *,
    require_motion: bool = False,
) -> pd.DataFrame:

    earliest = float(
        df["time"].min()
    )


    latest_start = float(
        df["time"].max()
        - WINDOW_SECONDS
    )


    if latest_start <= earliest:

        window = df.copy()

        window["time"] = (
            window["time"]
            - window["time"].min()
        )

        return window.reset_index(
            drop=True
        )


    best = None


    for _ in range(
        150
    ):

        start = float(
            rng.uniform(
                earliest,
                latest_start,
            )
        )


        end = (
            start
            + WINDOW_SECONDS
        )


        window = df[
            (
                df["time"]
                >= start
            )
            &
            (
                df["time"]
                <= end
            )
        ].copy()


        if window.empty:
            continue


        window["time"] = (
            window["time"]
            - start
        )


        if require_motion:

            cmd_vx = pd.to_numeric(
                window.get(
                    "cmd_vx",
                    0.0,
                ),
                errors="coerce",
            ).fillna(
                0
            )


            cmd_wz = pd.to_numeric(
                window.get(
                    "cmd_wz",
                    0.0,
                ),
                errors="coerce",
            ).fillna(
                0
            )


            observable = (
                (cmd_vx > 0.05)
                &
                (cmd_wz.abs() < 0.45)
            )


            observable_count = int(
                observable.sum()
            )


            if observable_count >= 15:

                return window.reset_index(
                    drop=True
                )


            if (
                best is None
                or observable_count
                > best[
                    0
                ]
            ):

                best = (
                    observable_count,
                    window.copy(),
                )


        else:

            return window.reset_index(
                drop=True
            )


    if best is not None:

        return best[
            1
        ].reset_index(
            drop=True
        )


    raise RuntimeError(
        "Unable to select telemetry window."
    )


# =========================================================
# FAULT INJECTION
# =========================================================

def inject_lidar_dropout(
    window: pd.DataFrame,
):

    df = window.copy()


    start = float(
        rng.uniform(
            5.0,
            11.0,
        )
    )


    duration = float(
        rng.uniform(
            0.7,
            7.0,
        )
    )


    end = min(
        start
        + duration,

        WINDOW_SECONDS
        - 0.4,
    )


    mask = (
        (df["time"] >= start)
        &
        (df["time"] < end)
    )


    if "scan_age" in df.columns:

        stale_age = (
            df.loc[
                mask,
                "time",
            ]
            - start
            + 0.20
        )


        original = pd.to_numeric(
            df.loc[
                mask,
                "scan_age",
            ],
            errors="coerce",
        ).fillna(
            0
        )


        df.loc[
            mask,
            "scan_age",
        ] = np.maximum(
            original,
            stale_age,
        )


    if "scan_gap" in df.columns:

        recovery = df[
            df["time"]
            >= end
        ]


        if not recovery.empty:

            index = (
                recovery.index[
                    0
                ]
            )


            existing = pd.to_numeric(
                pd.Series(
                    [
                        df.loc[
                            index,
                            "scan_gap",
                        ]
                    ]
                ),
                errors="coerce",
            ).fillna(
                0
            ).iloc[
                0
            ]


            df.loc[
                index,
                "scan_gap",
            ] = max(
                float(
                    existing
                ),

                (
                    end
                    - start
                    + float(
                        rng.uniform(
                            0.05,
                            0.35,
                        )
                    )
                ),
            )


    return (
        df,
        {
            "fault_start_s":
                start,

            "fault_duration_s":
                end
                - start,
        },
    )


def inject_tf_delay(
    window: pd.DataFrame,
):

    df = window.copy()


    start = float(
        rng.uniform(
            5.0,
            11.0,
        )
    )


    duration = float(
        rng.uniform(
            0.7,
            7.0,
        )
    )


    end = min(
        start
        + duration,

        WINDOW_SECONDS
        - 0.4,
    )


    mask = (
        (df["time"] >= start)
        &
        (df["time"] < end)
    )


    if "tf_map_odom_age" in df.columns:

        stale_age = (
            df.loc[
                mask,
                "time",
            ]
            - start
            + 0.20
        )


        original = pd.to_numeric(
            df.loc[
                mask,
                "tf_map_odom_age",
            ],
            errors="coerce",
        ).fillna(
            0
        )


        df.loc[
            mask,
            "tf_map_odom_age",
        ] = np.maximum(
            original,
            stale_age,
        )


    # Important:
    # odom→base remains healthy.


    return (
        df,
        {
            "fault_start_s":
                start,

            "fault_duration_s":
                end
                - start,
        },
    )


def inject_localization_jump(
    window: pd.DataFrame,
):

    df = window.copy()


    event_time = float(
        rng.uniform(
            5.0,
            15.0,
        )
    )


    offset = float(
        rng.uniform(
            0.52,
            2.50,
        )
    )


    nearest_index = (
        df[
            "time"
        ]
        .sub(
            event_time
        )
        .abs()
        .idxmin()
    )


    if "amcl_position_step" in df.columns:

        existing = pd.to_numeric(
            pd.Series(
                [
                    df.loc[
                        nearest_index,
                        "amcl_position_step",
                    ]
                ]
            ),
            errors="coerce",
        ).fillna(
            0
        ).iloc[
            0
        ]


        df.loc[
            nearest_index,
            "amcl_position_step",
        ] = max(
            float(
                existing
            ),
            offset,
        )


    position = (
        df.index.get_loc(
            nearest_index
        )
    )


    shift = int(
        rng.integers(
            -1,
            2,
        )
    )


    tf_position = max(
        0,

        min(
            position
            + shift,

            len(
                df
            )
            - 1,
        ),
    )


    tf_index = (
        df.index[
            tf_position
        ]
    )


    tf_offset = (
        offset
        * float(
            rng.uniform(
                0.85,
                1.08,
            )
        )
    )


    if (
        "tf_map_odom_position_step"
        in df.columns
    ):

        existing = pd.to_numeric(
            pd.Series(
                [
                    df.loc[
                        tf_index,
                        "tf_map_odom_position_step",
                    ]
                ]
            ),
            errors="coerce",
        ).fillna(
            0
        ).iloc[
            0
        ]


        df.loc[
            tf_index,
            "tf_map_odom_position_step",
        ] = max(
            float(
                existing
            ),
            tf_offset,
        )


    return (
        df,
        {
            "fault_start_s":
                float(
                    df.loc[
                        nearest_index,
                        "time",
                    ]
                ),

            "localization_offset_m":
                offset,
        },
    )


def inject_wheel_mismatch(
    window: pd.DataFrame,
):

    df = window.copy()


    cmd_vx = pd.to_numeric(
        df.get(
            "cmd_vx",
            0.0,
        ),
        errors="coerce",
    ).fillna(
        0
    )


    cmd_wz = pd.to_numeric(
        df.get(
            "cmd_wz",
            0.0,
        ),
        errors="coerce",
    ).fillna(
        0
    )


    # -----------------------------------------------------
    # Find observable translational motion.
    # -----------------------------------------------------

    candidates = df[
        (
            cmd_vx
            > 0.05
        )
        &
        (
            cmd_wz.abs()
            < 0.45
        )
        &
        (
            df["time"]
            > 2.0
        )
        &
        (
            df["time"]
            < 16.5
        )
    ]


    if candidates.empty:

        candidates = df[
            (
                cmd_vx.abs()
                > 0.05
            )
            &
            (
                df["time"]
                > 2.0
            )
            &
            (
                df["time"]
                < 16.5
            )
        ]


    if candidates.empty:

        raise RuntimeError(
            (
                "Wheel fault window contains "
                "no observable commanded motion."
            )
        )


    selected_index = random.choice(
        list(
            candidates.index
        )
    )


    start = float(
        df.loc[
            selected_index,
            "time",
        ]
    )


    duration = float(
        rng.uniform(
            1.0,
            5.0,
        )
    )


    end = min(
        start
        + duration,

        WINDOW_SECONDS
        - 0.3,
    )


    mask = (
        (df["time"] >= start)
        &
        (df["time"] < end)
        &
        (cmd_vx.abs() > 0.04)
    )


    # Ensure several observable samples.
    if int(
        mask.sum()
    ) < 4:

        end = min(
            start
            + 5.0,

            WINDOW_SECONDS
            - 0.2,
        )


        mask = (
            (df["time"] >= start)
            &
            (df["time"] < end)
        )


    wheel = random.choice(
        [
            "left",
            "right",
        ]
    )


    column = (
        "left_wheel_vel"
        if wheel == "left"
        else
        "right_wheel_vel"
    )


    if column not in df.columns:

        raise RuntimeError(
            (
                f"Wheel telemetry missing: "
                f"{column}"
            )
        )


    values = pd.to_numeric(
        df.loc[
            mask,
            column,
        ],
        errors="coerce",
    ).fillna(
        0
    )


    # Reverse measured encoder / wheel direction.
    df.loc[
        mask,
        column,
    ] = (
        -1.0
        * values
    )


    return (
        df,
        {
            "fault_start_s":
                start,

            "fault_duration_s":
                end
                - start,

            "modified_wheel":
                wheel,

            "modified_samples":
                int(
                    mask.sum()
                ),
        },
    )


# =========================================================
# CLASS DISPATCH
# =========================================================

def augment(
    window: pd.DataFrame,
    label: str,
):

    if label == "Healthy":

        return (
            window.copy(),
            {
                "fault_start_s":
                    np.nan,
            },
        )


    if label == "Localization Jump":

        return inject_localization_jump(
            window
        )


    if label == "LiDAR Dropout":

        return inject_lidar_dropout(
            window
        )


    if label == "TF Delay / Discontinuity":

        return inject_tf_delay(
            window
        )


    if (
        label
        == "Wheel / Encoder Direction Mismatch"
    ):

        return inject_wheel_mismatch(
            window
        )


    raise ValueError(
        f"Unknown class: {label}"
    )


# =========================================================
# SAMPLE CREATION
# =========================================================

def create_samples(
    source_name: str,
    df: pd.DataFrame,
    split: str,
    samples_per_class: int,
):

    records = []


    for label in CLASSES:

        require_motion = (
            label
            == "Wheel / Encoder Direction Mismatch"
        )


        produced = 0

        attempts = 0


        while (
            produced
            < samples_per_class
        ):

            attempts += 1


            if attempts > (
                samples_per_class
                * 15
            ):

                raise RuntimeError(
                    (
                        f"Unable to generate enough "
                        f"{label} samples from "
                        f"{source_name}."
                    )
                )


            try:

                window = choose_window(
                    df,
                    require_motion=
                        require_motion,
                )


                augmented, metadata = augment(
                    window,
                    label,
                )


            except RuntimeError:

                continue


            features = summarize_fault_window(
                augmented
            )


            record = {

                "sample_id":
                    (
                        f"{source_name}"
                        f"__{label.replace(' ', '_')}"
                        f"__{produced:03d}"
                    ),

                "source_mission":
                    source_name,

                "split":
                    split,

                "label":
                    label,

                **features,

                "fault_start_s":
                    metadata.get(
                        "fault_start_s",
                        np.nan,
                    ),

                "fault_duration_s":
                    metadata.get(
                        "fault_duration_s",
                        np.nan,
                    ),

                "localization_offset_m":
                    metadata.get(
                        "localization_offset_m",
                        np.nan,
                    ),

                "modified_wheel":
                    metadata.get(
                        "modified_wheel",
                        "",
                    ),

                "modified_samples":
                    metadata.get(
                        "modified_samples",
                        np.nan,
                    ),
            }


            records.append(
                record
            )


            produced += 1


    return records


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n=== RobotReplay Supervised Dataset Builder V2 ===\n"
    )


    frames = {}


    for source, path in SOURCES.items():

        if not path.is_file():

            raise FileNotFoundError(
                path
            )


        frame = pd.read_csv(
            path
        )


        frames[
            source
        ] = frame


        print(
            f"Loaded {source:<12} "
            f"{len(frame):>5} rows"
        )


    records = []


    for source in TRAIN_SOURCES:

        records.extend(
            create_samples(
                source,
                frames[
                    source
                ],
                "train",
                TRAIN_SAMPLES_PER_CLASS_PER_SOURCE,
            )
        )


    for source in TEST_SOURCES:

        records.extend(
            create_samples(
                source,
                frames[
                    source
                ],
                "test",
                TEST_SAMPLES_PER_CLASS,
            )
        )


    dataset = pd.DataFrame(
        records
    )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    dataset.to_csv(
        OUTPUT,
        index=False,
    )


    summary = (
        dataset.groupby(
            [
                "split",
                "source_mission",
                "label",
            ]
        )
        .size()
        .rename(
            "samples"
        )
        .reset_index()
    )


    print(
        "\nDataset composition:"
    )


    print(
        summary.to_string(
            index=False
        )
    )


    print(
        "\n----------------------------------------"
    )


    print(
        f"Classifier features: "
        f"{len(FEATURE_NAMES)}"
    )


    print(
        f"Training samples: "
        f"{int((dataset['split'] == 'train').sum())}"
    )


    print(
        f"Held-out test samples: "
        f"{int((dataset['split'] == 'test').sum())}"
    )


    print(
        "\nTrain source missions:"
    )


    for source in TRAIN_SOURCES:

        print(
            f"  • {source}"
        )


    print(
        "\nHeld-out source missions:"
    )


    for source in TEST_SOURCES:

        print(
            f"  • {source}"
        )


    print(
        f"\nSaved: {OUTPUT}"
    )


    print(
        "\nDataset build V2: PASS"
    )


if __name__ == "__main__":

    main()
