#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


import joblib
import numpy as np
import pandas as pd


from robotreplay.ml.fault_features import (
    summarize_fault_window,
)


# =========================================================
# PATHS / CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "fault_classifier.joblib"
)


WINDOW_SECONDS = 20.0

WINDOW_STRIDE_SECONDS = 2.0


HEALTHY_CLASS = (
    "Healthy"
)


FAULT_DECISION_THRESHOLD = 0.60


# ---------------------------------------------------------
# Cross-platform applicability
# ---------------------------------------------------------
#
# The classifier was trained using the complete canonical
# RobotReplay telemetry representation.
#
# Missing feature families must NOT silently become zeros
# and then be interpreted as genuine robot measurements.
#
# A mission therefore needs a minimum fraction of the
# classifier's required feature representation before
# predict_proba() is allowed.
# ---------------------------------------------------------

MIN_CLASSIFIER_FEATURE_COVERAGE = 0.60


class FaultClassifierError(
    RuntimeError
):
    pass


# =========================================================
# FEATURE REQUIREMENTS
# =========================================================
#
# Each classifier summary feature is mapped to the telemetry
# columns required to calculate it meaningfully.
#
# A tuple contains alternative valid requirement sets.
#
# Example:
#
#   (
#       {"command_active"},
#       {"cmd_vx", "cmd_wz"},
#   )
#
# means either representation is sufficient.
# =========================================================

FEATURE_REQUIREMENTS = {

    # -----------------------------------------------------
    # LiDAR
    # -----------------------------------------------------

    "scan_age_max":
        (
            {
                "scan_age",
            },
        ),

    "scan_age_p95":
        (
            {
                "scan_age",
            },
        ),

    "scan_gap_max":
        (
            {
                "scan_gap",
            },
        ),

    "scan_valid_fraction_min":
        (
            {
                "scan_valid_fraction",
            },
        ),


    # -----------------------------------------------------
    # TF
    # -----------------------------------------------------

    "tf_map_odom_age_max":
        (
            {
                "tf_map_odom_age",
            },
        ),

    "tf_map_odom_age_p95":
        (
            {
                "tf_map_odom_age",
            },
        ),

    "tf_odom_base_age_max":
        (
            {
                "tf_odom_base_age",
            },
        ),


    # -----------------------------------------------------
    # Localization
    # -----------------------------------------------------

    "amcl_position_step_max":
        (
            {
                "amcl_position_step",
            },
        ),

    "tf_map_odom_position_step_max":
        (
            {
                "tf_map_odom_position_step",
            },
        ),

    "amcl_tf_jump_product":
        (
            {
                "amcl_position_step",
                "tf_map_odom_position_step",
            },
        ),


    # -----------------------------------------------------
    # Wheel consistency
    # -----------------------------------------------------

    "wheel_opposite_forward_fraction":
        (
            {
                "cmd_vx",
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "wheel_opposite_motion_fraction":
        (
            {
                "cmd_vx",
                "cmd_wz",
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "right_negative_forward_fraction":
        (
            {
                "cmd_vx",
                "right_wheel_vel",
            },
        ),

    "left_negative_forward_fraction":
        (
            {
                "cmd_vx",
                "left_wheel_vel",
            },
        ),

    "right_negative_straight_fraction":
        (
            {
                "cmd_vx",
                "cmd_wz",
                "right_wheel_vel",
            },
        ),

    "left_negative_straight_fraction":
        (
            {
                "cmd_vx",
                "cmd_wz",
                "left_wheel_vel",
            },
        ),

    "left_cmd_sign_disagreement_fraction":
        (
            {
                "cmd_vx",
                "cmd_wz",
                "left_wheel_vel",
            },
        ),

    "right_cmd_sign_disagreement_fraction":
        (
            {
                "cmd_vx",
                "cmd_wz",
                "right_wheel_vel",
            },
        ),

    "wheel_velocity_difference_mean":
        (
            {
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "wheel_velocity_difference_p95":
        (
            {
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "wheel_velocity_difference_max":
        (
            {
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "wheel_velocity_product_min":
        (
            {
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "wheel_velocity_product_p05":
        (
            {
                "left_wheel_vel",
                "right_wheel_vel",
            },
        ),

    "left_wheel_abs_mean":
        (
            {
                "left_wheel_vel",
            },
        ),

    "right_wheel_abs_mean":
        (
            {
                "right_wheel_vel",
            },
        ),

    "left_cmd_vx_correlation":
        (
            {
                "cmd_vx",
                "left_wheel_vel",
            },
        ),

    "right_cmd_vx_correlation":
        (
            {
                "cmd_vx",
                "right_wheel_vel",
            },
        ),


    # -----------------------------------------------------
    # Motion consistency
    # -----------------------------------------------------

    "linear_motion_error_p95":
        (
            {
                "linear_motion_error",
            },
        ),

    "angular_motion_error_p95":
        (
            {
                "angular_motion_error",
            },
        ),

    "odom_imu_yaw_error_p95":
        (
            {
                "odom_imu_yaw_error",
            },
        ),


    # -----------------------------------------------------
    # Mission activity
    # -----------------------------------------------------

    "command_active_fraction":
        (
            {
                "command_active",
            },

            {
                "cmd_vx",
                "cmd_wz",
            },
        ),

    "straight_forward_fraction":
        (
            {
                "cmd_vx",
                "cmd_wz",
            },
        ),

    "cmd_vx_abs_mean":
        (
            {
                "cmd_vx",
            },
        ),

    "cmd_wz_abs_mean":
        (
            {
                "cmd_wz",
            },
        ),
}


# =========================================================
# MODEL LOADING
# =========================================================

def load_fault_classifier(
    model_path: str | Path =
        DEFAULT_MODEL_PATH,
) -> dict:

    model_path = Path(
        model_path
    ).expanduser().resolve()


    if not model_path.is_file():

        raise FaultClassifierError(
            (
                "Fault classifier model "
                f"not found: {model_path}"
            )
        )


    package = joblib.load(
        model_path
    )


    required = {
        "model",
        "features",
        "classes",
    }


    missing = (
        required
        -
        set(
            package.keys()
        )
    )


    if missing:

        raise FaultClassifierError(
            (
                "Fault classifier package "
                "is missing: "
                +
                ", ".join(
                    sorted(
                        missing
                    )
                )
            )
        )


    return package


# =========================================================
# APPLICABILITY
# =========================================================

def _column_has_data(
    mission: pd.DataFrame,
    column: str,
) -> bool:

    if column not in mission.columns:

        return False


    values = mission[
        column
    ]


    return bool(
        values.notna().any()
    )


def _requirements_satisfied(
    mission: pd.DataFrame,
    alternatives: tuple[
        set[str],
        ...,
    ],
) -> bool:

    for requirement_set in alternatives:

        if all(
            _column_has_data(
                mission,
                column,
            )

            for column
            in requirement_set
        ):

            return True


    return False


def classifier_feature_coverage(
    mission: pd.DataFrame,
    feature_names: list[str],
) -> dict:

    available = []

    missing = []


    for feature in feature_names:

        requirements = (
            FEATURE_REQUIREMENTS.get(
                feature
            )
        )


        if requirements is None:

            # Conservative fallback:
            # an unknown classifier feature is only
            # considered available if a real telemetry
            # column with that name exists.

            is_available = (
                _column_has_data(
                    mission,
                    feature,
                )
            )


        else:

            is_available = (
                _requirements_satisfied(
                    mission,
                    requirements,
                )
            )


        if is_available:

            available.append(
                feature
            )

        else:

            missing.append(
                feature
            )


    total = len(
        feature_names
    )


    coverage = (
        len(
            available
        )
        /
        total

        if total
        else 0.0
    )


    return {

        "coverage":
            float(
                coverage
            ),

        "available_count":
            len(
                available
            ),

        "required_count":
            total,

        "available_features":
            available,

        "missing_features":
            missing,
    }


def validate_classifier_applicability(
    mission: pd.DataFrame,
    package: dict,
) -> dict:

    features = list(
        package[
            "features"
        ]
    )


    coverage = (
        classifier_feature_coverage(
            mission,
            features,
        )
    )


    if (
        coverage[
            "coverage"
        ]
        <
        MIN_CLASSIFIER_FEATURE_COVERAGE
    ):

        raise FaultClassifierError(
            (
                "Fault classifier is not applicable: "
                f"only "
                f"{100 * coverage['coverage']:.0f}% "
                "of its required telemetry feature "
                "representation is available "
                f"({coverage['available_count']}/"
                f"{coverage['required_count']}; "
                f"minimum "
                f"{100 * MIN_CLASSIFIER_FEATURE_COVERAGE:.0f}%)."
            )
        )


    return coverage


# =========================================================
# FEATURE ROW
# =========================================================

def prepare_feature_row(
    window: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:

    summary = (
        summarize_fault_window(
            window
        )
    )


    row = {

        feature:
            float(
                summary.get(
                    feature,
                    0.0,
                )
            )

        for feature
        in feature_names
    }


    return pd.DataFrame(
        [
            row
        ],
        columns=
            feature_names,
    )


# =========================================================
# SINGLE WINDOW CLASSIFICATION
# =========================================================

def classify_window(
    window: pd.DataFrame,
    package: dict,
) -> dict:

    model = package[
        "model"
    ]


    features = list(
        package[
            "features"
        ]
    )


    X = prepare_feature_row(
        window,
        features,
    )


    probabilities_raw = (
        model.predict_proba(
            X
        )[
            0
        ]
    )


    classes = [

        str(
            class_name
        )

        for class_name
        in model.classes_
    ]


    probabilities = {

        class_name:
            float(
                probability
            )

        for (
            class_name,
            probability,
        )
        in zip(
            classes,
            probabilities_raw,
        )
    }


    predicted_class = max(
        probabilities,
        key=
            probabilities.get,
    )


    predicted_probability = float(
        probabilities[
            predicted_class
        ]
    )


    ranked = sorted(
        probabilities.items(),
        key=lambda item:
            item[
                1
            ],
        reverse=True,
    )


    return {

        "predicted_class":
            predicted_class,

        "predicted_probability":
            predicted_probability,

        "probabilities":
            probabilities,

        "ranked_probabilities":
            [

                {

                    "class":
                        name,

                    "probability":
                        float(
                            probability
                        ),
                }

                for (
                    name,
                    probability,
                )
                in ranked
            ],

        "features":
            {

                feature:
                    float(
                        X.iloc[
                            0
                        ][
                            feature
                        ]
                    )

                for feature
                in features
            },
    }


# =========================================================
# ROLLING WINDOWS
# =========================================================

def iter_windows(
    mission: pd.DataFrame,
    *,
    window_seconds: float =
        WINDOW_SECONDS,
    stride_seconds: float =
        WINDOW_STRIDE_SECONDS,
):

    if "time" not in mission.columns:

        raise FaultClassifierError(
            (
                "Mission telemetry does "
                "not contain 'time'."
            )
        )


    if mission.empty:

        return


    mission_start = float(
        mission[
            "time"
        ].min()
    )


    mission_end = float(
        mission[
            "time"
        ].max()
    )


    duration = (
        mission_end
        -
        mission_start
    )


    # -----------------------------------------------------
    # Short mission
    # -----------------------------------------------------

    if duration <= window_seconds:

        window = (
            mission.copy()
        )


        window[
            "time"
        ] = (
            window[
                "time"
            ]
            -
            mission_start
        )


        yield (
            mission_start,
            mission_end,
            window,
        )

        return


    # -----------------------------------------------------
    # Sliding windows
    # -----------------------------------------------------

    start = (
        mission_start
    )


    while (
        start
        +
        window_seconds
        <=
        mission_end
        +
        1e-6
    ):

        end = (
            start
            +
            window_seconds
        )


        window = mission[
            (
                mission[
                    "time"
                ]
                >= start
            )
            &
            (
                mission[
                    "time"
                ]
                <= end
            )
        ].copy()


        if len(
            window
        ) >= 20:

            window[
                "time"
            ] = (
                window[
                    "time"
                ]
                -
                start
            )


            yield (
                start,
                end,
                window,
            )


        start += (
            stride_seconds
        )


# =========================================================
# MISSION CLASSIFICATION
# =========================================================

def classify_mission(
    mission: pd.DataFrame,
    *,
    model_path: str | Path =
        DEFAULT_MODEL_PATH,
    window_seconds: float =
        WINDOW_SECONDS,
    stride_seconds: float =
        WINDOW_STRIDE_SECONDS,
) -> dict:

    package = load_fault_classifier(
        model_path
    )


    # -----------------------------------------------------
    # Applicability gate
    # -----------------------------------------------------

    coverage = (
        validate_classifier_applicability(
            mission,
            package,
        )
    )


    model_metrics = dict(
        package.get(
            "metrics",
            {},
        )
        or {}
    )


    window_results = []


    # -----------------------------------------------------
    # Score every rolling telemetry window
    # -----------------------------------------------------

    for (
        start,
        end,
        window,
    ) in iter_windows(

        mission,

        window_seconds=
            window_seconds,

        stride_seconds=
            stride_seconds,

    ):

        result = classify_window(
            window,
            package,
        )


        result[
            "window_start_s"
        ] = round(
            float(
                start
            ),
            3,
        )


        result[
            "window_end_s"
        ] = round(
            float(
                end
            ),
            3,
        )


        window_results.append(
            result
        )


    if not window_results:

        raise FaultClassifierError(
            (
                "No valid classifier windows "
                "could be generated."
            )
        )


    classes = [

        str(
            class_name
        )

        for class_name
        in package[
            "classes"
        ]
    ]


    fault_classes = [

        class_name

        for class_name
        in classes

        if class_name
        != HEALTHY_CLASS
    ]


    # -----------------------------------------------------
    # Mission-wide class support
    #
    # IMPORTANT:
    #
    # These values can come from different windows.
    # They are therefore support scores, NOT a probability
    # distribution.
    # -----------------------------------------------------

    mission_peak_support = {}


    peak_fault_windows = {}


    for class_name in classes:

        best = max(

            window_results,

            key=lambda item:
                float(
                    item[
                        "probabilities"
                    ].get(
                        class_name,
                        0.0,
                    )
                ),
        )


        support = float(
            best[
                "probabilities"
            ].get(
                class_name,
                0.0,
            )
        )


        mission_peak_support[
            class_name
        ] = (
            support
        )


        peak_fault_windows[
            class_name
        ] = {

            "window_start_s":
                best[
                    "window_start_s"
                ],

            "window_end_s":
                best[
                    "window_end_s"
                ],

            "support":
                support,
        }


    # -----------------------------------------------------
    # Fault decision
    # -----------------------------------------------------

    strongest_fault = max(
        fault_classes,
        key=lambda class_name:
            mission_peak_support.get(
                class_name,
                0.0,
            ),
    )


    strongest_fault_support = float(
        mission_peak_support.get(
            strongest_fault,
            0.0,
        )
    )


    if (
        strongest_fault_support
        >=
        FAULT_DECISION_THRESHOLD
    ):

        prediction = (
            strongest_fault
        )


        selected = max(

            window_results,

            key=lambda item:
                float(
                    item[
                        "probabilities"
                    ].get(
                        strongest_fault,
                        0.0,
                    )
                ),
        )


    else:

        prediction = (
            HEALTHY_CLASS
        )


        # Pick an actual representative healthy window
        # whose Healthy probability is closest to the
        # median Healthy probability.
        #
        # This keeps displayed probabilities tied to ONE
        # real predict_proba() call.

        healthy_values = np.asarray(
            [

                float(
                    item[
                        "probabilities"
                    ].get(
                        HEALTHY_CLASS,
                        0.0,
                    )
                )

                for item
                in window_results
            ],
            dtype=float,
        )


        median_healthy = float(
            np.median(
                healthy_values
            )
        )


        selected = min(

            window_results,

            key=lambda item:
                abs(
                    float(
                        item[
                            "probabilities"
                        ].get(
                            HEALTHY_CLASS,
                            0.0,
                        )
                    )
                    -
                    median_healthy
                ),
        )


    # -----------------------------------------------------
    # Selected-window probability distribution
    #
    # All values below come from ONE model decision window
    # and therefore sum to 1.
    # -----------------------------------------------------

    class_probabilities = {

        str(
            name
        ):
            float(
                probability
            )

        for (
            name,
            probability,
        )
        in selected[
            "probabilities"
        ].items()
    }


    probability_total = sum(
        class_probabilities.values()
    )


    if probability_total > 0:

        class_probabilities = {

            name:
                float(
                    value
                    /
                    probability_total
                )

            for (
                name,
                value,
            )
            in class_probabilities.items()
        }


    prediction_probability = float(
        class_probabilities.get(
            prediction,
            0.0,
        )
    )


    ranked_class_probabilities = [

        {

            "class":
                name,

            "probability":
                float(
                    probability
                ),
        }

        for (
            name,
            probability,
        )
        in sorted(

            class_probabilities.items(),

            key=lambda item:
                item[
                    1
                ],

            reverse=True,
        )
    ]


    ranked_mission_peak_support = [

        {

            "class":
                name,

            "support":
                float(
                    support
                ),
        }

        for (
            name,
            support,
        )
        in sorted(

            mission_peak_support.items(),

            key=lambda item:
                item[
                    1
                ],

            reverse=True,
        )
    ]


    selected_window = {

        "window_start_s":
            selected[
                "window_start_s"
            ],

        "window_end_s":
            selected[
                "window_end_s"
            ],

        "model_window_prediction":
            selected[
                "predicted_class"
            ],

        "model_window_probability":
            float(
                selected[
                    "predicted_probability"
                ]
            ),
    }


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    return {

        "model":
            package.get(
                "model_name",
                "Random Forest",
            ),

        "model_version":
            package.get(
                "model_version",
                "fault-classifier-v2",
            ),

        "prediction":
            prediction,

        "prediction_probability":
            prediction_probability,

        "decision_threshold":
            FAULT_DECISION_THRESHOLD,

        "selected_window":
            selected_window,

        "class_probabilities":
            class_probabilities,

        "ranked_class_probabilities":
            ranked_class_probabilities,

        "mission_peak_support":
            mission_peak_support,

        "ranked_mission_peak_support":
            ranked_mission_peak_support,

        "peak_fault_windows":
            peak_fault_windows,

        "windows_evaluated":
            len(
                window_results
            ),

        "window_seconds":
            float(
                window_seconds
            ),

        "stride_seconds":
            float(
                stride_seconds
            ),

        "feature_coverage":
            float(
                coverage[
                    "coverage"
                ]
            ),

        "available_feature_count":
            int(
                coverage[
                    "available_count"
                ]
            ),

        "required_feature_count":
            int(
                coverage[
                    "required_count"
                ]
            ),

        "missing_features":
            coverage[
                "missing_features"
            ],

        "validation_metrics":
            model_metrics,

        "method_note":
            (
                "The supervised fault classifier is used "
                "only when sufficient compatible telemetry "
                "is available. Mission-wide peak support "
                "values are calculated across rolling "
                "windows and are not probabilities. "
                "Displayed class probabilities come from "
                "one selected model decision window and "
                "sum to 1. Model probabilities are not "
                "calibrated real-world confidence."
            ),
    }
