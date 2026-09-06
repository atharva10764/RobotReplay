#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# SUPERVISED FAULT CLASSIFIER FEATURES
#
# No ground-truth marker is included.
# No rule-based final diagnosis is included.
# =========================================================

FEATURE_NAMES = [

    # -----------------------------------------------------
    # LiDAR
    # -----------------------------------------------------

    "scan_age_max",
    "scan_age_p95",
    "scan_gap_max",
    "scan_valid_fraction_min",

    # -----------------------------------------------------
    # TF
    # -----------------------------------------------------

    "tf_map_odom_age_max",
    "tf_map_odom_age_p95",
    "tf_odom_base_age_max",

    # -----------------------------------------------------
    # Localization
    # -----------------------------------------------------

    "amcl_position_step_max",
    "tf_map_odom_position_step_max",
    "amcl_tf_jump_product",

    # -----------------------------------------------------
    # Wheel consistency
    # -----------------------------------------------------

    "wheel_opposite_forward_fraction",
    "wheel_opposite_motion_fraction",

    "right_negative_forward_fraction",
    "left_negative_forward_fraction",

    "right_negative_straight_fraction",
    "left_negative_straight_fraction",

    "left_cmd_sign_disagreement_fraction",
    "right_cmd_sign_disagreement_fraction",

    "wheel_velocity_difference_mean",
    "wheel_velocity_difference_p95",
    "wheel_velocity_difference_max",

    "wheel_velocity_product_min",
    "wheel_velocity_product_p05",

    "left_wheel_abs_mean",
    "right_wheel_abs_mean",

    "left_cmd_vx_correlation",
    "right_cmd_vx_correlation",

    # -----------------------------------------------------
    # Motion consistency
    # -----------------------------------------------------

    "linear_motion_error_p95",
    "angular_motion_error_p95",
    "odom_imu_yaw_error_p95",

    # -----------------------------------------------------
    # Mission activity
    # -----------------------------------------------------

    "command_active_fraction",
    "straight_forward_fraction",

    "cmd_vx_abs_mean",
    "cmd_wz_abs_mean",
]


# =========================================================
# HELPERS
# =========================================================

def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in df.columns:

        return pd.Series(
            dtype=float
        )


    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).dropna()


def series_or_zero(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:

    if column not in df.columns:

        return pd.Series(
            0.0,
            index=df.index,
            dtype=float,
        )


    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(
        0.0
    )


def safe_max(
    df: pd.DataFrame,
    column: str,
) -> float:

    values = numeric_series(
        df,
        column,
    )


    if values.empty:
        return 0.0


    return float(
        values.max()
    )


def safe_min(
    df: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> float:

    values = numeric_series(
        df,
        column,
    )


    if values.empty:
        return default


    return float(
        values.min()
    )


def safe_percentile(
    df: pd.DataFrame,
    column: str,
    percentile: float,
) -> float:

    values = numeric_series(
        df,
        column,
    )


    if values.empty:
        return 0.0


    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def safe_abs_mean(
    df: pd.DataFrame,
    column: str,
) -> float:

    values = numeric_series(
        df,
        column,
    )


    if values.empty:
        return 0.0


    return float(
        values.abs().mean()
    )


def fraction_true(
    mask: pd.Series,
) -> float:

    if len(mask) == 0:
        return 0.0


    return float(
        mask.fillna(
            False
        ).mean()
    )


def safe_correlation(
    a: pd.Series,
    b: pd.Series,
    mask: pd.Series | None = None,
) -> float:

    if mask is not None:

        a = a[
            mask
        ]

        b = b[
            mask
        ]


    valid = (
        a.notna()
        &
        b.notna()
    )


    a = a[
        valid
    ]

    b = b[
        valid
    ]


    if len(a) < 4:
        return 0.0


    if (
        float(
            a.std()
        )
        < 1e-9
        or
        float(
            b.std()
        )
        < 1e-9
    ):

        return 0.0


    correlation = (
        a.corr(
            b
        )
    )


    if pd.isna(
        correlation
    ):

        return 0.0


    return float(
        correlation
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def summarize_fault_window(
    df: pd.DataFrame,
) -> dict[str, float]:

    # -----------------------------------------------------
    # Commands
    # -----------------------------------------------------

    cmd_vx = series_or_zero(
        df,
        "cmd_vx",
    )


    cmd_wz = series_or_zero(
        df,
        "cmd_wz",
    )


    if "command_active" in df.columns:

        command_active = (
            series_or_zero(
                df,
                "command_active",
            )
            > 0.5
        )

    else:

        command_active = (
            (cmd_vx.abs() > 0.02)
            |
            (cmd_wz.abs() > 0.02)
        )


    forward = (
        cmd_vx
        > 0.05
    )


    straight_forward = (
        (cmd_vx > 0.05)
        &
        (cmd_wz.abs() < 0.35)
    )


    active_motion = (
        (cmd_vx.abs() > 0.05)
        |
        (cmd_wz.abs() > 0.10)
    )


    # -----------------------------------------------------
    # Wheels
    # -----------------------------------------------------

    left = series_or_zero(
        df,
        "left_wheel_vel",
    )


    right = series_or_zero(
        df,
        "right_wheel_vel",
    )


    wheel_product = (
        left
        * right
    )


    wheel_difference = (
        left
        - right
    ).abs()


    moving_wheels = (
        (left.abs() > 0.3)
        &
        (right.abs() > 0.3)
    )


    opposite_forward = (
        forward
        &
        moving_wheels
        &
        (wheel_product < 0)
    )


    opposite_motion = (
        active_motion
        &
        moving_wheels
        &
        (wheel_product < 0)
    )


    right_negative_forward = (
        forward
        &
        (right < -0.3)
    )


    left_negative_forward = (
        forward
        &
        (left < -0.3)
    )


    right_negative_straight = (
        straight_forward
        &
        (right < -0.3)
    )


    left_negative_straight = (
        straight_forward
        &
        (left < -0.3)
    )


    # During mostly straight translation, both wheel
    # signs normally follow the sign of cmd_vx.

    strong_translation = (
        (cmd_vx.abs() > 0.05)
        &
        (cmd_wz.abs() < 0.35)
    )


    left_cmd_sign_disagreement = (
        strong_translation
        &
        (left.abs() > 0.3)
        &
        (
            np.sign(
                left
            )
            !=
            np.sign(
                cmd_vx
            )
        )
    )


    right_cmd_sign_disagreement = (
        strong_translation
        &
        (right.abs() > 0.3)
        &
        (
            np.sign(
                right
            )
            !=
            np.sign(
                cmd_vx
            )
        )
    )


    # -----------------------------------------------------
    # Localization evidence
    # -----------------------------------------------------

    amcl_jump = safe_max(
        df,
        "amcl_position_step",
    )


    tf_jump = safe_max(
        df,
        "tf_map_odom_position_step",
    )


    # -----------------------------------------------------
    # Derived temporary frame
    # -----------------------------------------------------

    derived = df.copy()


    derived[
        "_wheel_difference"
    ] = wheel_difference


    derived[
        "_wheel_product"
    ] = wheel_product


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    result = {

        # LiDAR
        "scan_age_max":
            safe_max(
                df,
                "scan_age",
            ),

        "scan_age_p95":
            safe_percentile(
                df,
                "scan_age",
                95,
            ),

        "scan_gap_max":
            safe_max(
                df,
                "scan_gap",
            ),

        "scan_valid_fraction_min":
            safe_min(
                df,
                "scan_valid_fraction",
                default=1.0,
            ),


        # TF
        "tf_map_odom_age_max":
            safe_max(
                df,
                "tf_map_odom_age",
            ),

        "tf_map_odom_age_p95":
            safe_percentile(
                df,
                "tf_map_odom_age",
                95,
            ),

        "tf_odom_base_age_max":
            safe_max(
                df,
                "tf_odom_base_age",
            ),


        # Localization
        "amcl_position_step_max":
            amcl_jump,

        "tf_map_odom_position_step_max":
            tf_jump,

        "amcl_tf_jump_product":
            (
                amcl_jump
                * tf_jump
            ),


        # Wheels
        "wheel_opposite_forward_fraction":
            fraction_true(
                opposite_forward
            ),

        "wheel_opposite_motion_fraction":
            fraction_true(
                opposite_motion
            ),

        "right_negative_forward_fraction":
            fraction_true(
                right_negative_forward
            ),

        "left_negative_forward_fraction":
            fraction_true(
                left_negative_forward
            ),

        "right_negative_straight_fraction":
            fraction_true(
                right_negative_straight
            ),

        "left_negative_straight_fraction":
            fraction_true(
                left_negative_straight
            ),

        "left_cmd_sign_disagreement_fraction":
            fraction_true(
                left_cmd_sign_disagreement
            ),

        "right_cmd_sign_disagreement_fraction":
            fraction_true(
                right_cmd_sign_disagreement
            ),

        "wheel_velocity_difference_mean":
            float(
                wheel_difference.mean()
            )
            if len(
                wheel_difference
            )
            else 0.0,

        "wheel_velocity_difference_p95":
            safe_percentile(
                derived,
                "_wheel_difference",
                95,
            ),

        "wheel_velocity_difference_max":
            float(
                wheel_difference.max()
            )
            if len(
                wheel_difference
            )
            else 0.0,

        "wheel_velocity_product_min":
            float(
                wheel_product.min()
            )
            if len(
                wheel_product
            )
            else 0.0,

        "wheel_velocity_product_p05":
            safe_percentile(
                derived,
                "_wheel_product",
                5,
            ),

        "left_wheel_abs_mean":
            float(
                left.abs().mean()
            )
            if len(
                left
            )
            else 0.0,

        "right_wheel_abs_mean":
            float(
                right.abs().mean()
            )
            if len(
                right
            )
            else 0.0,

        "left_cmd_vx_correlation":
            safe_correlation(
                cmd_vx,
                left,
                command_active,
            ),

        "right_cmd_vx_correlation":
            safe_correlation(
                cmd_vx,
                right,
                command_active,
            ),


        # Motion consistency
        "linear_motion_error_p95":
            safe_percentile(
                df,
                "linear_motion_error",
                95,
            ),

        "angular_motion_error_p95":
            safe_percentile(
                df,
                "angular_motion_error",
                95,
            ),

        "odom_imu_yaw_error_p95":
            safe_percentile(
                df,
                "odom_imu_yaw_error",
                95,
            ),


        # Mission activity
        "command_active_fraction":
            fraction_true(
                command_active
            ),

        "straight_forward_fraction":
            fraction_true(
                straight_forward
            ),

        "cmd_vx_abs_mean":
            safe_abs_mean(
                df,
                "cmd_vx",
            ),

        "cmd_wz_abs_mean":
            safe_abs_mean(
                df,
                "cmd_wz",
            ),
    }


    return {

        feature:
            float(
                result[
                    feature
                ]
            )

        for feature
        in FEATURE_NAMES
    }


def select_incident_window(
    df: pd.DataFrame,
    center_time_s: float,
    *,
    before_s: float = 7.0,
    after_s: float = 13.0,
) -> pd.DataFrame:

    if "time" not in df.columns:
        return df.copy()


    start = (
        center_time_s
        - before_s
    )


    end = (
        center_time_s
        + after_s
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
        return df.copy()


    return window
