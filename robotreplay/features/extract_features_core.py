#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math

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


from robotreplay.ingestion.validation import (
    inspect_bag_directory,
    resolve_semantic_topics,
)


# =========================================================
# HELPERS
# =========================================================

def quat_to_yaw(
    q,
):

    siny_cosp = (
        2.0
        * (
            q.w
            * q.z
            +
            q.x
            * q.y
        )
    )


    cosy_cosp = (
        1.0
        -
        2.0
        * (
            q.y
            * q.y
            +
            q.z
            * q.z
        )
    )


    return math.atan2(
        siny_cosp,
        cosy_cosp,
    )


def wrapped_angle_diff(
    series,
):

    d = (
        series.diff()
    )


    return d.apply(
        lambda x:
            abs(
                math.atan2(
                    math.sin(
                        x
                    ),
                    math.cos(
                        x
                    ),
                )
            )
            if pd.notna(
                x
            )
            else np.nan
    )


def merge_topic(
    grid,
    df,
):

    if df.empty:

        return grid


    return pd.merge_asof(
        grid.sort_values(
            "time"
        ),

        df.sort_values(
            "time"
        ),

        on="time",

        direction="backward",
    )


def command_twist(
    msg,
):

    # geometry_msgs/TwistStamped
    if hasattr(
        msg,
        "twist",
    ):

        return msg.twist


    # geometry_msgs/Twist
    return msg


# =========================================================
# FEATURE EXTRACTION
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


    # -----------------------------------------------------
    # Validate bag + detect rosbag storage backend
    # -----------------------------------------------------

    metadata = (
        inspect_bag_directory(
            bag_path
        )
    )


    reader = (
        rosbag2_py.SequentialReader()
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


    # -----------------------------------------------------
    # Reader topic metadata
    #
    # Resolve from the actual rosbag reader rather than
    # assuming TurtleBot topic names.
    # -----------------------------------------------------

    topic_types = {

        x.name:
            x.type

        for x
        in reader.get_all_topics_and_types()
    }


    semantic_topics = (
        resolve_semantic_topics(
            topic_types
        )
    )


    roles_used = {
        "scan",
        "cmd_vel",
        "odom",
        "imu",
        "joint_states",
        "amcl_pose",
        "tf",
    }


    active_topics = {

        role:
            semantic_topics.get(
                role
            )

        for role
        in roles_used

        if semantic_topics.get(
            role
        )
        is not None
    }


    role_by_topic = {

        topic:
            role

        for (
            role,
            topic,
        )
        in active_topics.items()
    }


    message_classes = {

        topic:
            get_message(
                topic_types[
                    topic
                ]
            )

        for topic
        in role_by_topic
    }


    # -----------------------------------------------------
    # Raw-event buffers
    # -----------------------------------------------------

    scan_rows = []

    cmd_rows = []

    odom_rows = []

    imu_rows = []

    joint_rows = []

    amcl_rows = []


    tf_map_odom_rows = []

    tf_odom_base_rows = []


    first_ns = None

    last_ns = None


    joint_names = set()


    # -----------------------------------------------------
    # Read recording
    # -----------------------------------------------------

    while reader.has_next():

        (
            topic,
            raw,
            timestamp_ns,
        ) = reader.read_next()


        # Mission time is based on the full recording,
        # not only topics consumed by RobotReplay.
        if first_ns is None:

            first_ns = (
                timestamp_ns
            )


        last_ns = (
            timestamp_ns
        )


        if topic not in message_classes:

            continue


        t = (
            timestamp_ns
            - first_ns
        ) / 1e9


        msg = (
            deserialize_message(
                raw,
                message_classes[
                    topic
                ],
            )
        )


        role = (
            role_by_topic[
                topic
            ]
        )


        # =================================================
        # LiDAR
        # =================================================

        if role == "scan":

            ranges = np.asarray(
                msg.ranges,
                dtype=float,
            )


            valid = ranges[
                np.isfinite(
                    ranges
                )
                &
                (
                    ranges
                    >= msg.range_min
                )
                &
                (
                    ranges
                    <= msg.range_max
                )
            ]


            scan_rows.append({

                "time":
                    t,

                "scan_msg_time":
                    t,

                "scan_min":
                    (
                        float(
                            np.min(
                                valid
                            )
                        )
                        if len(
                            valid
                        )
                        else np.nan
                    ),

                "scan_mean":
                    (
                        float(
                            np.mean(
                                valid
                            )
                        )
                        if len(
                            valid
                        )
                        else np.nan
                    ),

                "scan_valid_fraction":
                    (
                        float(
                            len(
                                valid
                            )
                            /
                            len(
                                ranges
                            )
                        )
                        if len(
                            ranges
                        )
                        else np.nan
                    ),
            })


        # =================================================
        # Command
        # =================================================

        elif role == "cmd_vel":

            twist = (
                command_twist(
                    msg
                )
            )


            cmd_rows.append({

                "time":
                    t,

                "cmd_msg_time":
                    t,

                "cmd_vx":
                    twist.linear.x,

                "cmd_vy":
                    twist.linear.y,

                "cmd_wz":
                    twist.angular.z,
            })


        # =================================================
        # Odometry
        # =================================================

        elif role == "odom":

            odom_rows.append({

                "time":
                    t,

                "odom_msg_time":
                    t,

                "odom_x":
                    msg.pose.pose.position.x,

                "odom_y":
                    msg.pose.pose.position.y,

                "odom_yaw":
                    quat_to_yaw(
                        msg.pose.pose.orientation
                    ),

                "odom_vx":
                    msg.twist.twist.linear.x,

                "odom_vy":
                    msg.twist.twist.linear.y,

                "odom_wz":
                    msg.twist.twist.angular.z,
            })


        # =================================================
        # IMU
        # =================================================

        elif role == "imu":

            imu_rows.append({

                "time":
                    t,

                "imu_msg_time":
                    t,

                "imu_wz":
                    msg.angular_velocity.z,

                "imu_ax":
                    msg.linear_acceleration.x,

                "imu_ay":
                    msg.linear_acceleration.y,
            })


        # =================================================
        # Wheel joints
        # =================================================

        elif role == "joint_states":

            names = list(
                msg.name
            )


            velocities = list(
                msg.velocity
            )


            joint_names.update(
                names
            )


            values = {

                name:
                    velocities[
                        i
                    ]

                for (
                    i,
                    name,
                )
                in enumerate(
                    names
                )

                if i
                < len(
                    velocities
                )
            }


            left = (
                np.nan
            )


            right = (
                np.nan
            )


            for (
                name,
                velocity,
            ) in values.items():

                lname = (
                    name.lower()
                )


                if (
                    "wheel"
                    in lname
                    and
                    "left"
                    in lname
                ):

                    left = (
                        velocity
                    )


                if (
                    "wheel"
                    in lname
                    and
                    "right"
                    in lname
                ):

                    right = (
                        velocity
                    )


            joint_rows.append({

                "time":
                    t,

                "joint_msg_time":
                    t,

                "left_wheel_vel":
                    left,

                "right_wheel_vel":
                    right,
            })


        # =================================================
        # Localization pose
        # =================================================

        elif role == "amcl_pose":

            cov = (
                msg.pose.covariance
            )


            amcl_rows.append({

                "time":
                    t,

                "amcl_msg_time":
                    t,

                "amcl_x":
                    msg.pose.pose.position.x,

                "amcl_y":
                    msg.pose.pose.position.y,

                "amcl_yaw":
                    quat_to_yaw(
                        msg.pose.pose.orientation
                    ),

                "amcl_cov_x":
                    cov[
                        0
                    ],

                "amcl_cov_y":
                    cov[
                        7
                    ],

                "amcl_cov_yaw":
                    cov[
                        35
                    ],
            })


        # =================================================
        # Dynamic TF
        # =================================================

        elif role == "tf":

            for tr in msg.transforms:

                parent = (
                    tr.header.frame_id
                    .lstrip(
                        "/"
                    )
                )


                child = (
                    tr.child_frame_id
                    .lstrip(
                        "/"
                    )
                )


                if (
                    parent
                    == "map"
                    and
                    child
                    == "odom"
                ):

                    tf_map_odom_rows.append({

                        "time":
                            t,

                        "tf_map_odom_msg_time":
                            t,

                        "tf_map_odom_x":
                            tr.transform.translation.x,

                        "tf_map_odom_y":
                            tr.transform.translation.y,

                        "tf_map_odom_yaw":
                            quat_to_yaw(
                                tr.transform.rotation
                            ),
                    })


                elif (
                    parent
                    == "odom"
                    and
                    child
                    in {
                        "base_footprint",
                        "base_link",
                    }
                ):

                    tf_odom_base_rows.append({

                        "time":
                            t,

                        "tf_odom_base_msg_time":
                            t,

                        "tf_odom_base_x":
                            tr.transform.translation.x,

                        "tf_odom_base_y":
                            tr.transform.translation.y,

                        "tf_odom_base_yaw":
                            quat_to_yaw(
                                tr.transform.rotation
                            ),
                    })


    if first_ns is None:

        raise RuntimeError(
            "Bag contains no messages."
        )


    duration = (
        last_ns
        - first_ns
    ) / 1e9


    # -----------------------------------------------------
    # DataFrames
    # -----------------------------------------------------

    scan_df = (
        pd.DataFrame(
            scan_rows
        )
    )


    cmd_df = (
        pd.DataFrame(
            cmd_rows
        )
    )


    odom_df = (
        pd.DataFrame(
            odom_rows
        )
    )


    imu_df = (
        pd.DataFrame(
            imu_rows
        )
    )


    joint_df = (
        pd.DataFrame(
            joint_rows
        )
    )


    amcl_df = (
        pd.DataFrame(
            amcl_rows
        )
    )


    tf_map_odom_df = (
        pd.DataFrame(
            tf_map_odom_rows
        )
    )


    tf_odom_base_df = (
        pd.DataFrame(
            tf_odom_base_rows
        )
    )


    # =====================================================
    # Raw-event features
    # =====================================================

    if not scan_df.empty:

        scan_df[
            "scan_gap"
        ] = (
            scan_df[
                "time"
            ].diff()
        )


        scan_df[
            "scan_rate_hz"
        ] = np.where(

            scan_df[
                "scan_gap"
            ] > 0,

            1.0
            /
            scan_df[
                "scan_gap"
            ],

            np.nan,
        )


    if not amcl_df.empty:

        dx = (
            amcl_df[
                "amcl_x"
            ].diff()
        )


        dy = (
            amcl_df[
                "amcl_y"
            ].diff()
        )


        amcl_df[
            "amcl_position_step"
        ] = np.sqrt(
            dx**2
            +
            dy**2
        )


        amcl_df[
            "amcl_yaw_step"
        ] = (
            wrapped_angle_diff(
                amcl_df[
                    "amcl_yaw"
                ]
            )
        )


    if not tf_map_odom_df.empty:

        dx = (
            tf_map_odom_df[
                "tf_map_odom_x"
            ].diff()
        )


        dy = (
            tf_map_odom_df[
                "tf_map_odom_y"
            ].diff()
        )


        tf_map_odom_df[
            "tf_map_odom_position_step"
        ] = (
            np.sqrt(
                dx**2
                +
                dy**2
            )
        )


        tf_map_odom_df[
            "tf_map_odom_yaw_step"
        ] = (
            wrapped_angle_diff(
                tf_map_odom_df[
                    "tf_map_odom_yaw"
                ]
            )
        )


    # =====================================================
    # Synchronization
    # =====================================================

    grid = pd.DataFrame({

        "time":
            np.arange(
                0.0,
                duration
                +
                sample_period,
                sample_period,
            )
    })


    for df in [

        scan_df,

        cmd_df,

        odom_df,

        imu_df,

        joint_df,

        amcl_df,

        tf_map_odom_df,

        tf_odom_base_df,

    ]:

        grid = (
            merge_topic(
                grid,
                df,
            )
        )


    # =====================================================
    # Message ages
    # =====================================================

    age_columns = {

        "scan_msg_time":
            "scan_age",

        "cmd_msg_time":
            "cmd_age",

        "odom_msg_time":
            "odom_age",

        "imu_msg_time":
            "imu_age",

        "joint_msg_time":
            "joint_age",

        "amcl_msg_time":
            "amcl_age",

        "tf_map_odom_msg_time":
            "tf_map_odom_age",

        "tf_odom_base_msg_time":
            "tf_odom_base_age",
    }


    for (
        source_col,
        age_col,
    ) in age_columns.items():

        if source_col in grid.columns:

            grid[
                age_col
            ] = (
                grid[
                    "time"
                ]
                -
                grid[
                    source_col
                ]
            ).clip(
                lower=0
            )


    # No recent cmd_vel message means no active command.
    if "cmd_age" in grid.columns:

        stale_cmd = (
            grid[
                "cmd_age"
            ]
            > 0.5
        )


        for col in [
            "cmd_vx",
            "cmd_vy",
            "cmd_wz",
        ]:

            if col in grid.columns:

                grid.loc[
                    stale_cmd,
                    col,
                ] = 0.0


    # =====================================================
    # Derived signals
    # =====================================================

    if {
        "cmd_vx",
        "odom_vx",
    }.issubset(
        grid.columns
    ):

        grid[
            "linear_motion_error"
        ] = (

            grid[
                "cmd_vx"
            ]
            -
            grid[
                "odom_vx"
            ]

        ).abs()


    if {
        "cmd_wz",
        "odom_wz",
    }.issubset(
        grid.columns
    ):

        grid[
            "angular_motion_error"
        ] = (

            grid[
                "cmd_wz"
            ]
            -
            grid[
                "odom_wz"
            ]

        ).abs()


    if {
        "left_wheel_vel",
        "right_wheel_vel",
    }.issubset(
        grid.columns
    ):

        grid[
            "wheel_velocity_difference"
        ] = (

            grid[
                "left_wheel_vel"
            ]
            -
            grid[
                "right_wheel_vel"
            ]

        ).abs()


        grid[
            "wheel_velocity_product"
        ] = (

            grid[
                "left_wheel_vel"
            ]
            *
            grid[
                "right_wheel_vel"
            ]
        )


    if {
        "odom_wz",
        "imu_wz",
    }.issubset(
        grid.columns
    ):

        grid[
            "odom_imu_yaw_error"
        ] = (

            grid[
                "odom_wz"
            ]
            -
            grid[
                "imu_wz"
            ]

        ).abs()


    if {
        "cmd_vx",
        "cmd_wz",
    }.issubset(
        grid.columns
    ):

        grid[
            "command_active"
        ] = (

            (
                grid[
                    "cmd_vx"
                ].abs()
                > 0.02
            )

            |

            (
                grid[
                    "cmd_wz"
                ].abs()
                > 0.02
            )

        ).astype(
            int
        )


    # =====================================================
    # Metadata columns
    # =====================================================

    grid[
        "label"
    ] = (
        label
    )


    grid[
        "source_bag"
    ] = (
        bag_path.name
    )


    # Remove internal timestamp helper columns.
    helper_columns = [

        c

        for c
        in grid.columns

        if c.endswith(
            "_msg_time"
        )
    ]


    grid.drop(

        columns=
            helper_columns,

        inplace=True,

        errors="ignore",
    )


    # =====================================================
    # SAVE
    # =====================================================

    output_csv = Path(
        output_csv
    ).expanduser().resolve()


    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    grid.to_csv(
        output_csv,
        index=False,
    )


    # =====================================================
    # REPORT
    # =====================================================

    print(
        "\n=== RobotReplay Feature Extraction V2 ==="
    )


    print(
        f"Bag:           {bag_path}"
    )


    print(
        f"Storage:       {metadata.storage_id}"
    )


    print(
        f"Duration:      {duration:.3f} s"
    )


    print(
        f"Feature rows:  {len(grid)}"
    )


    print(
        f"Feature cols:  {len(grid.columns)}"
    )


    print(
        "\nResolved telemetry:"
    )


    display_roles = [
        "scan",
        "cmd_vel",
        "odom",
        "imu",
        "joint_states",
        "amcl_pose",
        "tf",
    ]


    for role in display_roles:

        resolved = (
            semantic_topics.get(
                role
            )
        )


        print(
            f"  {role:<16} "
            f"{resolved or 'not available'}"
        )


    if joint_names:

        print(
            "\nJoint names:"
        )


        for name in sorted(
            joint_names
        ):

            print(
                f"  {name}"
            )


    print(
        "\nHealth / age features:"
    )


    age_features = [
        "scan_age",
        "cmd_age",
        "odom_age",
        "imu_age",
        "joint_age",
        "amcl_age",
        "tf_map_odom_age",
        "tf_odom_base_age",
    ]


    found_age = False


    for column in age_features:

        if column not in grid.columns:

            continue


        values = (
            pd.to_numeric(
                grid[
                    column
                ],
                errors="coerce",
            )
            .dropna()
        )


        if values.empty:

            continue


        found_age = True


        print(
            f"  {column:<25} "
            f"max={values.max():.3f}s"
        )


    if not found_age:

        print(
            "  No age features available."
        )


    print(
        f"\nSaved: {output_csv}"
    )


    print(
        "Feature extraction V2: PASS"
    )


    return grid


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract RobotReplay telemetry features "
            "from a ROS 2 sqlite3 or MCAP recording."
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
