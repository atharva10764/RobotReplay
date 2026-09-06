from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)

from pathlib import Path

import yaml


from robotreplay.config import (
    SUPPORTED_STORAGE_IDS,
)


# =========================================================
# ERRORS
# =========================================================

class BagValidationError(
    ValueError
):
    pass


# =========================================================
# METADATA
# =========================================================

@dataclass
class BagMetadata:

    path: str

    storage_id: str

    duration_s: float

    message_count: int

    topic_count: int

    topics: list[str]

    topic_types: dict[str, str]

    semantic_topics: dict[str, str | None]

    files: list[str]

    size_bytes: int

    capabilities: dict[str, bool]


    def to_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )


# =========================================================
# SEMANTIC TOPIC DISCOVERY
# =========================================================

def _choose_exact(
    topic_types: dict[str, str],
    candidates: list[str],
    allowed_types: set[str] | None = None,
) -> str | None:

    for name in candidates:

        if name not in topic_types:
            continue


        if (
            allowed_types is None
            or
            topic_types[
                name
            ] in allowed_types
        ):

            return name


    return None


def _choose_by_type(
    topic_types: dict[str, str],
    allowed_types: set[str],
    preferred_terms: tuple[str, ...] = (),
    rejected_terms: tuple[str, ...] = (),
) -> str | None:

    candidates = []


    for (
        name,
        type_name,
    ) in topic_types.items():

        if type_name not in allowed_types:
            continue


        lower = name.lower()


        if any(
            term in lower
            for term in rejected_terms
        ):

            continue


        score = 0


        for index, term in enumerate(
            preferred_terms
        ):

            if term in lower:

                score += (
                    len(
                        preferred_terms
                    )
                    - index
                )


        candidates.append(
            (
                score,
                len(name),
                name,
            )
        )


    if not candidates:

        return None


    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        )
    )


    return candidates[
        0
    ][
        2
    ]


def resolve_semantic_topics(
    topic_types: dict[str, str],
) -> dict[str, str | None]:

    laser_types = {
        "sensor_msgs/msg/LaserScan",
    }


    odom_types = {
        "nav_msgs/msg/Odometry",
    }


    command_types = {
        "geometry_msgs/msg/Twist",
        "geometry_msgs/msg/TwistStamped",
    }


    imu_types = {
        "sensor_msgs/msg/Imu",
    }


    joint_types = {
        "sensor_msgs/msg/JointState",
    }


    localization_types = {
        "geometry_msgs/msg/PoseWithCovarianceStamped",
    }


    tf_types = {
        "tf2_msgs/msg/TFMessage",
    }


    map_types = {
        "nav_msgs/msg/OccupancyGrid",
    }


    # -----------------------------------------------------
    # 2D LiDAR
    # -----------------------------------------------------

    scan = _choose_exact(
        topic_types,
        [
            "/scan",
            "/scan_raw",
        ],
        laser_types,
    )


    if scan is None:

        scan = _choose_by_type(
            topic_types,
            laser_types,
            preferred_terms=(
                "scan",
                "lidar",
                "laser",
            ),
        )


    # -----------------------------------------------------
    # Odometry
    # -----------------------------------------------------

    odom = _choose_exact(
        topic_types,
        [
            "/odom",
        ],
        odom_types,
    )


    if odom is None:

        odom = _choose_by_type(
            topic_types,
            odom_types,
            preferred_terms=(
                "odom",
                "odometry",
            ),
        )


    # -----------------------------------------------------
    # Base velocity command
    # -----------------------------------------------------

    cmd_vel = _choose_exact(
        topic_types,
        [
            "/cmd_vel",
        ],
        command_types,
    )


    if cmd_vel is None:

        cmd_vel = _choose_by_type(
            topic_types,
            command_types,
            preferred_terms=(
                "cmd_vel",
                "command",
                "cmd",
                "velocity",
            ),
        )


    # -----------------------------------------------------
    # IMU
    # -----------------------------------------------------

    imu = _choose_exact(
        topic_types,
        [
            "/imu",
        ],
        imu_types,
    )


    if imu is None:

        imu = _choose_by_type(
            topic_types,
            imu_types,
            preferred_terms=(
                "imu",
            ),
        )


    # -----------------------------------------------------
    # Joint states
    # -----------------------------------------------------

    joint_states = _choose_exact(
        topic_types,
        [
            "/joint_states",
        ],
        joint_types,
    )


    if joint_states is None:

        joint_states = _choose_by_type(
            topic_types,
            joint_types,
            preferred_terms=(
                "joint_states",
                "joint",
            ),
        )


    # -----------------------------------------------------
    # Localization pose
    #
    # Deliberately conservative.
    #
    # A generic PoseWithCovarianceStamped topic should NOT
    # automatically be treated as AMCL localization.
    # -----------------------------------------------------

    amcl_pose = _choose_exact(
        topic_types,
        [
            "/amcl_pose",
        ],
        localization_types,
    )


    if amcl_pose is None:

        localization_candidates = {
            name:
                type_name

            for (
                name,
                type_name,
            ) in topic_types.items()

            if (
                type_name
                in localization_types
                and
                (
                    "amcl"
                    in name.lower()
                    or
                    "localization"
                    in name.lower()
                )
            )
        }


        amcl_pose = _choose_by_type(
            localization_candidates,
            localization_types,
            preferred_terms=(
                "amcl",
                "localization",
            ),
        )


    # -----------------------------------------------------
    # Dynamic TF
    # -----------------------------------------------------

    tf = _choose_exact(
        topic_types,
        [
            "/tf",
        ],
        tf_types,
    )


    if tf is None:

        tf = _choose_by_type(
            topic_types,
            tf_types,
            preferred_terms=(
                "tf",
            ),
            rejected_terms=(
                "static",
            ),
        )


    # -----------------------------------------------------
    # Static TF
    # -----------------------------------------------------

    tf_static = _choose_exact(
        topic_types,
        [
            "/tf_static",
        ],
        tf_types,
    )


    # -----------------------------------------------------
    # Map
    # -----------------------------------------------------

    map_topic = _choose_exact(
        topic_types,
        [
            "/map",
        ],
        map_types,
    )


    if map_topic is None:

        map_candidates = {
            name:
                type_name

            for (
                name,
                type_name,
            ) in topic_types.items()

            if (
                type_name
                in map_types
                and
                "costmap"
                not in name.lower()
            )
        }


        map_topic = _choose_by_type(
            map_candidates,
            map_types,
            preferred_terms=(
                "map",
            ),
        )


    # -----------------------------------------------------
    # Local costmap
    # -----------------------------------------------------

    costmap_candidates = {
        name:
            type_name

        for (
            name,
            type_name,
        ) in topic_types.items()

        if (
            type_name
            in map_types
            and
            "costmap"
            in name.lower()
        )
    }


    costmap = _choose_by_type(
        costmap_candidates,
        map_types,
        preferred_terms=(
            "local_costmap",
            "costmap",
        ),
    )


    return {

        "scan":
            scan,

        "cmd_vel":
            cmd_vel,

        "odom":
            odom,

        "imu":
            imu,

        "joint_states":
            joint_states,

        "amcl_pose":
            amcl_pose,

        "tf":
            tf,

        "tf_static":
            tf_static,

        "map":
            map_topic,

        "costmap":
            costmap,
    }


# =========================================================
# CAPABILITY DISCOVERY
# =========================================================

def determine_capabilities(
    topics: set[str],
    topic_types: dict[str, str] | None = None,
) -> dict[str, bool]:

    if topic_types is None:

        topic_types = {
            topic:
                ""
            for topic in topics
        }


    semantic = resolve_semantic_topics(
        topic_types
    )


    # Backward compatibility for metadata that might not
    # expose topic types correctly.
    if semantic[
        "scan"
    ] is None and "/scan" in topics:

        semantic[
            "scan"
        ] = "/scan"


    if semantic[
        "cmd_vel"
    ] is None and "/cmd_vel" in topics:

        semantic[
            "cmd_vel"
        ] = "/cmd_vel"


    if semantic[
        "odom"
    ] is None and "/odom" in topics:

        semantic[
            "odom"
        ] = "/odom"


    if semantic[
        "imu"
    ] is None and "/imu" in topics:

        semantic[
            "imu"
        ] = "/imu"


    if (
        semantic[
            "joint_states"
        ] is None
        and
        "/joint_states"
        in topics
    ):

        semantic[
            "joint_states"
        ] = "/joint_states"


    if (
        semantic[
            "amcl_pose"
        ] is None
        and
        "/amcl_pose"
        in topics
    ):

        semantic[
            "amcl_pose"
        ] = "/amcl_pose"


    if semantic[
        "tf"
    ] is None and "/tf" in topics:

        semantic[
            "tf"
        ] = "/tf"


    return {

        "generic_anomaly_model":
            bool(
                semantic[
                    "cmd_vel"
                ]
                or
                semantic[
                    "odom"
                ]
            ),

        "lidar_dropout":
            bool(
                semantic[
                    "scan"
                ]
            ),

        "tf_delay":
            bool(
                semantic[
                    "tf"
                ]
            ),

        "localization_jump":
            bool(
                semantic[
                    "amcl_pose"
                ]
                and
                semantic[
                    "tf"
                ]
            ),

        "wheel_mismatch":
            bool(
                semantic[
                    "joint_states"
                ]
                and
                semantic[
                    "cmd_vel"
                ]
            ),
    }


# =========================================================
# BAG INSPECTION
# =========================================================

def inspect_bag_directory(
    path: str | Path,
) -> BagMetadata:

    bag_dir = Path(
        path
    ).expanduser().resolve()


    if not bag_dir.is_dir():

        raise BagValidationError(
            (
                "ROS 2 recording path "
                "is not a directory."
            )
        )


    metadata_path = (
        bag_dir
        / "metadata.yaml"
    )


    if not metadata_path.is_file():

        raise BagValidationError(
            (
                "metadata.yaml was not found "
                "in the ROS 2 bag directory."
            )
        )


    try:

        payload = yaml.safe_load(
            metadata_path.read_text()
        ) or {}

    except Exception as exc:

        raise BagValidationError(
            (
                "metadata.yaml could not "
                f"be parsed: {exc}"
            )
        ) from exc


    info = payload.get(
        "rosbag2_bagfile_information"
    )


    if not isinstance(
        info,
        dict,
    ):

        raise BagValidationError(
            (
                "metadata.yaml does not contain "
                "rosbag2 bag information."
            )
        )


    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    storage_id = str(
        info.get(
            "storage_identifier"
        )
        or ""
    ).strip()


    if not storage_id:

        raise BagValidationError(
            (
                "ROS 2 bag storage "
                "identifier is missing."
            )
        )


    if (
        storage_id
        not in SUPPORTED_STORAGE_IDS
    ):

        raise BagValidationError(
            (
                f"Unsupported storage backend "
                f"'{storage_id}'. "
                f"Supported: "
                f"{', '.join(sorted(SUPPORTED_STORAGE_IDS))}."
            )
        )


    # -----------------------------------------------------
    # Duration / messages
    # -----------------------------------------------------

    duration_data = (
        info.get(
            "duration"
        )
        or {}
    )


    if isinstance(
        duration_data,
        dict,
    ):

        duration_ns = int(
            duration_data.get(
                "nanoseconds",
                0,
            )
            or 0
        )

    else:

        duration_ns = int(
            duration_data
            or 0
        )


    duration_s = (
        duration_ns
        / 1_000_000_000.0
    )


    message_count = int(
        info.get(
            "message_count",
            0,
        )
        or 0
    )


    if message_count <= 0:

        raise BagValidationError(
            (
                "The ROS 2 bag contains "
                "no recorded messages."
            )
        )


    # -----------------------------------------------------
    # Topics
    # -----------------------------------------------------

    topic_entries = (
        info.get(
            "topics_with_message_count"
        )
        or []
    )


    topics = []

    topic_types = {}


    for entry in topic_entries:

        topic_metadata = (
            (
                entry
                or {}
            ).get(
                "topic_metadata"
            )
            or {}
        )


        topic_name = (
            topic_metadata.get(
                "name"
            )
        )


        type_name = (
            topic_metadata.get(
                "type"
            )
        )


        if topic_name:

            topic_name = str(
                topic_name
            )


            topics.append(
                topic_name
            )


            if type_name:

                topic_types[
                    topic_name
                ] = str(
                    type_name
                )


    topics = sorted(
        set(
            topics
        )
    )


    semantic_topics = (
        resolve_semantic_topics(
            topic_types
        )
    )


    capabilities = (
        determine_capabilities(
            set(
                topics
            ),
            topic_types,
        )
    )


    # -----------------------------------------------------
    # Data files
    # -----------------------------------------------------

    relative_paths = list(
        info.get(
            "relative_file_paths"
        )
        or []
    )


    if storage_id == "sqlite3":

        expected_suffix = (
            ".db3"
        )

    elif storage_id == "mcap":

        expected_suffix = (
            ".mcap"
        )

    else:

        expected_suffix = ""


    if not relative_paths:

        relative_paths = [

            str(
                file_path.relative_to(
                    bag_dir
                )
            )

            for file_path
            in sorted(
                bag_dir.iterdir()
            )

            if (
                file_path.is_file()
                and
                (
                    not expected_suffix
                    or
                    file_path.suffix.lower()
                    == expected_suffix
                )
            )
        ]


    if not relative_paths:

        raise BagValidationError(
            (
                "No rosbag data file was found "
                f"for storage backend '{storage_id}'."
            )
        )


    size_bytes = 0

    resolved_files = []


    for relative in relative_paths:

        relative = str(
            relative
        )


        candidate = (
            bag_dir
            / relative
        ).resolve()


        try:

            candidate.relative_to(
                bag_dir
            )

        except ValueError as exc:

            raise BagValidationError(
                (
                    "Bag metadata references "
                    "a file outside the bag directory."
                )
            ) from exc


        if not candidate.is_file():

            raise BagValidationError(
                (
                    "Bag data file is missing: "
                    f"{relative}"
                )
            )


        if (
            expected_suffix
            and
            candidate.suffix.lower()
            != expected_suffix
        ):

            raise BagValidationError(
                (
                    f"Bag storage backend "
                    f"'{storage_id}' references "
                    f"an unexpected data file: "
                    f"{relative}"
                )
            )


        size_bytes += (
            candidate.stat().st_size
        )


        resolved_files.append(
            relative
        )


    return BagMetadata(

        path=str(
            bag_dir
        ),

        storage_id=
            storage_id,

        duration_s=
            duration_s,

        message_count=
            message_count,

        topic_count=
            len(
                topics
            ),

        topics=
            topics,

        topic_types=
            topic_types,

        semantic_topics=
            semantic_topics,

        files=
            resolved_files,

        size_bytes=
            size_bytes,

        capabilities=
            capabilities,
    )
