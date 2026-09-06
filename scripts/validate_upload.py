#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile

from pathlib import Path


# ---------------------------------------------------------
# Allow this developer utility to run directly from source:
#
#   python3 scripts/validate_upload.py <bag.zip>
#
# even before `pip install -e .`.
# ---------------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from robotreplay.ingestion.upload import (
    UploadSecurityError,
    safe_extract_rosbag_zip,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate a RobotReplay "
            "ROS 2 bag upload."
        )
    )


    parser.add_argument(
        "zip_file",
        help=(
            "Path to a ZIP containing "
            "one ROS 2 bag."
        ),
    )


    args = parser.parse_args()


    zip_path = Path(
        args.zip_file
    ).expanduser().resolve()


    if not zip_path.is_file():

        raise SystemExit(
            f"File not found: {zip_path}"
        )


    payload = (
        zip_path.read_bytes()
    )


    try:

        with tempfile.TemporaryDirectory(
            prefix="robotreplay_upload_test_"
        ) as temp_dir:

            result = safe_extract_rosbag_zip(

                payload,

                Path(
                    temp_dir
                )
                / "extracted",
            )


            metadata = (
                result.metadata
            )


            print(
                "\n=== RobotReplay Upload Validation ==="
            )


            print(
                f"Archive:       "
                f"{zip_path.name}"
            )


            print(
                f"Upload size:   "
                f"{result.upload_size_bytes / 1024 / 1024:.2f} MiB"
            )


            print(
                f"Extracted:     "
                f"{result.extracted_size_bytes / 1024 / 1024:.2f} MiB"
            )


            print(
                f"Files:         "
                f"{result.extracted_file_count}"
            )


            print(
                f"Bag directory: "
                f"{result.bag_dir.name}"
            )


            print(
                f"Storage:       "
                f"{metadata.storage_id}"
            )


            print(
                f"Duration:      "
                f"{metadata.duration_s:.3f} s"
            )


            print(
                f"Messages:      "
                f"{metadata.message_count:,}"
            )


            print(
                f"Topics:        "
                f"{metadata.topic_count}"
            )


            print(
                "\nTopics detected:"
            )


            for topic in metadata.topics:

                print(
                    f"  • {topic}"
                )


            print(
                "\nDiagnostic capabilities:"
            )


            labels = {

                "generic_anomaly_model":
                    "Generic ML anomaly layer",

                "lidar_dropout":
                    "LiDAR dropout",

                "tf_delay":
                    "TF delay",

                "localization_jump":
                    "Localization jump",

                "wheel_mismatch":
                    "Wheel mismatch",
            }


            for (
                capability,
                available,
            ) in metadata.capabilities.items():

                marker = (
                    "YES"
                    if available
                    else "NO"
                )


                print(
                    f"  "
                    f"{labels.get(capability, capability):<28} "
                    f"{marker}"
                )


            print(
                "\nSecurity checks:"
            )

            print(
                "  ✓ Valid ZIP archive"
            )

            print(
                "  ✓ Path traversal check"
            )

            print(
                "  ✓ Symlink check"
            )

            print(
                "  ✓ File-type allowlist"
            )

            print(
                "  ✓ Archive-size limits"
            )

            print(
                "  ✓ Single ROS bag structure"
            )

            print(
                "  ✓ rosbag metadata validation"
            )


            print(
                "\nUpload validation: PASS"
            )


    except UploadSecurityError as exc:

        print(
            "\nUpload validation: REJECTED"
        )

        print(
            f"Reason: {exc}"
        )

        raise SystemExit(
            1
        )


    except Exception as exc:

        print(
            "\nUpload validation: FAILED"
        )

        print(
            f"Reason: {exc}"
        )

        raise


if __name__ == "__main__":

    main()
