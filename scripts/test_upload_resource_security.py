#!/usr/bin/env python3

from __future__ import annotations

import io
import sys
import tempfile
import zipfile

from pathlib import Path


# =========================================================
# PROJECT IMPORT BOOTSTRAP
# =========================================================
#
# This script is executed directly:
#
#   python3 scripts/test_upload_resource_security.py
#
# Therefore Python initially places only the scripts/
# directory on sys.path. Add the RobotReplay project root
# so the local robotreplay package can be imported reliably.
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


if str(
    PROJECT_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from robotreplay.ingestion.upload import (
    UploadSecurityError,
    safe_extract_rosbag_zip,
)


# =========================================================
# ZIP CREATION
# =========================================================

def make_zip(
    entries: list[
        tuple[
            str,
            bytes,
        ]
    ],
) -> bytes:

    buffer = io.BytesIO()


    with zipfile.ZipFile(
        buffer,
        "w",
        compression=
            zipfile.ZIP_DEFLATED,
    ) as archive:

        for (
            name,
            payload,
        ) in entries:

            archive.writestr(
                name,
                payload,
            )


    return buffer.getvalue()


# =========================================================
# REJECTION ASSERTION
# =========================================================

def expect_rejection(
    name: str,
    payload: bytes,
    expected_text: str,
) -> bool:

    with tempfile.TemporaryDirectory(
        prefix=
            "robotreplay_security_"
    ) as temp_dir:

        try:

            safe_extract_rosbag_zip(
                payload,
                Path(
                    temp_dir
                ),
            )


        except UploadSecurityError as exc:

            message = str(
                exc
            )


            passed = (
                expected_text.lower()
                in
                message.lower()
            )


            print(
                f"[{'PASS' if passed else 'FAIL'}] "
                f"{name}"
            )


            print(
                f"       {message}"
            )


            return passed


        except Exception as exc:

            print(
                f"[FAIL] {name}"
            )


            print(
                "       Unexpected exception: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )


            return False


        else:

            print(
                f"[FAIL] {name}"
            )


            print(
                "       Upload was unexpectedly accepted."
            )


            return False


# =========================================================
# REGRESSION SUITE
# =========================================================

def main() -> None:

    print(
        "\n=== RobotReplay Resource Security Regression ===\n"
    )


    results = []


    # -----------------------------------------------------
    # TEST 1
    # Archive file-count limit
    #
    # MAX_ARCHIVE_FILES = 32.
    #
    # metadata.yaml + 32 db3 files = 33 files total.
    # The archive must be rejected before rosbag parsing.
    # -----------------------------------------------------

    many_files = [
        (
            "bag/metadata.yaml",
            b"placeholder",
        )
    ]


    for index in range(
        32
    ):

        many_files.append(
            (
                (
                    "bag/"
                    f"data_{index:02d}.db3"
                ),
                b"",
            )
        )


    results.append(
        expect_rejection(
            "Archive file-count limit",
            make_zip(
                many_files
            ),
            "too many files",
        )
    )


    # -----------------------------------------------------
    # TEST 2
    # Suspicious compression-ratio protection
    #
    # The upload validator ignores compression-ratio checks
    # for files <=1 MiB because metadata may compress very
    # efficiently naturally.
    #
    # Create a 2 MiB highly compressible .db3 payload.
    # -----------------------------------------------------

    compression_bomb = make_zip(
        [
            (
                "bag/metadata.yaml",
                b"placeholder",
            ),
            (
                "bag/data.db3",
                b"\x00"
                *
                (
                    2
                    *
                    1024
                    *
                    1024
                ),
            ),
        ]
    )


    results.append(
        expect_rejection(
            "Suspicious compression ratio",
            compression_bomb,
            "compression ratio",
        )
    )


    # -----------------------------------------------------
    # TEST 3
    # Allow-listed data file outside selected bag directory
    #
    # The extension itself is permitted, but once the one
    # metadata.yaml identifies bag/ as the recording
    # directory, unrelated.db3 outside bag/ must be rejected.
    # -----------------------------------------------------

    outside_file = make_zip(
        [
            (
                "bag/metadata.yaml",
                b"""
rosbag2_bagfile_information:
  storage_identifier: sqlite3
  duration:
    nanoseconds: 1
  message_count: 1
  topics_with_message_count: []
  relative_file_paths:
    - data.db3
""",
            ),
            (
                "bag/data.db3",
                b"not-a-real-database",
            ),
            (
                "unrelated.db3",
                b"outside-selected-bag",
            ),
        ]
    )


    results.append(
        expect_rejection(
            "File outside selected bag directory",
            outside_file,
            "outside the ROS 2 bag directory",
        )
    )


    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    passed = sum(
        bool(
            result
        )
        for result
        in results
    )


    total = len(
        results
    )


    print(
        "\n----------------------------------------"
    )


    print(
        "Resource security tests passed: "
        f"{passed} / {total}"
    )


    if passed != total:

        print(
            "\nResource security regression: FAIL"
        )


        raise SystemExit(
            1
        )


    print(
        "\nResource security regression: PASS"
    )


if __name__ == "__main__":

    main()
