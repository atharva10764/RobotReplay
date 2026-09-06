#!/usr/bin/env python3

from __future__ import annotations

import io
import stat
import sys
import tempfile
import zipfile

from pathlib import Path


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


def expect_rejected(
    name: str,
    payload: bytes,
    expected_text: str | None = None,
) -> None:

    try:

        with tempfile.TemporaryDirectory(
            prefix="robotreplay_security_"
        ) as temp_dir:

            safe_extract_rosbag_zip(
                payload,
                Path(temp_dir) / "extract",
            )

    except UploadSecurityError as exc:

        message = str(exc)

        if (
            expected_text is not None
            and expected_text.lower()
            not in message.lower()
        ):
            raise AssertionError(
                (
                    f"{name}: rejected, but reason "
                    f"did not contain '{expected_text}'. "
                    f"Actual: {message}"
                )
            )

        print(
            f"[PASS] {name}"
        )
        print(
            f"       {message}"
        )

        return

    raise AssertionError(
        f"{name}: malicious archive was accepted"
    )


def zip_bytes(
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
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for name, content in entries:

            archive.writestr(
                name,
                content,
            )

    return buffer.getvalue()


def test_path_traversal() -> None:

    payload = zip_bytes(
        [
            (
                "../escape.db3",
                b"bad",
            ),
        ]
    )

    expect_rejected(
        "Path traversal",
        payload,
        "path-traversal",
    )


def test_absolute_path() -> None:

    payload = zip_bytes(
        [
            (
                "/tmp/escape.db3",
                b"bad",
            ),
        ]
    )

    expect_rejected(
        "Absolute path",
        payload,
        "absolute path",
    )


def test_windows_traversal() -> None:

    payload = zip_bytes(
        [
            (
                "..\\escape.db3",
                b"bad",
            ),
        ]
    )

    expect_rejected(
        "Windows-style path traversal",
        payload,
        "path-traversal",
    )


def test_unexpected_file_type() -> None:

    payload = zip_bytes(
        [
            (
                "mission/metadata.yaml",
                b"test",
            ),
            (
                "mission/payload.py",
                b"print('bad')",
            ),
        ]
    )

    expect_rejected(
        "Unexpected file type",
        payload,
        "unexpected file type",
    )


def test_symlink() -> None:

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
    ) as archive:

        info = zipfile.ZipInfo(
            "mission/link.db3"
        )

        info.create_system = 3

        info.external_attr = (
            stat.S_IFLNK
            | 0o777
        ) << 16

        archive.writestr(
            info,
            "/etc/passwd",
        )

    expect_rejected(
        "Symlink entry",
        buffer.getvalue(),
        "symbolic links",
    )


def test_empty_zip() -> None:

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
    ):
        pass

    expect_rejected(
        "Empty ZIP",
        buffer.getvalue(),
        "empty",
    )


def test_not_zip() -> None:

    expect_rejected(
        "Invalid ZIP",
        b"this is not a zip archive",
        "valid ZIP",
    )


def test_missing_metadata() -> None:

    payload = zip_bytes(
        [
            (
                "mission/data.db3",
                b"fake sqlite content",
            ),
        ]
    )

    expect_rejected(
        "Missing metadata",
        payload,
        "metadata.yaml",
    )


def test_multiple_bags() -> None:

    payload = zip_bytes(
        [
            (
                "bag_a/metadata.yaml",
                b"fake",
            ),
            (
                "bag_a/a.db3",
                b"fake",
            ),
            (
                "bag_b/metadata.yaml",
                b"fake",
            ),
            (
                "bag_b/b.db3",
                b"fake",
            ),
        ]
    )

    expect_rejected(
        "Multiple bags in one upload",
        payload,
        "Exactly one ROS 2 bag",
    )


def main() -> None:

    print(
        "\n=== RobotReplay Upload Security Regression ===\n"
    )

    tests = [
        test_path_traversal,
        test_absolute_path,
        test_windows_traversal,
        test_unexpected_file_type,
        test_symlink,
        test_empty_zip,
        test_not_zip,
        test_missing_metadata,
        test_multiple_bags,
    ]

    passed = 0

    for test in tests:

        try:

            test()
            passed += 1

        except Exception as exc:

            print(
                f"[FAIL] {test.__name__}"
            )
            print(
                f"       {exc}"
            )

            raise

    print(
        "\n----------------------------------------"
    )

    print(
        f"Security tests passed: "
        f"{passed} / {len(tests)}"
    )

    print(
        "\nUpload security regression: PASS"
    )


if __name__ == "__main__":
    main()
