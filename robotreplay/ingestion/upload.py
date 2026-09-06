#!/usr/bin/env python3

from __future__ import annotations

import io
import shutil
import stat
import zipfile

from dataclasses import dataclass
from pathlib import (
    Path,
    PurePosixPath,
)

from robotreplay.ingestion.validation import (
    BagMetadata,
    BagValidationError,
    inspect_bag_directory,
)


# ---------------------------------------------------------
# Upload limits
# ---------------------------------------------------------

MAX_UPLOAD_BYTES = (
    250 * 1024 * 1024
)

MAX_EXTRACTED_BYTES = (
    1024 * 1024 * 1024
)

MAX_SINGLE_FILE_BYTES = (
    750 * 1024 * 1024
)

MAX_ARCHIVE_FILES = 32

MAX_COMPRESSION_RATIO = 200.0


# Only files expected inside a ROS 2 recording.
ALLOWED_EXACT_FILENAMES = {
    "metadata.yaml",
}

ALLOWED_DATA_SUFFIXES = {
    ".db3",
    ".mcap",
}


class UploadSecurityError(
    ValueError
):
    pass


@dataclass
class SafeUploadResult:

    bag_dir: Path

    metadata: BagMetadata

    upload_size_bytes: int

    extracted_size_bytes: int

    extracted_file_count: int


    def to_dict(
        self,
    ) -> dict:

        return {

            "bag_dir":
                str(
                    self.bag_dir
                ),

            "upload_size_bytes":
                self.upload_size_bytes,

            "extracted_size_bytes":
                self.extracted_size_bytes,

            "extracted_file_count":
                self.extracted_file_count,

            "metadata":
                self.metadata.to_dict(),
        }


def _normalized_member_name(
    raw_name: str,
) -> PurePosixPath:

    if "\x00" in raw_name:

        raise UploadSecurityError(
            "Archive contains a null byte in a filename."
        )


    normalized = raw_name.replace(
        "\\",
        "/",
    )


    path = PurePosixPath(
        normalized
    )


    if path.is_absolute():

        raise UploadSecurityError(
            (
                "Archive contains an "
                "absolute path."
            )
        )


    if ".." in path.parts:

        raise UploadSecurityError(
            (
                "Archive contains a "
                "path-traversal entry."
            )
        )


    if not path.parts:

        raise UploadSecurityError(
            "Archive contains an invalid path."
        )


    return path


def _is_symlink(
    info: zipfile.ZipInfo,
) -> bool:

    mode = (
        info.external_attr
        >> 16
    ) & 0xFFFF


    return stat.S_ISLNK(
        mode
    )


def _validate_file_type(
    member: PurePosixPath,
) -> None:

    filename = (
        member.name
    )


    if (
        filename
        in ALLOWED_EXACT_FILENAMES
    ):

        return


    suffix = (
        Path(
            filename
        ).suffix.lower()
    )


    if (
        suffix
        not in ALLOWED_DATA_SUFFIXES
    ):

        raise UploadSecurityError(
            (
                "Archive contains an unexpected "
                f"file type: {filename}"
            )
        )


def inspect_zip_security(
    archive: zipfile.ZipFile,
) -> tuple[
    int,
    int,
]:

    infos = archive.infolist()


    file_infos = [
        info
        for info
        in infos
        if not info.is_dir()
    ]


    if not file_infos:

        raise UploadSecurityError(
            "Uploaded ZIP archive is empty."
        )


    if (
        len(
            file_infos
        )
        > MAX_ARCHIVE_FILES
    ):

        raise UploadSecurityError(
            (
                "Archive contains too many files. "
                f"Maximum allowed: {MAX_ARCHIVE_FILES}."
            )
        )


    extracted_bytes = 0


    for info in file_infos:

        member = _normalized_member_name(
            info.filename
        )


        if _is_symlink(
            info
        ):

            raise UploadSecurityError(
                (
                    "Symbolic links are not "
                    "allowed in uploaded archives."
                )
            )


        _validate_file_type(
            member
        )


        if (
            info.file_size
            > MAX_SINGLE_FILE_BYTES
        ):

            raise UploadSecurityError(
                (
                    f"{member.name} exceeds the "
                    "maximum allowed file size."
                )
            )


        extracted_bytes += (
            info.file_size
        )


        if (
            extracted_bytes
            > MAX_EXTRACTED_BYTES
        ):

            raise UploadSecurityError(
                (
                    "Archive expands beyond the "
                    "maximum allowed extraction size."
                )
            )


        compressed_size = max(
            int(
                info.compress_size
            ),
            1,
        )


        compression_ratio = (
            info.file_size
            / compressed_size
        )


        # Ignore very small files because YAML and
        # metadata can naturally compress extremely well.
        if (
            info.file_size
            > 1024 * 1024
            and
            compression_ratio
            > MAX_COMPRESSION_RATIO
        ):

            raise UploadSecurityError(
                (
                    "Archive contains a suspiciously "
                    "high compression ratio."
                )
            )


    return (
        len(
            file_infos
        ),
        extracted_bytes,
    )


def safe_extract_rosbag_zip(
    upload_bytes: bytes,
    destination: str | Path,
) -> SafeUploadResult:

    upload_size = len(
        upload_bytes
    )


    if upload_size <= 0:

        raise UploadSecurityError(
            "Uploaded file is empty."
        )


    if (
        upload_size
        > MAX_UPLOAD_BYTES
    ):

        raise UploadSecurityError(
            (
                "Uploaded archive exceeds the "
                "250 MB upload limit."
            )
        )


    buffer = io.BytesIO(
        upload_bytes
    )


    if not zipfile.is_zipfile(
        buffer
    ):

        raise UploadSecurityError(
            (
                "Uploaded file is not a "
                "valid ZIP archive."
            )
        )


    buffer.seek(
        0
    )


    destination = Path(
        destination
    ).expanduser().resolve()


    destination.mkdir(
        parents=True,
        exist_ok=True,
    )


    with zipfile.ZipFile(
        buffer,
        "r",
    ) as archive:

        (
            file_count,
            extracted_bytes,
        ) = inspect_zip_security(
            archive
        )


        for info in archive.infolist():

            member = _normalized_member_name(
                info.filename
            )


            target = (
                destination
                / Path(
                    *member.parts
                )
            ).resolve()


            try:

                target.relative_to(
                    destination
                )

            except ValueError as exc:

                raise UploadSecurityError(
                    (
                        "Archive attempted to "
                        "escape the extraction directory."
                    )
                ) from exc


            if info.is_dir():

                target.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                continue


            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            with archive.open(
                info,
                "r",
            ) as source:

                with open(
                    target,
                    "wb",
                ) as destination_file:

                    shutil.copyfileobj(
                        source,
                        destination_file,
                        length=1024 * 1024,
                    )


    metadata_files = list(
        destination.rglob(
            "metadata.yaml"
        )
    )


    if not metadata_files:

        raise UploadSecurityError(
            (
                "No metadata.yaml was found. "
                "Upload a ZIP containing one "
                "ROS 2 bag directory."
            )
        )


    if (
        len(
            metadata_files
        )
        != 1
    ):

        raise UploadSecurityError(
            (
                "Exactly one ROS 2 bag must be "
                "included in each uploaded archive."
            )
        )


    bag_dir = (
        metadata_files[
            0
        ].parent.resolve()
    )


    # Reject unrelated files outside the selected
    # bag directory even if their extension is valid.
    for extracted_file in (
        destination.rglob(
            "*"
        )
    ):

        if not extracted_file.is_file():

            continue


        try:

            extracted_file.resolve().relative_to(
                bag_dir
            )

        except ValueError as exc:

            raise UploadSecurityError(
                (
                    "Archive contains files outside "
                    "the ROS 2 bag directory."
                )
            ) from exc


    try:

        metadata = inspect_bag_directory(
            bag_dir
        )

    except BagValidationError:

        raise

    except Exception as exc:

        raise UploadSecurityError(
            (
                "Extracted recording could not "
                f"be validated: {exc}"
            )
        ) from exc


    return SafeUploadResult(

        bag_dir=
            bag_dir,

        metadata=
            metadata,

        upload_size_bytes=
            upload_size,

        extracted_size_bytes=
            extracted_bytes,

        extracted_file_count=
            file_count,
    )
