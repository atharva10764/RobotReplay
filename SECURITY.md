# Security

RobotReplay processes uploaded ROS 2 recording archives and therefore treats
input handling as part of the system's security boundary.

## Current Upload Protections

The current ingestion layer includes:

- path-traversal rejection,
- absolute-path rejection,
- Windows-style traversal rejection,
- symlink rejection,
- file-type allow-listing,
- archive file-count limits,
- extracted-size limits,
- single-file limits,
- suspicious compression-ratio rejection,
- single-bag structural validation,
- ROS bag metadata validation,
- rejection of files outside the selected bag directory.

Current configured limits include:

- compressed upload: 250 MiB,
- extracted archive: 1 GiB,
- single extracted file: 750 MiB,
- archive file count: 32,
- compression ratio guard: 200.

## Data Handling

In the current Streamlit workflow, uploaded recordings are extracted into a
temporary workspace, analyzed locally, and removed after the analysis result
has been transferred into the active application session.

RobotReplay does not currently provide a multi-user hosted storage layer.

## Regression Evidence

The baseline upload regression contains:

- 9 / 9 input-security cases passing,
- 3 / 3 resource-security cases passing,
- positive SQLite3 upload validation,
- positive MCAP upload validation.

These tests verify the implemented controls. They are not a penetration test,
formal security audit, or security certification.

## Reporting a Security Issue

Please do not publish sensitive exploit details in a public issue.

Contact the repository owner through the GitHub profile and provide:

- affected component,
- reproduction conditions,
- expected impact,
- minimal proof of concept where appropriate.

Security fixes may be handled privately before public disclosure.
