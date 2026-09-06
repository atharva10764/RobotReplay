# Baseline Release Evidence

This directory stores reproducibility evidence for the current RobotReplay
baseline.

It contains:

- controlled mission regression outputs,
- upload-security regression outputs,
- positive SQLite3 / MCAP upload validation,
- public external-dataset development evidence,
- generated runtime/package manifest,
- SHA-256 hashes for selected release-critical files.

Run the verification from the repository root:

```bash
bash scripts/verify_release.sh
```

The generated manifest summarizes the exact environment and baseline outcomes.

Evaluation scope and limitations are documented in:

```text
docs/EVALUATION.md
```
