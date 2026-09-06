#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from robotreplay.config import (
    DEFAULT_HEALTHY_BASELINES,
    DEFAULT_MODEL_PATH,
)

from robotreplay.services.analyzer import (
    analyze_bag,
)


def print_report(
    report: dict,
) -> None:

    print(
        "\n========================================"
    )

    print(
        "        ROBOTREPLAY ANALYSIS"
    )

    print(
        "========================================"
    )


    print(
        f"\nMission: "
        f"{report.get('mission')}"
    )


    bag = report.get(
        "bag",
        {},
    )


    performance = report.get(
        "performance",
        {},
    )


    print(
        (
            f"Bag: "
            f"{bag.get('duration_s', 0):.1f} s"
            f" | "
            f"{bag.get('message_count', 0):,} messages"
            f" | "
            f"{bag.get('topic_count', 0)} topics"
            f" | "
            f"{bag.get('storage_id', 'unknown')}"
        )
    )


    print(
        (
            f"Processing time: "
            f"{performance.get('processing_time_s', 0):.3f} s"
        )
    )


    if (
        performance.get(
            "realtime_factor"
        )
        is not None
    ):

        print(
            (
                f"Processing speed: "
                f"{performance['realtime_factor']:.2f}x realtime"
            )
        )


    print(
        "\nINCIDENT REPORT"
    )

    print(
        "----------------------------------------"
    )


    print(
        (
            f"Mission status: "
            f"{report['mission_status']}"
        )
    )


    print(
        (
            f"Most likely cause: "
            f"{report['root_cause']}"
        )
    )


    if (
        report.get(
            "first_abnormal_time_s"
        )
        is not None
    ):

        print(
            (
                f"First diagnostic event: "
                f"{report['first_abnormal_time_s']:.3f} s"
            )
        )


    print(
        (
            f"Evidence strength: "
            f"{report.get('evidence_strength', 0):.3f}"
        )
    )


    ml = report.get(
        "generic_ml",
        {},
    )


    if ml.get(
        "available"
    ):

        print(
            (
                f"ML anomalous-window rate: "
                f"{100 * ml.get('anomaly_window_rate', 0):.2f}%"
            )
        )

    else:

        print(
            (
                "ML layer: unavailable "
                f"({ml.get('reason', 'unknown reason')})"
            )
        )


    print(
        "\nRanked hypotheses:"
    )


    for (
        index,
        hypothesis,
    ) in enumerate(
        report.get(
            "ranked_hypotheses",
            [],
        ),
        start=1,
    ):

        if hypothesis.get(
            "available",
            True,
        ):

            suffix = ""

        else:

            suffix = (
                " [unavailable]"
            )


        print(
            (
                f"  {index}. "
                f"{hypothesis['name']:<38} "
                f"{hypothesis.get('score', 0):.3f}"
                f"{suffix}"
            )
        )


    if report.get(
        "supporting_evidence"
    ):

        print(
            "\nSupporting evidence:"
        )

        for evidence in report[
            "supporting_evidence"
        ]:

            print(
                f"  ✓ {evidence}"
            )


    if report.get(
        "failure_chain"
    ):

        print(
            "\nFailure chain:"
        )

        for step in report[
            "failure_chain"
        ]:

            print(
                f"  → {step}"
            )


    if report.get(
        "recommended_checks"
    ):

        print(
            "\nRecommended checks:"
        )

        for check in report[
            "recommended_checks"
        ]:

            print(
                f"  • {check}"
            )


    artifacts = report.get(
        "artifacts",
        {},
    )


    if artifacts.get(
        "incident_json"
    ):

        print(
            (
                "\nIncident report saved: "
                f"{artifacts['incident_json']}"
            )
        )


    print(
        "\nRobotReplay analysis: PASS"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "RobotReplay ROS 2 "
            "incident analyzer"
        )
    )


    parser.add_argument(
        "bag",
        help=(
            "ROS 2 bag directory"
        ),
    )


    parser.add_argument(
        "--healthy",
        nargs="+",
        default=[
            str(path)
            for path
            in DEFAULT_HEALTHY_BASELINES
        ],
        help=(
            "Healthy baseline feature CSVs. "
            "healthy_003 is intentionally "
            "excluded from defaults."
        ),
    )


    parser.add_argument(
        "--model",
        default=str(
            DEFAULT_MODEL_PATH
        ),
    )


    parser.add_argument(
        "--output",
        default=None,
    )


    args = parser.parse_args()


    report, _ = analyze_bag(

        Path(
            args.bag
        ),

        healthy_paths=
            args.healthy,

        model_path=
            args.model,

        output_path=
            args.output,
    )


    print_report(
        report
    )


if __name__ == "__main__":
    main()
