from __future__ import annotations


import pandas as pd


from robotreplay.detectors.lidar_dropout import (
    detect_lidar_dropout,
)

from robotreplay.detectors.lidar_integrity import (
    detect_lidar_integrity,
)

from robotreplay.detectors.localization_jump import (
    detect_localization_jump,
)

from robotreplay.detectors.tf_delay import (
    detect_tf_delay,
)

from robotreplay.detectors.wheel_mismatch import (
    detect_wheel_mismatch,
)


# =========================================================
# DETECTOR ORDER
# =========================================================
#
# When two hypotheses have identical heuristic evidence
# scores, stable sorting in the analyzer preserves this
# order.
#
# Upstream sensor-integrity evidence is therefore evaluated
# before downstream transform-staleness evidence.
# =========================================================

DETECTORS = [

    detect_localization_jump,

    detect_lidar_dropout,

    detect_lidar_integrity,

    detect_tf_delay,

    detect_wheel_mismatch,
]


def run_detectors(
    mission: pd.DataFrame,
    healthy: list[pd.DataFrame],
) -> list[dict]:

    results = []


    for detector in DETECTORS:

        try:

            result = detector(
                mission,
                healthy,
            )


        except Exception as exc:

            result = {

                "name":
                    detector.__name__,

                "score":
                    0.0,

                "available":
                    False,

                "reason":
                    (
                        "Detector execution failed: "
                        f"{exc}"
                    ),

                "evidence":
                    {},

                "supporting_evidence":
                    [],

                "contradicting_evidence":
                    [],

                "failure_chain":
                    [],

                "recommended_checks":
                    [],
            }


        results.append(
            result
        )


    return results
