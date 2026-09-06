#!/usr/bin/env python3

"""
Stable RobotReplay supervised-classifier interface.

The validated classifier implementation lives in:

    robotreplay.ml.fault_classifier_core

This wrapper preserves that implementation unchanged and
normalizes the public result contract so a successful
classification always explicitly contains:

    "available": True

Failures continue to raise FaultClassifierError and are
converted into "available": False by the analyzer service.
"""

from __future__ import annotations


from robotreplay.ml.fault_classifier_core import *  # noqa: F401,F403

from robotreplay.ml.fault_classifier_core import (
    classify_mission as _classify_mission,
)


def classify_mission(
    *args,
    **kwargs,
) -> dict:

    result = _classify_mission(
        *args,
        **kwargs,
    )


    if not isinstance(
        result,
        dict,
    ):

        raise FaultClassifierError(
            (
                "Fault classifier returned "
                "an invalid result object."
            )
        )


    return {
        "available":
            True,

        **result,
    }
