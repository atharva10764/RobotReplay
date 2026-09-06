from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def healthy_values(
    frames: Iterable[pd.DataFrame],
    column: str,
) -> pd.Series:

    values = [
        df[column]
        for df in frames
        if column in df.columns
    ]

    if not values:
        return pd.Series(dtype=float)

    return pd.concat(
        values,
        ignore_index=True,
    ).dropna()


def unavailable(
    name: str,
    reason: str,
) -> dict:

    return {
        "name": name,
        "score": 0.0,
        "available": False,
        "reason": reason,
        "evidence": {},
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "failure_chain": [],
        "recommended_checks": [],
    }


def no_detection(
    name: str,
    evidence: dict | None = None,
) -> dict:

    return {
        "name": name,
        "score": 0.0,
        "available": True,
        "evidence": evidence or {},
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "failure_chain": [],
        "recommended_checks": [],
    }


def robust_threshold(
    values: pd.Series,
    *,
    q: float = 99.5,
    q_multiplier: float = 1.5,
    max_multiplier: float = 1.10,
    floor: float | None = None,
) -> float:

    if values.empty:
        raise ValueError(
            "Cannot derive threshold from an empty baseline."
        )

    threshold = max(
        float(
            np.percentile(
                values,
                q,
            )
        )
        * q_multiplier,

        float(
            values.max()
        )
        * max_multiplier,
    )

    if floor is not None:
        threshold = max(
            threshold,
            floor,
        )

    return float(
        threshold
    )
