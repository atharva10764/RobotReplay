from __future__ import annotations

import json
import time

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from robotreplay.config import (
    DEFAULT_HEALTHY_BASELINES,
    DEFAULT_MODEL_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RUNTIME_DIR,
    DEFAULT_SAMPLE_PERIOD,
)

from robotreplay.detectors.registry import (
    run_detectors,
)

from robotreplay.ingestion.validation import (
    inspect_bag_directory,
)

from robotreplay.ml.fault_classifier import (
    FaultClassifierError,
    classify_mission,
)


class AnalysisError(
    RuntimeError
):
    pass


# =========================================================
# HEALTHY BASELINES
# =========================================================

def load_healthy_frames(
    paths: Iterable[str | Path] | None = None,
) -> list[pd.DataFrame]:

    selected = [
        Path(path)

        for path
        in (
            paths
            or DEFAULT_HEALTHY_BASELINES
        )
    ]


    missing = [
        str(path)

        for path
        in selected

        if not path.is_file()
    ]


    if missing:

        raise AnalysisError(
            (
                "Healthy baseline file(s) missing: "
                + ", ".join(
                    missing
                )
            )
        )


    return [
        pd.read_csv(
            path
        )

        for path
        in selected
    ]


# =========================================================
# ISOLATION FOREST
# =========================================================

def score_generic_anomaly(
    mission: pd.DataFrame,
    model_path: str | Path =
        DEFAULT_MODEL_PATH,
    *,
    minimum_feature_coverage: float =
        0.70,
) -> tuple[
    pd.DataFrame,
    dict,
]:

    model_path = Path(
        model_path
    )


    if not model_path.is_file():

        return (
            mission.copy(),

            {
                "available":
                    False,

                "reason":
                    (
                        "Healthy-behaviour model "
                        f"not found: {model_path}"
                    ),

                "model":
                    "Isolation Forest",

                "feature_coverage":
                    0.0,
            },
        )


    package = joblib.load(
        model_path
    )


    required_features = list(
        package[
            "features"
        ]
    )


    present_features = [
        feature

        for feature
        in required_features

        if feature
        in mission.columns
    ]


    missing_features = [
        feature

        for feature
        in required_features

        if feature
        not in mission.columns
    ]


    if required_features:

        coverage = (
            len(
                present_features
            )
            /
            len(
                required_features
            )
        )

    else:

        coverage = 0.0


    scored = mission.copy()


    if (
        coverage
        < minimum_feature_coverage
    ):

        return (
            scored,

            {
                "available":
                    False,

                "reason":
                    (
                        f"Only {100 * coverage:.0f}% "
                        "of Isolation Forest "
                        "features are available."
                    ),

                "model":
                    "Isolation Forest",

                "feature_coverage":
                    round(
                        coverage,
                        4,
                    ),

                "present_features":
                    present_features,

                "missing_features":
                    missing_features,
            },
        )


    X = pd.DataFrame(
        index=scored.index
    )


    for feature in required_features:

        if feature in scored.columns:

            X[
                feature
            ] = scored[
                feature
            ]

        else:

            X[
                feature
            ] = np.nan


    try:

        X_imputed = (
            package[
                "imputer"
            ].transform(
                X
            )
        )


        X_scaled = (
            package[
                "scaler"
            ].transform(
                X_imputed
            )
        )


        scores = -(
            package[
                "model"
            ].score_samples(
                X_scaled
            )
        )


    except Exception as exc:

        return (
            scored,

            {
                "available":
                    False,

                "reason":
                    (
                        "Isolation Forest scoring "
                        f"failed: {exc}"
                    ),

                "model":
                    "Isolation Forest",

                "feature_coverage":
                    round(
                        coverage,
                        4,
                    ),
            },
        )


    threshold = float(
        package[
            "threshold"
        ]
    )


    anomaly = (
        scores
        > threshold
    )


    scored[
        "anomaly_score"
    ] = scores


    scored[
        "is_anomaly"
    ] = anomaly.astype(
        int
    )


    first_anomaly_s = None


    if (
        anomaly.any()
        and
        "time"
        in scored.columns
    ):

        first_anomaly_s = float(
            scored.loc[
                anomaly,
                "time",
            ].iloc[
                0
            ]
        )


    return (
        scored,

        {
            "available":
                True,

            "model":
                "Isolation Forest",

            "feature_coverage":
                round(
                    coverage,
                    4,
                ),

            "present_features":
                present_features,

            "missing_features":
                missing_features,

            "anomaly_threshold":
                round(
                    threshold,
                    6,
                ),

            "anomaly_window_rate":
                round(
                    float(
                        anomaly.mean()
                    ),
                    6,
                ),

            "anomalous_windows":
                int(
                    anomaly.sum()
                ),

            "total_windows":
                int(
                    len(
                        scored
                    )
                ),

            "peak_anomaly_score":
                (
                    round(
                        float(
                            np.max(
                                scores
                            )
                        ),
                        6,
                    )

                    if len(
                        scores
                    )

                    else None
                ),

            "first_anomaly_s":
                (
                    round(
                        first_anomaly_s,
                        3,
                    )

                    if first_anomaly_s
                    is not None

                    else None
                ),
        },
    )


# =========================================================
# RANDOM FOREST FAULT CLASSIFIER
# =========================================================

def run_fault_classifier(
    mission: pd.DataFrame,
) -> dict:

    try:

        return classify_mission(
            mission
        )


    except FaultClassifierError as exc:

        return {
            "available":
                False,

            "reason":
                str(
                    exc
                ),

            "model":
                "Random Forest",
        }


    except Exception as exc:

        return {
            "available":
                False,

            "reason":
                (
                    "Fault classification failed: "
                    f"{exc}"
                ),

            "model":
                "Random Forest",
        }


# =========================================================
# INTELLIGENCE AGREEMENT
# =========================================================

def build_intelligence_agreement(
    mission_status: str,
    root_cause: str,
    evidence_strength: float,
    classifier: dict,
) -> dict:

    if not classifier.get(
        "available"
    ):

        return {
            "available":
                False,

            "reason":
                classifier.get(
                    "reason",
                    (
                        "Fault classifier unavailable."
                    ),
                ),
        }


    ml_prediction = (
        classifier.get(
            "prediction"
        )
    )


    ml_probability = float(
        classifier.get(
            "prediction_probability",
            0.0,
        )
    )


    if mission_status == "HEALTHY":

        evidence_prediction = (
            "Healthy"
        )


    elif mission_status == "ANOMALOUS":

        evidence_prediction = (
            root_cause
        )


    else:

        evidence_prediction = None


    agreement = (
        evidence_prediction
        is not None
        and
        evidence_prediction
        == ml_prediction
    )


    if agreement:

        state = (
            "AGREEMENT"
        )

        summary = (
            "Supervised ML and ROS evidence "
            "reasoning independently support "
            "the same mission diagnosis."
        )


    elif evidence_prediction is None:

        state = (
            "EVIDENCE_UNRESOLVED"
        )

        summary = (
            "The supervised classifier produced "
            "a prediction while ROS-specific "
            "evidence reasoning remained unresolved."
        )


    else:

        state = (
            "DISAGREEMENT"
        )

        summary = (
            "Supervised ML and ROS evidence "
            "reasoning disagree. RobotReplay "
            "retains the explainable ROS-evidence "
            "diagnosis and surfaces the disagreement."
        )


    return {
        "available":
            True,

        "state":
            state,

        "agreement":
            agreement,

        "evidence_prediction":
            evidence_prediction,

        "evidence_strength":
            float(
                evidence_strength
            ),

        "ml_prediction":
            ml_prediction,

        "ml_probability":
            ml_probability,

        "summary":
            summary,
    }


# =========================================================
# ANALYZE FEATURE DATAFRAME
# =========================================================

def analyze_features(
    mission: pd.DataFrame,
    healthy_frames: list[pd.DataFrame],
    *,
    model_path: str | Path =
        DEFAULT_MODEL_PATH,
) -> tuple[
    dict,
    pd.DataFrame,
]:

    # -----------------------------------------------------
    # 1. Isolation Forest
    # -----------------------------------------------------

    (
        scored,
        ml_anomaly,
    ) = score_generic_anomaly(
        mission,
        model_path,
    )


    # -----------------------------------------------------
    # 2. ROS-aware evidence detectors
    # -----------------------------------------------------

    hypotheses = run_detectors(
        scored,
        healthy_frames,
    )


    hypotheses = sorted(
        hypotheses,

        key=lambda hypothesis:
            float(
                hypothesis.get(
                    "score",
                    0.0,
                )
            ),

        reverse=True,
    )


    if hypotheses:

        top = hypotheses[
            0
        ]

    else:

        top = {
            "name":
                "No supported fault detected",

            "score":
                0.0,

            "available":
                False,
        }


    top_score = float(
        top.get(
            "score",
            0.0,
        )
    )


    available_detectors = [
        hypothesis

        for hypothesis
        in hypotheses

        if hypothesis.get(
            "available",
            True,
        )
    ]


    # -----------------------------------------------------
    # 3. Evidence-based mission state
    # -----------------------------------------------------

    if top_score >= 0.55:

        mission_status = (
            "ANOMALOUS"
        )

        root_cause = (
            top[
                "name"
            ]
        )

        first_event = (
            top.get(
                "first_event_s"
            )
        )


    elif (
        ml_anomaly.get(
            "available"
        )
        and
        float(
            ml_anomaly.get(
                "anomaly_window_rate",
                0.0,
            )
        )
        >= 0.05
    ):

        mission_status = (
            "ANOMALOUS_UNCLASSIFIED"
        )

        root_cause = (
            "Unclassified telemetry anomaly"
        )

        first_event = (
            ml_anomaly.get(
                "first_anomaly_s"
            )
        )


    elif not available_detectors:

        mission_status = (
            "INSUFFICIENT_TELEMETRY"
        )

        root_cause = (
            "Insufficient telemetry "
            "for supported diagnosis"
        )

        first_event = None


    else:

        mission_status = (
            "HEALTHY"
        )

        root_cause = (
            "No supported fault detected"
        )

        first_event = None


    # -----------------------------------------------------
    # 4. Supervised fault classifier
    # -----------------------------------------------------

    fault_classifier = (
        run_fault_classifier(
            scored
        )
    )


    # -----------------------------------------------------
    # 5. Cross-check intelligence layers
    # -----------------------------------------------------

    intelligence_agreement = (
        build_intelligence_agreement(
            mission_status,
            root_cause,
            top_score,
            fault_classifier,
        )
    )


    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    report = {
        "schema_version":
            "2.0",

        "mission":
            None,

        "mission_status":
            mission_status,

        "root_cause":
            root_cause,

        "first_abnormal_time_s":
            first_event,

        "evidence_strength":
            round(
                top_score,
                3,
            ),

        "generic_ml":
            ml_anomaly,

        "fault_classifier":
            fault_classifier,

        "intelligence_agreement":
            intelligence_agreement,

        "ranked_hypotheses":
            hypotheses,

        "supporting_evidence":
            (
                top.get(
                    "supporting_evidence",
                    [],
                )

                if top_score > 0

                else []
            ),

        "contradicting_evidence":
            (
                top.get(
                    "contradicting_evidence",
                    [],
                )

                if top_score > 0

                else []
            ),

        "failure_chain":
            (
                top.get(
                    "failure_chain",
                    [],
                )

                if top_score > 0

                else []
            ),

        "recommended_checks":
            (
                top.get(
                    "recommended_checks",
                    [],
                )

                if top_score > 0

                else []
            ),

        "method_note":
            (
                "RobotReplay combines three layers: "
                "(1) an Isolation Forest trained on "
                "healthy telemetry, (2) a supervised "
                "Random Forest fault classifier, and "
                "(3) ROS-specific evidence reasoning. "
                "Ground-truth fault-event markers are "
                "not used by any production diagnosis layer."
            ),
    }


    return (
        report,
        scored,
    )


# =========================================================
# ANALYZE ROS BAG
# =========================================================

def analyze_bag(
    bag_path: str | Path,
    *,
    healthy_paths: Iterable[
        str | Path
    ]
    | None = None,
    model_path: str | Path =
        DEFAULT_MODEL_PATH,
    sample_period: float =
        DEFAULT_SAMPLE_PERIOD,
    runtime_dir: str | Path | None =
        DEFAULT_RUNTIME_DIR,
    results_dir: str | Path | None =
        DEFAULT_RESULTS_DIR,
    output_path: str | Path | None =
        None,
    persist: bool =
        True,
) -> tuple[
    dict,
    pd.DataFrame,
]:

    start_time = (
        time.perf_counter()
    )


    bag_path = Path(
        bag_path
    ).expanduser().resolve()


    # -----------------------------------------------------
    # Validate ROS bag
    # -----------------------------------------------------

    metadata = inspect_bag_directory(
        bag_path
    )


    mission_name = (
        bag_path.name
    )


    healthy_frames = (
        load_healthy_frames(
            healthy_paths
        )
    )


    # -----------------------------------------------------
    # Runtime output
    # -----------------------------------------------------

    if runtime_dir is None:

        runtime_root = (
            bag_path.parent
        )

    else:

        runtime_root = Path(
            runtime_dir
        )


    runtime_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    feature_csv = (
        runtime_root
        /
        f"{mission_name}_features.csv"
    )


    # -----------------------------------------------------
    # Feature extraction
    # -----------------------------------------------------

    try:

        from robotreplay.features.extract_features import (
            extract_features,
        )


    except Exception as exc:

        raise AnalysisError(
            (
                "ROS 2 feature extraction is unavailable. "
                "Source ROS 2 before running RobotReplay."
            )
        ) from exc


    extracted = extract_features(
        bag_path,
        feature_csv,
        sample_period=
            sample_period,
        label=
            "unknown",
    )


    if isinstance(
        extracted,
        pd.DataFrame,
    ):

        mission = (
            extracted
        )

    else:

        mission = pd.read_csv(
            feature_csv
        )


    # -----------------------------------------------------
    # Full RobotReplay intelligence pipeline
    # -----------------------------------------------------

    (
        report,
        scored,
    ) = analyze_features(
        mission,
        healthy_frames,
        model_path=
            model_path,
    )


    elapsed = (
        time.perf_counter()
        - start_time
    )


    if (
        elapsed > 0
        and
        metadata.duration_s > 0
    ):

        realtime_factor = (
            metadata.duration_s
            / elapsed
        )

    else:

        realtime_factor = None


    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    report[
        "mission"
    ] = mission_name


    report[
        "bag"
    ] = metadata.to_dict()


    report[
        "capabilities"
    ] = metadata.capabilities


    report[
        "performance"
    ] = {
        "processing_time_s":
            round(
                elapsed,
                3,
            ),

        "bag_duration_s":
            round(
                metadata.duration_s,
                3,
            ),

        "realtime_factor":
            (
                round(
                    realtime_factor,
                    2,
                )

                if realtime_factor
                is not None

                else None
            ),

        "feature_rows":
            int(
                len(
                    scored
                )
            ),

        "sample_period_s":
            sample_period,
    }


    # -----------------------------------------------------
    # Feature artifact
    # -----------------------------------------------------

    scored.to_csv(
        feature_csv,
        index=False,
    )


    report[
        "artifacts"
    ] = {
        "features_csv":
            (
                str(
                    feature_csv
                )

                if persist

                else None
            ),

        "incident_json":
            None,
    }


    # -----------------------------------------------------
    # Persist report
    # -----------------------------------------------------

    if persist:

        if output_path is None:

            result_root = Path(
                results_dir
                or DEFAULT_RESULTS_DIR
            )


            result_root.mkdir(
                parents=True,
                exist_ok=True,
            )


            output = (
                result_root
                /
                f"{mission_name}_incident.json"
            )

        else:

            output = Path(
                output_path
            )


            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


        report[
            "artifacts"
        ][
            "incident_json"
        ] = str(
            output
        )


        output.write_text(
            json.dumps(
                report,
                indent=2,
            )
        )


    return (
        report,
        scored,
    )
