#!/usr/bin/env python3

from __future__ import annotations

import json
import sys

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import joblib
import numpy as np
import pandas as pd


from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


from robotreplay.ml.fault_features import (
    FEATURE_NAMES,
)


# =========================================================
# PATHS
# =========================================================

DATASET_PATH = (
    PROJECT_ROOT
    / "data/ml/fault_classifier_dataset.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models/fault_classifier.joblib"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "data/results/fault_classifier_metrics.json"
)

CONFUSION_PATH = (
    PROJECT_ROOT
    / "data/results/fault_classifier_confusion_matrix.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data/results/fault_classifier_test_predictions.csv"
)

IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data/results/fault_classifier_feature_importance.csv"
)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    print(
        "\n=== RobotReplay Fault Classifier Training ===\n"
    )


    if not DATASET_PATH.is_file():

        raise FileNotFoundError(
            (
                "Dataset not found. Run:\n"
                "python3 evaluation/"
                "build_fault_classifier_dataset.py"
            )
        )


    df = pd.read_csv(
        DATASET_PATH
    )


    train = df[
        df["split"]
        == "train"
    ].copy()


    test = df[
        df["split"]
        == "test"
    ].copy()


    if train.empty:

        raise RuntimeError(
            "Training split is empty."
        )


    if test.empty:

        raise RuntimeError(
            "Test split is empty."
        )


    # -----------------------------------------------------
    # Strict source-mission separation check
    # -----------------------------------------------------

    train_sources = set(
        train[
            "source_mission"
        ].unique()
    )


    test_sources = set(
        test[
            "source_mission"
        ].unique()
    )


    overlap = (
        train_sources
        &
        test_sources
    )


    if overlap:

        raise RuntimeError(
            (
                "SOURCE-MISSION LEAKAGE DETECTED: "
                f"{sorted(overlap)}"
            )
        )


    print(
        "Training source missions:"
    )

    for source in sorted(
        train_sources
    ):

        print(
            f"  • {source}"
        )


    print(
        "\nHeld-out test source missions:"
    )

    for source in sorted(
        test_sources
    ):

        print(
            f"  • {source}"
        )


    X_train = train[
        FEATURE_NAMES
    ].copy()


    y_train = train[
        "label"
    ].copy()


    X_test = test[
        FEATURE_NAMES
    ].copy()


    y_test = test[
        "label"
    ].copy()


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=
            600,

        max_depth=
            None,

        min_samples_leaf=
            2,

        max_features=
            "sqrt",

        class_weight=
            "balanced_subsample",

        random_state=
            42,

        n_jobs=
            -1,
    )


    model.fit(
        X_train,
        y_train,
    )


    predictions = model.predict(
        X_test
    )


    probabilities = model.predict_proba(
        X_test
    )


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = float(
        accuracy_score(
            y_test,
            predictions,
        )
    )


    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(

        y_test,
        predictions,

        average=
            "macro",

        zero_division=
            0,
    )


    classes = list(
        model.classes_
    )


    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes,
    )


    report = classification_report(

        y_test,
        predictions,

        labels=
            classes,

        output_dict=
            True,

        zero_division=
            0,
    )


    # Healthy false positive rate:
    # healthy windows incorrectly classified as a fault.
    healthy_mask = (
        y_test
        == "Healthy"
    )


    if healthy_mask.any():

        healthy_false_positive_rate = float(
            (
                predictions[
                    healthy_mask.to_numpy()
                ]
                != "Healthy"
            ).mean()
        )

    else:

        healthy_false_positive_rate = None


    # -----------------------------------------------------
    # Prediction output
    # -----------------------------------------------------

    prediction_output = test[
        [
            "sample_id",
            "source_mission",
            "label",
        ]
    ].copy()


    prediction_output[
        "predicted_label"
    ] = predictions


    prediction_output[
        "correct"
    ] = (
        prediction_output[
            "label"
        ]
        ==
        prediction_output[
            "predicted_label"
        ]
    )


    for index, class_name in enumerate(
        classes
    ):

        safe_name = (
            class_name
            .lower()
            .replace(
                " / ",
                "_",
            )
            .replace(
                " ",
                "_",
            )
            .replace(
                "/",
                "_",
            )
        )


        prediction_output[
            f"prob_{safe_name}"
        ] = probabilities[
            :,
            index
        ]


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    confusion_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    )


    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------

    importance_df = pd.DataFrame(
        {
            "feature":
                FEATURE_NAMES,

            "importance":
                model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )


    # -----------------------------------------------------
    # Metrics JSON
    # -----------------------------------------------------

    metrics = {

        "evaluation_type":
            (
                "Held-out source-mission "
                "controlled telemetry augmentation"
            ),

        "train_source_missions":
            sorted(
                train_sources
            ),

        "test_source_missions":
            sorted(
                test_sources
            ),

        "training_samples":
            int(
                len(
                    train
                )
            ),

        "test_samples":
            int(
                len(
                    test
                )
            ),

        "classes":
            classes,

        "accuracy":
            accuracy,

        "macro_precision":
            float(
                macro_precision
            ),

        "macro_recall":
            float(
                macro_recall
            ),

        "macro_f1":
            float(
                macro_f1
            ),

        "healthy_false_positive_rate":
            (
                healthy_false_positive_rate
            ),

        "classification_report":
            report,

        "method_note":
            (
                "Training variants are controlled "
                "telemetry augmentations derived from "
                "healthy_001 and healthy_002. "
                "All test variants are derived only "
                "from held-out source mission "
                "healthy_003. Source missions do not "
                "overlap between train and test."
            ),
    }


    # -----------------------------------------------------
    # Save model package
    # -----------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    model_package = {

        "model":
            model,

        "features":
            FEATURE_NAMES,

        "classes":
            classes,

        "metrics":
            metrics,

        "training_dataset":
            str(
                DATASET_PATH
            ),

        "model_type":
            "RandomForestClassifier",

        "version":
            "1.0",
    }


    joblib.dump(
        model_package,
        MODEL_PATH,
    )


    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        )
    )


    confusion_df.to_csv(
        CONFUSION_PATH
    )


    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )


    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print(
        "\n----------------------------------------"
    )


    print(
        "HELD-OUT TEST RESULTS"
    )


    print(
        "----------------------------------------"
    )


    print(
        f"Accuracy:             "
        f"{100 * accuracy:.2f}%"
    )


    print(
        f"Macro Precision:      "
        f"{100 * macro_precision:.2f}%"
    )


    print(
        f"Macro Recall:         "
        f"{100 * macro_recall:.2f}%"
    )


    print(
        f"Macro F1:             "
        f"{100 * macro_f1:.2f}%"
    )


    if (
        healthy_false_positive_rate
        is not None
    ):

        print(
            f"Healthy false-positive:"
            f" {100 * healthy_false_positive_rate:.2f}%"
        )


    print(
        "\nPer-class metrics:"
    )


    for class_name in classes:

        values = report[
            class_name
        ]


        print(
            (
                f"\n{class_name}\n"
                f"  Precision: "
                f"{100 * values['precision']:.2f}%\n"
                f"  Recall:    "
                f"{100 * values['recall']:.2f}%\n"
                f"  F1:        "
                f"{100 * values['f1-score']:.2f}%\n"
                f"  Support:   "
                f"{int(values['support'])}"
            )
        )


    print(
        "\nConfusion matrix:"
    )


    print(
        confusion_df.to_string()
    )


    print(
        "\nTop classifier features:"
    )


    print(
        importance_df.head(
            10
        ).to_string(
            index=False
        )
    )


    print(
        "\nArtifacts:"
    )


    print(
        f"  Model:       {MODEL_PATH}"
    )


    print(
        f"  Metrics:     {METRICS_PATH}"
    )


    print(
        f"  Confusion:   {CONFUSION_PATH}"
    )


    print(
        f"  Predictions: {PREDICTIONS_PATH}"
    )


    print(
        f"  Importance:  {IMPORTANCE_PATH}"
    )


    print(
        "\nFault classifier training: PASS"
    )


if __name__ == "__main__":
    main()
