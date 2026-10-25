"""
evaluation.py

Final evaluation pipeline for the FYP:

LLM-Assisted Bug Report Triaging and Prioritization System
for Software Quality Management.

Models evaluated:
1. Naive Bayes
2. Support Vector Machine
3. Random Forest
4. LLM Zero-Shot (facebook/bart-large-mnli)

Evaluation methodology:
- Ground-truth labelled Eclipse bug reports stored in EvalExample.
- Traditional machine-learning models use TF-IDF.
- Stratified cross-validation is used when possible.
- If a severity class has only one example, standard KFold is used instead.
- TF-IDF is fitted separately inside every training fold using Pipeline.
- LLM predictions use the previously generated zero-shot predictions
  stored in llm_predictions_v2.csv / EvalExample.

Metrics:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Per-class Precision / Recall / F1 / Support
- Confusion Matrix

Run:
    python -m app.evaluation
"""

import json
import os
from collections import Counter

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from .database import SessionLocal, Base, engine
from . import models


# ============================================================
# Configuration
# ============================================================

LABEL_ORDER = [
    "Critical",
    "High",
    "Medium",
    "Low",
]

N_SPLITS = 5
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 500

CONFUSION_MATRIX_FILE = "llm_confusion_matrix.json"


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1
    - Per-class metrics
    """

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )

    per_class = {}

    for i, label in enumerate(LABEL_ORDER):
        per_class[label] = {
            "precision": round(
                float(precision[i]),
                3,
            ),
            "recall": round(
                float(recall[i]),
                3,
            ),
            "f1": round(
                float(f1[i]),
                3,
            ),
            "support": int(
                support[i]
            ),
        }

    macro_precision = float(
        precision.mean()
    )

    macro_recall = float(
        recall.mean()
    )

    macro_f1 = float(
        f1.mean()
    )

    return (
        round(float(accuracy), 3),
        round(macro_precision, 3),
        round(macro_recall, 3),
        round(macro_f1, 3),
        per_class,
    )


# ============================================================
# Result printing
# ============================================================

def print_model_result(
    model_name,
    result,
):
    """
    Print one model's result clearly.
    """

    (
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        per_class,
    ) = result

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"Accuracy        : "
        f"{accuracy:.3f} "
        f"({accuracy:.1%})"
    )

    print(
        f"Macro Precision : "
        f"{macro_precision:.3f}"
    )

    print(
        f"Macro Recall    : "
        f"{macro_recall:.3f}"
    )

    print(
        f"Macro F1        : "
        f"{macro_f1:.3f}"
    )

    print()
    print("Per-class results:")

    print(
        f"{'Class':<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'Support':>12}"
    )

    print("-" * 60)

    for label in LABEL_ORDER:
        values = per_class[label]

        print(
            f"{label:<12}"
            f"{values['precision']:>12.3f}"
            f"{values['recall']:>12.3f}"
            f"{values['f1']:>12.3f}"
            f"{values['support']:>12d}"
        )


# ============================================================
# Cross-validation strategy
# ============================================================

def get_cv_strategy(distribution):
    """
    Use StratifiedKFold where possible.

    If one class has only one sample,
    stratification is impossible, so
    use standard KFold instead.
    """

    minimum_class_count = min(
        distribution.values()
    )

    if minimum_class_count >= N_SPLITS:

        print()
        print(
            "Using 5-fold Stratified "
            "Cross-Validation."
        )

        return StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    elif minimum_class_count >= 2:

        n_splits = (
            minimum_class_count
        )

        print()
        print(
            "WARNING: The smallest "
            "severity class contains "
            f"only {minimum_class_count} "
            "examples."
        )

        print(
            f"Using {n_splits}-fold "
            "Stratified "
            "Cross-Validation."
        )

        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    else:

        print()
        print("=" * 70)
        print(
            "WARNING: SEVERE CLASS "
            "IMBALANCE"
        )
        print("=" * 70)

        print(
            "At least one severity "
            "class contains only one "
            "example."
        )

        print(
            "Stratified cross-validation "
            "is therefore not possible."
        )

        print(
            "Using standard 5-fold "
            "KFold instead."
        )

        print(
            "Accuracy must be interpreted "
            "together with Macro Precision, "
            "Macro Recall and Macro F1."
        )

        print(
            "For a stronger final research "
            "evaluation, additional labelled "
            "minority-class examples are "
            "recommended."
        )

        return KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )


# ============================================================
# Full evaluation
# ============================================================

def run_full_evaluation():
    """
    Run final model evaluation and
    save results into SQLite.
    """

    Base.metadata.create_all(
        bind=engine
    )

    db = SessionLocal()

    try:

        # ====================================================
        # Load dataset
        # ====================================================

        examples = (
            db.query(
                models.EvalExample
            )
            .order_by(
                models.EvalExample.id
            )
            .all()
        )

        if not examples:

            print()
            print(
                "ERROR: No evaluation "
                "examples were found."
            )

            print(
                "Run:"
            )

            print(
                "python -m app.seed_data"
            )

            return

        # ====================================================
        # Keep valid ground-truth labels only
        # ====================================================

        valid_examples = [
            example
            for example in examples
            if example.mapped_severity
            in LABEL_ORDER
        ]

        if not valid_examples:

            print(
                "ERROR: No valid labelled "
                "evaluation examples found."
            )

            return

        texts = [
            example.clean_text or ""
            for example
            in valid_examples
        ]

        y_true = [
            example.mapped_severity
            for example
            in valid_examples
        ]

        # ====================================================
        # Dataset summary
        # ====================================================

        print()
        print("=" * 70)
        print(
            "FINAL MODEL EVALUATION"
        )
        print("=" * 70)

        print()
        print(
            f"Evaluation examples: "
            f"{len(valid_examples)}"
        )

        distribution = Counter(
            y_true
        )

        print()
        print(
            "Ground-truth severity "
            "distribution:"
        )

        for label in LABEL_ORDER:

            count = (
                distribution.get(
                    label,
                    0,
                )
            )

            percentage = (
                count
                / len(y_true)
                * 100
            )

            print(
                f"  {label:<10}: "
                f"{count:>3} "
                f"({percentage:>5.1f}%)"
            )

        # ====================================================
        # Select CV strategy
        # ====================================================

        cv = get_cv_strategy(
            distribution
        )

        # ====================================================
        # Result containers
        # ====================================================

        results = {}
        predictions = {}

        # ====================================================
        # Naive Bayes
        # ====================================================

        print()
        print(
            "Evaluating "
            "Naive Bayes..."
        )

        nb_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=(
                            MAX_TFIDF_FEATURES
                        ),
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    MultinomialNB(),
                ),
            ]
        )

        nb_preds = cross_val_predict(
            nb_pipeline,
            texts,
            y_true,
            cv=cv,
        )

        predictions[
            "Naive Bayes"
        ] = list(
            nb_preds
        )

        results[
            "Naive Bayes"
        ] = calculate_metrics(
            y_true,
            nb_preds,
        )

        # ====================================================
        # SVM
        # ====================================================

        print(
            "Evaluating SVM..."
        )

        svm_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=(
                            MAX_TFIDF_FEATURES
                        ),
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    SVC(
                        class_weight=(
                            "balanced"
                        ),
                        kernel="linear",
                    ),
                ),
            ]
        )

        svm_preds = cross_val_predict(
            svm_pipeline,
            texts,
            y_true,
            cv=cv,
        )

        predictions[
            "SVM"
        ] = list(
            svm_preds
        )

        results[
            "SVM"
        ] = calculate_metrics(
            y_true,
            svm_preds,
        )

        # ====================================================
        # Random Forest
        # ====================================================

        print(
            "Evaluating "
            "Random Forest..."
        )

        rf_pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=(
                            MAX_TFIDF_FEATURES
                        ),
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        class_weight=(
                            "balanced"
                        ),
                        n_estimators=100,
                        random_state=(
                            RANDOM_STATE
                        ),
                    ),
                ),
            ]
        )

        rf_preds = cross_val_predict(
            rf_pipeline,
            texts,
            y_true,
            cv=cv,
        )

        predictions[
            "Random Forest"
        ] = list(
            rf_preds
        )

        results[
            "Random Forest"
        ] = calculate_metrics(
            y_true,
            rf_preds,
        )

        # ====================================================
        # LLM Zero-Shot
        # ====================================================

        print(
            "Evaluating "
            "LLM Zero-Shot..."
        )

        missing_llm = [
            example.id
            for example
            in valid_examples
            if (
                example.predicted_severity
                not in LABEL_ORDER
            )
        ]

        if missing_llm:

            print()
            print(
                "WARNING: Some examples "
                "have missing or invalid "
                "LLM predictions."
            )

            print(
                "Affected IDs:"
            )

            print(
                missing_llm
            )

            print(
                "Medium will be used "
                "as the fallback "
                "severity."
            )

        llm_preds = [
            (
                example.predicted_severity
                if (
                    example.predicted_severity
                    in LABEL_ORDER
                )
                else "Medium"
            )
            for example
            in valid_examples
        ]

        predictions[
            "LLM (Zero-Shot)"
        ] = list(
            llm_preds
        )

        results[
            "LLM (Zero-Shot)"
        ] = calculate_metrics(
            y_true,
            llm_preds,
        )

        # ====================================================
        # Save evaluation results to SQLite
        # ====================================================

        print()
        print(
            "Saving evaluation "
            "results to SQLite..."
        )

        db.query(
            models.ModelEvalResult
        ).delete()

        for (
            model_name,
            result,
        ) in results.items():

            (
                accuracy,
                macro_precision,
                macro_recall,
                macro_f1,
                per_class,
            ) = result

            row = (
                models.ModelEvalResult(
                    model_name=(
                        model_name
                    ),
                    accuracy=(
                        accuracy
                    ),
                    macro_precision=(
                        macro_precision
                    ),
                    macro_recall=(
                        macro_recall
                    ),
                    macro_f1=(
                        macro_f1
                    ),
                    per_class_report=(
                        json.dumps(
                            per_class
                        )
                    ),
                )
            )

            db.add(
                row
            )

        db.commit()

        # ====================================================
        # LLM confusion matrix
        # ====================================================

        llm_cm = confusion_matrix(
            y_true,
            llm_preds,
            labels=LABEL_ORDER,
        )

        confusion_output = {
            "model_name": (
                "LLM (Zero-Shot)"
            ),
            "labels": (
                LABEL_ORDER
            ),
            "matrix": (
                llm_cm.tolist()
            ),
        }

        with open(
            CONFUSION_MATRIX_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                confusion_output,
                file,
                indent=4,
            )

        # ====================================================
        # Print detailed model results
        # ====================================================

        for (
            model_name,
            result,
        ) in results.items():

            print_model_result(
                model_name,
                result,
            )

        # ====================================================
        # Print confusion matrix
        # ====================================================

        print()
        print("=" * 70)
        print(
            "LLM ZERO-SHOT "
            "CONFUSION MATRIX"
        )
        print("=" * 70)

        print(
            "Rows = Actual severity"
        )

        print(
            "Columns = "
            "Predicted severity"
        )

        print()

        header = (
            f"{'Actual':<12}"
            + "".join(
                f"{label:>12}"
                for label
                in LABEL_ORDER
            )
        )

        print(
            header
        )

        print(
            "-" * len(header)
        )

        for i, label in enumerate(
            LABEL_ORDER
        ):

            row_values = "".join(
                f"{value:>12}"
                for value
                in llm_cm[i]
            )

            print(
                f"{label:<12}"
                f"{row_values}"
            )

        # ====================================================
        # Final comparison table
        # ====================================================

        print()
        print("=" * 90)
        print(
            "FINAL MODEL COMPARISON"
        )
        print("=" * 90)

        print(
            f"{'Model':<25}"
            f"{'Accuracy':>12}"
            f"{'Macro P':>12}"
            f"{'Macro R':>12}"
            f"{'Macro F1':>12}"
        )

        print(
            "-" * 73
        )

        sorted_results = sorted(
            results.items(),
            key=lambda item:
            item[1][3],
            reverse=True,
        )

        for (
            model_name,
            result,
        ) in sorted_results:

            (
                accuracy,
                precision,
                recall,
                f1,
                _,
            ) = result

            print(
                f"{model_name:<25}"
                f"{accuracy:>11.1%}"
                f"{precision:>12.3f}"
                f"{recall:>12.3f}"
                f"{f1:>12.3f}"
            )

        # ====================================================
        # Best models
        # ====================================================

        best_accuracy = max(
            results.items(),
            key=lambda item:
            item[1][0],
        )

        best_f1 = max(
            results.items(),
            key=lambda item:
            item[1][3],
        )

        print()
        print(
            "Best Accuracy:"
        )

        print(
            f"  {best_accuracy[0]} "
            f"({best_accuracy[1][0]:.1%})"
        )

        print()
        print(
            "Best Macro F1:"
        )

        print(
            f"  {best_f1[0]} "
            f"({best_f1[1][3]:.3f})"
        )

        print()
        print(
            "Evaluation successfully "
            "saved to database."
        )

        print(
            "Confusion matrix saved to:"
        )

        print(
            os.path.abspath(
                CONFUSION_MATRIX_FILE
            )
        )

        print()

    finally:
        db.close()


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    run_full_evaluation()