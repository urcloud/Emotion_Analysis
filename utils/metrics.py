import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# =========================================================
# Calculate Metrics
# =========================================================

def calculate_metrics(y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }


# =========================================================
# Classification Report
# =========================================================

def get_classification_report(y_true, y_pred):

    return classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )


# =========================================================
# Confusion Matrix
# =========================================================

def get_confusion_matrix(y_true, y_pred):

    return confusion_matrix(
        y_true,
        y_pred
    )


# =========================================================
# Print Metrics
# =========================================================

def print_metrics(metrics):

    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(f"Macro Precision   : {metrics['macro_precision']:.4f}")
    print(f"Macro Recall      : {metrics['macro_recall']:.4f}")
    print(f"Macro F1          : {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall   : {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1       : {metrics['weighted_f1']:.4f}")