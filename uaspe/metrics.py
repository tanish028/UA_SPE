"""Evaluation metrics.

Positive class = 1 (minority, fraud). Threshold-based metrics predict label 1 when the
predicted probability of class 1 is >= the threshold.
"""
import math

import numpy as np
from sklearn.metrics import average_precision_score

from uaspe import settings


def predict_labels(y_prob, threshold=settings.CLASSIFICATION_THRESHOLD):
    """Probability of class 1 -> 0/1 label. A probability equal to the threshold counts as 1."""
    return (np.asarray(y_prob) >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    """TP, FP, TN, FN with 1 as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def ratio(numerator, denominator):
    """numerator / denominator, or 0.0 when the denominator is 0.

    For example, precision is 0 when nothing is predicted positive (as in scikit-learn).
    """
    if denominator > 0:
        return numerator / denominator
    return 0.0


def evaluate(y_true, y_prob, threshold=settings.CLASSIFICATION_THRESHOLD):
    """All metrics for one set of predictions. y_prob = predicted probability of class 1."""
    y_true = np.asarray(y_true)
    y_pred = predict_labels(y_prob, threshold)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    # threshold-based metrics
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)          # true positive rate
    specificity = ratio(tn, tn + fp)     # true negative rate
    f1 = ratio(2 * precision * recall, precision + recall)

    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ratio(tp * tn - fp * fn, mcc_denominator)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        # balance between the two class-wise recalls
        "gmean_recall_specificity": math.sqrt(recall * specificity),
        # the SPE paper's "G-mean"; reported only for comparison with the paper
        "gmean_precision_recall": math.sqrt(precision * recall),
        # average precision: sum over thresholds of (R_n - R_{n-1}) * P_n
        # step-wise, no interpolation (not the trapezoidal area), and threshold-free
        "aucprc_average_precision": float(average_precision_score(y_true, y_prob)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
