# -*- coding: utf-8 -*-
"""
Metric computation for evaluate_metrics.py, kept separate (no torch/CUDA) so
it can be tested on CPU. Covers Table 1 and Table S7 of Bravo Sanchez et al.
2021: accuracy, ROC AUC, precision, recall, F1, false positive rate, false
negative rate, top-3/top-5 accuracy, and "ROC AUC Mean Exp".
"""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             top_k_accuracy_score)


def weighted_fpr_fnr(y_true, y_pred, labels):
    """Per-class FP/(FP+TN) and FN/(FN+TP), averaged weighted by class support."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    support = cm.sum(axis=1).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        fpr = np.where(fp + tn > 0, fp / (fp + tn), 0.0)
        fnr = np.where(fn + tp > 0, fn / (fn + tp), 0.0)
    return float(np.average(fpr, weights=support)), float(np.average(fnr, weights=support))


def compute_metrics(y_true, y_pred, y_score, y_score_exp, n_classes):
    """y_score: (N, n_classes) rows summing to 1 (softmax of summed frame
    log-probs, as in call_id.py's decision rule). y_score_exp: (N, n_classes)
    mean over frames of exp(LogSoftmax output), the paper's "Mean Exp"."""
    labels = list(range(n_classes))
    fpr, fnr = weighted_fpr_fnr(y_true, y_pred, labels)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score, labels=labels, multi_class='ovr', average='weighted'),
        'precision': precision_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'top3_accuracy': top_k_accuracy_score(y_true, y_score, k=3, labels=labels),
        'top5_accuracy': top_k_accuracy_score(y_true, y_score, k=5, labels=labels),
        'roc_auc_mean_exp': roc_auc_score(y_true, y_score_exp, labels=labels,
                                          multi_class='ovr', average='weighted'),
    }
