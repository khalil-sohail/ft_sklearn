"""Classification metrics for model evaluation.

Provides functions to compute common classification performance metrics.
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    
    Proportion of correct predictions among the total number of predictions.
    Useful for balanced datasets.
    
    Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: Accuracy score between 0 and 1. Higher is better.
    """
    TP_AND_FN = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return TP_AND_FN / len(y_true)


def recall_score(y_true, y_pred):
    """Recall (sensitivity or true positive rate) classification score.
    
    Proportion of actual positive cases that were correctly identified.
    Useful when false negatives are costly.
    
    Formula:
        Recall = TP / (TP + FN)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: Recall score between 0 and 1. Higher is better.
    """
    TP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 1)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true == 1)
    return TP / (TP + FN)


def precision_score(y_true, y_pred):
    """Precision classification score.
    
    Proportion of predicted positive cases that were actually positive.
    Useful when false positives are costly.
    
    Formula:
        Precision = TP / (TP + FP)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: Precision score between 0 and 1. Higher is better.
    """
    TP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 0)
    return TP / (TP + FP)


def f1_score(y_true, y_pred):
    """F1 score (harmonic mean of precision and recall).
    
    Provides a single score that balances precision and recall. Useful for
    imbalanced datasets where you care about both false positives and negatives.
    
    Formula:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
          = (2 * TP) / (2 * TP + FP + FN)
    
    Args:
        y_true (array-like): Ground truth binary labels.
        y_pred (array-like): Predicted binary labels.
    
    Returns:
        float: F1 score between 0 and 1. Higher is better.
    """
    TP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if pred == 1 and true == 0)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if pred == 0 and true == 1)
    return (2 * TP) / ((2 * TP) + FP + FN)
