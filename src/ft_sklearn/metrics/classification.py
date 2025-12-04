# accuracy, precision, recall
import numpy as np

def accuracy_score(y_true, y_pred):
    TP_AND_FN = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return TP_AND_FN / len(y_true)

def recall_score(y_true, y_pred):
    TP = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 1 and true == 1)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 0 and true == 1)
    return TP / (TP + FN)

def precision_score(y_true, y_pred):
    TP = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 1 and true == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 1 and true == 0)
    return TP / (TP + FP)


def f1_score(y_true, y_pred):
    TP = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 1 and true == 1)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 1 and true == 0)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if  pred == 0 and true == 1)

    return (2 * TP) / ((2 * TP) + FP + FN)
