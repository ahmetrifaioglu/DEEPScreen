import numpy as np
import copy
from math import sqrt
from scipy import stats
from sklearn import preprocessing, metrics

def prec_rec_f1_acc_mcc(y_true, y_pred):
    performance_threshold_dict  = dict()
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    performance_threshold_dict["Precision"] = precision
    performance_threshold_dict["Recall"] = recall
    performance_threshold_dict["F1-Score"] = f1_score
    performance_threshold_dict["Accuracy"] = accuracy
    performance_threshold_dict["MCC"] = mcc

    return performance_threshold_dict

def get_list_of_scores():
    return ["Precision", "Recall", "F1-Score", "Accuracy", "MCC"]