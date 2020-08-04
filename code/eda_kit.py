# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import thinkstats2
import math
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from scipy.stats import entropy
from scipy.stats import ks_2samp
from sklearn import metrics

def conditional_entropy(x, cond, flag='flag'):
    """
    :var x : Pandas.DataFrame
    """
    res = 0
    column_res = {}

    cond_column = x[cond].dropna()
    for v in cond_column.unique():
        part = x[x[cond]==v]
        part_count = part.shape[0]
        part_entropy = []
        for c in part.groupby(flag)[flag].count():
            part_entropy.append( (c-0.)/part_count)
        column_res[v] = entropy(part_entropy),part_count
        res += (part_count-0.) / cond_column.shape[0] * column_res[v][0]

    return res,column_res

def roc_auc_score(x,y):
    score = metrics.roc_auc_score(x,y)
    return score

def auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    score = metrics.auc(fpr, tpr)
    return score