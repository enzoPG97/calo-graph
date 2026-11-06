### Metrics for edge classification."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

def edge_auc(labels, scores)->float:
    """ 
    Area under the ROC curbe (probability a positive edge outsscores a negative one).
    Ties fet half credit. Return nan of only one class is present.
    """

    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores,dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores) #average ranks for ties
    return float((ranks[labels].sum() - n_pos * (n_pos+1) /2) / (n_pos*n_neg))
