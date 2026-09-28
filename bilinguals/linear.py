"""Linear models, cross-validation and stepwise feature selection.

Replaces TrainTest, CV, CV10x10, CVK_Lin, PredTestFromTrain, FS, BS, FSBS
and FindBestLinear.

Errors are always ``predicted - actual``. (The MATLAB code used
actual - predicted for L1 patients and predicted - actual for L2 patients,
then compared the two with a t-test.)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def fit_predict(X_train, y_train, X_test):
    """OLS with an intercept (MATLAB ``regress``)."""
    A = np.column_stack([np.ones(len(X_train)), X_train])
    coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
    return np.column_stack([np.ones(len(X_test)), X_test]) @ coef


def make_folds(n, k=10, repeats=10, rng=0) -> np.ndarray:
    """(repeats, n) array of balanced fold numbers. The same folds are used
    for every candidate feature set, so candidates are compared fairly.
    (CV10x10 drew fresh unbalanced folds with randi on every call.)"""
    rng = np.random.default_rng(rng)
    k = min(k, n)
    return np.stack([rng.permutation(np.arange(n) % k) for _ in range(repeats)])


def cv_predict(X, y, folds) -> np.ndarray:
    """Mean out-of-fold prediction over the repeats in ``folds``."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    preds = np.empty(folds.shape)
    for r, assign in enumerate(folds):
        for f in np.unique(assign):
            test = assign == f
            preds[r, test] = fit_predict(X[~test], y[~test], X[test])
    return preds.mean(axis=0)


def loo_predict(X, y) -> np.ndarray:
    """Leave-one-out predictions (the original CV function)."""
    return cv_predict(X, y, np.arange(len(y))[None, :])


def score(y, pred, objective="r") -> float:
    """Higher is better: Pearson r, or negative mean absolute error."""
    if objective == "mae":
        return -float(np.mean(np.abs(pred - y)))
    if np.std(pred) == 0:
        return -1.0
    return float(np.corrcoef(y, pred)[0, 1])


@dataclass
class Selection:
    selected: list
    score: float
    history: list = field(default_factory=list)


def stepwise_select(X, y, protected=("time_post",), candidates=None, objective="r",
                    max_features=40, folds=10, repeats=10, seed=0, max_rounds=6) -> Selection:
    """Forward then backward selection, repeated while the score improves.

    ``protected`` features are always in the model and never removed;
    ``candidates`` defaults to every other column. Forward adds the best
    feature while it improves the score; backward removes a feature while
    that improves the score. Up to ``max_rounds`` forward/backward rounds
    (the original stopped after 6 cycles).
    """
    cols = list(X.columns)
    Xa = X.to_numpy(float)
    y = np.asarray(y, float)
    fold_ids = make_folds(len(y), folds, repeats, seed)
    protected = [c for c in protected if c in cols]
    usable = [c for c in (candidates or cols) if c not in protected and np.ptp(X[c].to_numpy()) > 0]
    cache = {}

    def evaluate(feats):
        key = frozenset(feats)
        if key not in cache:
            idx = [cols.index(f) for f in feats]
            cache[key] = score(y, cv_predict(Xa[:, idx], y, fold_ids), objective) if idx else -np.inf
        return cache[key]

    selected = list(protected)
    best = evaluate(selected)
    history = []
    for _ in range(max_rounds):
        start = best
        while len(selected) < max_features:
            options = [c for c in usable if c not in selected]
            if not options:
                break
            scores = [evaluate(selected + [c]) for c in options]
            i = int(np.argmax(scores))
            if scores[i] <= best:
                break
            selected.append(options[i])
            best = scores[i]
            history.append(("add", options[i], best))
        while True:
            removable = [c for c in selected if c not in protected]
            if not removable:
                break
            scores = [evaluate([f for f in selected if f != c]) for c in removable]
            i = int(np.argmax(scores))
            if scores[i] <= best:
                break
            selected.remove(removable[i])
            best = scores[i]
            history.append(("drop", removable[i], best))
        if best <= start:
            break
    return Selection(selected, best, history)
