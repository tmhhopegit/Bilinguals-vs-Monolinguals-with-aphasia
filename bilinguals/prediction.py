"""Does a lesion model trained on L1 patients predict L2 patients as well?

Replaces Analysis 1 and Analysis 4 of RebootBilAnalysis, plus PermutePreds,
LHget, GetAllpreds and PlotErrs.

Both groups are now scored out of sample:
* L1 patients by nested cross-validation (features are chosen without the
  patient being predicted);
* L2 patients by a model selected and fitted on all L1 patients.
Previously L1 accuracy came from the same patients used to choose the
features, which flattered L1 relative to L2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .config import Selection as SelectionSettings
from .data import TaskData
from .linear import fit_predict, make_folds, stepwise_select


def _select(X, y, s: SelectionSettings, seed):
    return stepwise_select(X, y, objective=s.objective, max_features=s.max_features,
                           folds=s.folds, repeats=s.repeats, seed=seed).selected


def _fit_predict(X, y, train, test, feats):
    return fit_predict(X.loc[train, feats].to_numpy(float), y[train], X.loc[test, feats].to_numpy(float))


def _r2_p(y, pred):
    if len(y) < 3 or np.std(pred) == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(pred, y)
    return float(r ** 2), float(p)


@dataclass
class TaskResult:
    task: str
    features: list                  # chosen on all L1 patients (used for L2)
    predictions: pd.DataFrame       # id, group, actual, predicted, error (= predicted - actual)
    summary: dict
    fold_features: list = field(default_factory=list)


def analyse_task(td: TaskData, s: SelectionSettings, outer_folds=10, seed=0) -> TaskResult:
    X, y = td.X, td.y
    l1 = np.flatnonzero(td.l1)
    l2 = np.flatnonzero(~td.l1)
    if len(l1) < 2 * outer_folds or len(l2) < 3:
        raise ValueError(f"{td.task}: too few patients (L1={len(l1)}, L2={len(l2)})")

    # L1: nested CV - select features inside each outer training fold
    pred = np.full(len(y), np.nan)
    fold_features = []
    folds = make_folds(len(l1), outer_folds, 1, seed)[0]
    for f in range(outer_folds):
        train, test = l1[folds != f], l1[folds == f]
        feats = _select(X.iloc[train].reset_index(drop=True), y[train], s, seed + 1 + f)
        fold_features.append(feats)
        pred[test] = _fit_predict(X, y, train, test, feats)

    # L2: select and fit on all L1 patients
    feats = _select(X.iloc[l1].reset_index(drop=True), y[l1], s, seed)
    pred[l2] = _fit_predict(X, y, l1, l2, feats)

    err = pred - y
    e1, e2 = err[l1], err[l2]
    r2_1, p_1 = _r2_p(y[l1], pred[l1])
    r2_2, p_2 = _r2_p(y[l2], pred[l2])
    bias = stats.ttest_ind(e1, e2, equal_var=False)
    acc = stats.ttest_ind(np.abs(e1), np.abs(e2), equal_var=False)
    mw = stats.mannwhitneyu(np.abs(e1), np.abs(e2))
    summary = {
        "task": td.task, "n_l1": len(l1), "n_l2": len(l2), "n_features": len(feats),
        "l1_r2": r2_1, "l1_p": p_1, "l2_r2": r2_2, "l2_p": p_2,
        "l1_mean_error": float(e1.mean()), "l2_mean_error": float(e2.mean()),
        "l1_mae": float(np.abs(e1).mean()), "l2_mae": float(np.abs(e2).mean()),
        "bias_t": float(bias.statistic), "bias_p": float(bias.pvalue),
        "mae_t": float(acc.statistic), "mae_p": float(acc.pvalue), "mae_mannwhitney_p": float(mw.pvalue),
    }
    table = pd.DataFrame({"id": td.ids, "group": np.where(td.l1, "L1", "L2"),
                          "actual": y, "predicted": pred, "error": err})
    return TaskResult(td.task, feats, table, summary, fold_features)


def resample_comparison(td: TaskData, s: SelectionSettings, n_resamples=100, reselect=True,
                        features=None, seed=0) -> pd.DataFrame:
    """Hold out a random set of L1 patients the same size as the L2 group,
    train on the remaining L1 patients, and compare errors for the held-out
    L1 patients and the L2 patients (replaces PermutePreds).

    With ``reselect`` (default) features are chosen on the training L1 patients
    only. Otherwise ``features`` (e.g. from analyse_task) are used, which is
    faster but lets the held-out L1 patients influence the model.
    """
    if not reselect and not features:
        raise ValueError("give features, or use reselect=True")
    X, y = td.X, td.y
    l1 = np.flatnonzero(td.l1)
    l2 = np.flatnonzero(~td.l1)
    m = min(len(l2), len(l1) // 2)  # sampled without replacement (PermutePreds used randi, with replacement)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_resamples):
        hold = rng.choice(l1, size=m, replace=False)
        train = np.setdiff1d(l1, hold)
        feats = _select(X.iloc[train].reset_index(drop=True), y[train], s, seed + r) if reselect else features
        e_hold = _fit_predict(X, y, train, hold, feats) - y[hold]
        e_l2 = _fit_predict(X, y, train, l2, feats) - y[l2]
        rows.append({
            "resample": r, "n_features": len(feats),
            "l1_mae": np.abs(e_hold).mean(), "l2_mae": np.abs(e_l2).mean(),
            "bias_t": stats.ttest_ind(e_hold, e_l2, equal_var=False).statistic,
            "mae_t": stats.ttest_ind(np.abs(e_hold), np.abs(e_l2), equal_var=False).statistic,
        })
    return pd.DataFrame(rows)


def summarise_resamples(df: pd.DataFrame) -> dict:
    diff = df["l2_mae"] - df["l1_mae"]
    return {"resamples": len(df), "mean_l1_mae": float(df["l1_mae"].mean()),
            "mean_l2_mae": float(df["l2_mae"].mean()), "mean_mae_difference": float(diff.mean()),
            "share_l2_worse": float((diff > 0).mean()),
            "mae_difference_95ci": [float(diff.quantile(0.025)), float(diff.quantile(0.975))]}


def history_regression(predictions: pd.DataFrame, history: pd.DataFrame, min_n=10) -> pd.DataFrame:
    """Regress each L2 patient's prediction error on each language-history
    measure (Analysis 4). Patients are matched by ID; blank or negative values
    are treated as missing, as in the original.
    """
    errors = predictions[predictions["group"] == "L2"].set_index("id")["error"]
    hist = history.copy()
    hist.index = hist.iloc[:, 0].astype(str).str.strip()
    hist = hist.iloc[:, 1:]
    rows = []
    for measure in hist.columns:
        x = pd.to_numeric(hist[measure], errors="coerce")
        joined = pd.concat([errors, x.rename("x")], axis=1, join="inner").dropna()
        joined = joined[joined["x"] >= 0]
        n = len(joined)
        if n >= min_n and joined["x"].nunique() > 1:
            res = stats.linregress(joined["x"], joined["error"])
            rows.append({"measure": measure, "n": n, "slope": res.slope, "r2": res.rvalue ** 2, "p": res.pvalue})
        else:
            rows.append({"measure": measure, "n": n, "slope": np.nan, "r2": np.nan, "p": np.nan})
    return pd.DataFrame(rows)


def plot_errors(predictions: pd.DataFrame, path, score_range=None, bins=np.arange(-20, 21)):
    """Predicted vs actual and error histograms for L1 and L2 (as PlotErrs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = predictions.copy()
    if score_range:
        df["predicted"] = df["predicted"].clip(*score_range)
        df["error"] = df["predicted"] - df["actual"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for col, (group, label) in enumerate([("L1", "L1 group"), ("L2", "L2 group")]):
        g = df[df["group"] == group]
        ax = axes[0, col]
        ax.scatter(g["actual"], g["predicted"], s=8, alpha=0.6)
        lo, hi = df[["actual", "predicted"]].min().min(), df[["actual", "predicted"]].max().max()
        ax.plot([lo, hi], [lo, hi], color="#c53030", lw=1.5)
        ax.set(xlabel="Actual score", ylabel="Predicted score", title=label)
        ax = axes[1, col]
        ax.hist(g["error"], bins=bins, color="#2b6cb0")
        ax.set(xlabel="Predicted minus actual score", ylabel="Frequency", title=label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
