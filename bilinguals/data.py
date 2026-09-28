"""Load the patient table and build one dataset per language task.

Replaces the ``cls`` object, ``FilterPatients`` and the data-assembly part
of RebootBilAnalysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config


def read_table(path, sheet=0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet)
    return pd.read_csv(path)


def load_patients(cfg: Config) -> pd.DataFrame:
    """Standardised patient table: 'id', 'l1' (bool, True = first language
    English), 'time_post', 'volume', 'volume_right' plus the original columns."""
    if cfg.patients is None:
        raise ValueError("config: 'patients' is not set")
    raw = read_table(cfg.patients, cfg.sheet)
    for col in [cfg.id_column, cfg.l1_column, cfg.time_post_column, cfg.volume_column,
                cfg.right_volume_column, *cfg.tasks, *cfg.require_complete, *cfg.extra_predictors]:
        if col not in raw.columns:
            raise KeyError(f"column {col!r} not found in {cfg.patients.name}")
    t = raw.copy()
    t["id"] = raw[cfg.id_column].astype(str).str.strip()
    wanted = {str(v).strip().lower() for v in cfg.l1_values}
    t["l1"] = raw[cfg.l1_column].map(lambda v: str(v).strip().lower() in wanted
                                     or (isinstance(v, (int, float)) and v == 1))
    t["time_post"] = pd.to_numeric(raw[cfg.time_post_column], errors="coerce")
    t["volume"] = pd.to_numeric(raw[cfg.volume_column], errors="coerce")
    t["volume_right"] = pd.to_numeric(raw[cfg.right_volume_column], errors="coerce")
    return t


def lesion_load_columns(table: pd.DataFrame, cfg: Config) -> list:
    cols = [c for c in table.columns if str(c).startswith(cfg.lesion_load_prefix)]
    if not cols:
        raise ValueError(f"no lesion-load columns starting with {cfg.lesion_load_prefix!r}")
    return cols


def eligible(table: pd.DataFrame, cfg: Config) -> np.ndarray:
    """BasicSelector: time post-stroke > min, a lesion, and (optionally) no
    right-hemisphere damage."""
    keep = (table["time_post"] > cfg.min_time_post) & (table["volume"] > 0)
    if cfg.left_hemisphere_only:
        keep &= table["volume_right"] == 0
    return keep.to_numpy(dtype=bool).copy()


def first_per_patient(ids, mask) -> np.ndarray:
    """Within ``mask``, keep only the first row of each patient (file order).
    The MATLAB version compared IDs from the unfiltered list, so it kept the
    wrong rows."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    seen = set()
    for i in np.flatnonzero(mask):
        if ids[i] not in seen:
            seen.add(ids[i])
            out[i] = True
    return out


@dataclass
class TaskData:
    task: str
    X: pd.DataFrame        # predictors: time post-stroke, extra predictors, lesion loads
    y: np.ndarray          # the task score
    l1: np.ndarray         # True = monolingual / English first language
    ids: np.ndarray

    @property
    def n_l1(self):
        return int(self.l1.sum())

    @property
    def n_l2(self):
        return int((~self.l1).sum())


def task_data(table: pd.DataFrame, cfg: Config, task: str) -> TaskData:
    """Rows usable for one task: eligible, complete (task, predictors and the
    require_complete columns), one scan per patient."""
    loads = lesion_load_columns(table, cfg)
    predictors = ["time_post", *cfg.extra_predictors, *loads]
    needed = predictors + list(cfg.require_complete) + [task]
    complete = table[needed].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).to_numpy()
    mask = eligible(table, cfg) & complete
    if cfg.first_scan_only:
        mask = first_per_patient(table["id"].to_numpy(), mask)
    sub = table[mask]
    X = sub[predictors].apply(pd.to_numeric).reset_index(drop=True).astype(float)
    return TaskData(task, X, pd.to_numeric(sub[task]).to_numpy(float),
                    sub["l1"].to_numpy(bool), sub["id"].to_numpy())
