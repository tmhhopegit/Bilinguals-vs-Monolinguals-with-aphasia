"""Settings from a TOML file (see config.example.toml).

The MATLAB code took its data from a ``cls`` object that isn't part of this
project, and picked tasks and predictors by column number (e.g. demographic
column 5 = first language, behaviour columns [3 8 9 10 ...]). Here every
column is named in the config file.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Selection:
    max_features: int = 40        # MaxRandoms in RebootBilAnalysis
    objective: str = "r"          # "r" (correlation, the original Mode=1) or "mae"
    folds: int = 10
    repeats: int = 10             # CV10x10


@dataclass
class VLSM:
    min_patients: int = 10        # MinSet: lesioned AND spared in each group
    permutations: int = 1000
    alpha: float = 0.10           # the original thresholds used the 10th percentile
    statistic: str = "kw"         # "kw" (Kruskal-Wallis) or "t"


@dataclass
class Config:
    patients: Path | None = None           # one row per scan: .csv or .xlsx
    sheet: str | int = 0
    id_column: str = "ID"
    l1_column: str = "English first language"
    l1_values: list = field(default_factory=lambda: [1, "1", "yes", "english", "true"])
    time_post_column: str = "Time post stroke"
    volume_column: str = "Lesion volume"
    right_volume_column: str = "Right hemisphere volume"
    lesion_load_prefix: str = "LL_"        # lesion-load predictor columns start with this
    extra_predictors: list = field(default_factory=list)
    tasks: list = field(default_factory=list)            # language score columns to analyse
    require_complete: list = field(default_factory=list) # rows missing these are dropped (the cognitive scores)
    min_time_post: float = 3.0
    left_hemisphere_only: bool = True
    first_scan_only: bool = True
    score_range: list | None = None        # e.g. [28, 75] to clip predictions for plots, as GetAllpreds did
    language_history: Path | None = None   # table: first column = patient ID, then one column per measure
    lesions_npz: Path | None = None        # arrays 'ids' (n,) and 'lesions' (n, voxels) for VLSM
    lesion_image_column: str | None = None # alternatively: per-scan binary lesion image paths
    brain_mask: Path | None = None         # needed to read lesion images / write map images
    results_dir: Path = Path("results")
    seed: int = 0
    selection: Selection = field(default_factory=Selection)
    vlsm: VLSM = field(default_factory=VLSM)

    @classmethod
    def load(cls, path) -> "Config":
        path = Path(path)
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        base = path.parent

        def p(v):
            if v is None:
                return None
            q = Path(v).expanduser()
            return q if q.is_absolute() else base / q

        sel = Selection(**raw.pop("selection", {}))
        vl = VLSM(**raw.pop("vlsm", {}))
        cfg = cls(selection=sel, vlsm=vl, **{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})
        for name in ("patients", "language_history", "lesions_npz", "brain_mask", "results_dir"):
            setattr(cfg, name, p(getattr(cfg, name)))
        if cfg.selection.objective not in ("r", "mae"):
            raise ValueError("selection.objective must be 'r' or 'mae'")
        if cfg.vlsm.statistic not in ("kw", "t"):
            raise ValueError("vlsm.statistic must be 'kw' or 't'")
        return cfg
