"""Command line: ``python -m bilinguals <command> --config my.toml``

Commands
  predict   train on L1 patients, predict L1 (nested CV) and L2; compare errors
  resample  L1 hold-out vs L2 comparison over many random splits
  history   regress L2 prediction errors on language-history measures
  vlsm      lesion-symptom maps for L1, L2 and their interaction
  bayes     Bayes factors for lesion-load / score correlations per group
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import bayes, prediction, vlsm
from .config import Config
from .data import lesion_load_columns, load_patients, read_table, task_data


def _setup(args):
    cfg = Config.load(args.config)
    table = load_patients(cfg)
    tasks = args.tasks or cfg.tasks
    if not tasks:
        sys.exit("no tasks: set 'tasks' in the config or pass --tasks")
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    return cfg, table, tasks


def _save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"saved {path}", file=sys.stderr)


def cmd_predict(args):
    cfg, table, tasks = _setup(args)
    summaries, preds, features = [], [], {}
    for task in tasks:
        print(f"{task} ...", file=sys.stderr)
        res = prediction.analyse_task(task_data(table, cfg, task), cfg.selection,
                                      outer_folds=args.outer_folds, seed=cfg.seed)
        summaries.append(res.summary)
        preds.append(res.predictions.assign(task=task))
        features[task] = {"selected_on_all_l1": res.features, "per_fold": res.fold_features}
    summary = pd.DataFrame(summaries)
    summary.to_csv(cfg.results_dir / "prediction_summary.csv", index=False)
    allpreds = pd.concat(preds, ignore_index=True)
    allpreds.to_csv(cfg.results_dir / "predictions.csv", index=False)
    _save_json(features, cfg.results_dir / "selected_features.json")
    prediction.plot_errors(allpreds, cfg.results_dir / "prediction_errors.png", cfg.score_range)
    cols = ["task", "n_l1", "n_l2", "l1_r2", "l2_r2", "l1_mae", "l2_mae", "mae_p", "bias_p"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def cmd_resample(args):
    cfg, table, tasks = _setup(args)
    out = {}
    for task in tasks:
        print(f"{task} ...", file=sys.stderr)
        df = prediction.resample_comparison(task_data(table, cfg, task), cfg.selection,
                                            n_resamples=args.n, seed=cfg.seed)
        df.to_csv(cfg.results_dir / f"resamples_{task}.csv", index=False)
        out[task] = prediction.summarise_resamples(df)
    _save_json(out, cfg.results_dir / "resample_summary.json")
    print(json.dumps(out, indent=2, default=float))


def cmd_history(args):
    cfg, _, tasks = _setup(args)
    if cfg.language_history is None:
        sys.exit("config: 'language_history' is not set")
    preds_path = cfg.results_dir / "predictions.csv"
    if not preds_path.exists():
        sys.exit("run 'predict' first (it writes predictions.csv)")
    preds = pd.read_csv(preds_path, dtype={"id": str})
    hist = read_table(cfg.language_history)
    frames = [prediction.history_regression(preds[preds["task"] == t], hist).assign(task=t) for t in tasks]
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(cfg.results_dir / "language_history.csv", index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


def cmd_vlsm(args):
    cfg, table, tasks = _setup(args)
    v = cfg.vlsm
    rows = []
    for t_i, task in enumerate(tasks):
        td = task_data(table, cfg, task)
        lesions, space = vlsm.load_lesions(cfg, td.ids, table)
        keep = vlsm.relevant_voxels(lesions, td.l1, v.min_patients)
        print(f"{task}: {keep.sum()} voxels tested", file=sys.stderr)
        seed = cfg.seed + 100 * t_i
        maps = {
            "l1": vlsm.group_map(lesions[td.l1], td.y[td.l1], v.statistic, v.permutations, v.alpha,
                                 keep=keep, rng=seed),
            "l2": vlsm.group_map(lesions[~td.l1], td.y[~td.l1], v.statistic, v.permutations, v.alpha,
                                 keep=keep, rng=seed + 1),
            "interaction": vlsm.interaction_map(lesions, td.y, td.l1, v.permutations, v.alpha,
                                                keep=keep, rng=seed + 2),
        }
        for name, m in maps.items():
            vlsm.save_map(m.stat, space, cfg.results_dir / f"vlsm_{task}_{name}_stat")
            vlsm.save_map(m.p, space, cfg.results_dir / f"vlsm_{task}_{name}_p")
        rows.append({"task": task, "voxels_tested": int(keep.sum()),
                     **{f"{k}_threshold": m.threshold for k, m in maps.items()},
                     **vlsm.compare_maps(maps["l1"], maps["l2"], maps["interaction"])})
    res = pd.DataFrame(rows)
    res.to_csv(cfg.results_dir / "vlsm_summary.csv", index=False)
    print(res.to_string(index=False))


def cmd_bayes(args):
    cfg, table, tasks = _setup(args)
    loads = lesion_load_columns(table, cfg)
    frames = []
    for task in tasks:
        td = task_data(table, cfg, task)
        frames.append(bayes.correlation_table(td.X[loads], pd.DataFrame({task: td.y}), td.l1,
                                              args.min_lesioned))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(cfg.results_dir / "bayes_correlations.csv", index=False)
    print(f"{len(res)} region x task pairs -> {cfg.results_dir / 'bayes_correlations.csv'}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bilinguals", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, func, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(func=func)
        p.add_argument("--config", required=True)
        p.add_argument("--tasks", nargs="+", help="task columns (default: 'tasks' in the config)")
        return p

    p = add("predict", cmd_predict, "L1-trained model: L1 vs L2 accuracy")
    p.add_argument("--outer-folds", type=int, default=10)
    p = add("resample", cmd_resample, "repeated L1 hold-out vs L2 comparison")
    p.add_argument("--n", type=int, default=100, help="number of random splits")
    add("history", cmd_history, "L2 errors vs language history")
    add("vlsm", cmd_vlsm, "lesion-symptom maps")
    p = add("bayes", cmd_bayes, "Bayes factors for lesion-load correlations")
    p.add_argument("--min-lesioned", type=int, default=10)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
