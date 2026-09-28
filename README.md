# Bilinguals (Python)

This project asks whether a lesion-based model of post-stroke language scores, trained on monolingual (L1) patients, predicts bilingual patients tested in their second language (L2) equally well. It also looks at where the L2 shortfall comes from:

- **Language history:** do the L2 prediction errors relate to language-history measures?
- **Lesion-symptom maps:** are there L1 and L2 maps, and do they differ?
- **Bayes factors:** for lesion-load / score correlations in each group.

This is a Python port of a MATLAB Bilinguals project, used in Hope et al., 2015 (Brain)
> **Patient data:** keep patient tables, lesion files and `results/` out of version control. `.gitignore` excludes them. The tests use made-up patients only.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .                                       # add [images] to read NIfTI lesions: pip install -e ".[images]"
cp config.example.toml config.toml                     # then edit
python -m unittest discover -s tests                   # about 10 seconds
```

This version reads one table, with one row per scan, and `config.toml` names each column:

- patient ID
- first language
- time post-stroke
- total and right-hemisphere lesion volume
- lesion-load columns (by prefix)
- task scores
- cognitive scores that patients must have

## Commands

```bash
python -m bilinguals predict  --config config.toml     # main analysis: L1 vs L2 accuracy per task
python -m bilinguals resample --config config.toml --n 200
python -m bilinguals history  --config config.toml     # needs predict's output
python -m bilinguals vlsm     --config config.toml
python -m bilinguals bayes    --config config.toml
```

Add `--tasks Naming Repetition` to any command to run only some tasks. Outputs go to `results/`:

- `prediction_summary.csv`
- `predictions.csv`
- `prediction_errors.png`
- `resample_summary.json`
- `language_history.csv`
- `vlsm_summary.csv` and the maps
- `bayes_correlations.csv`

## What each command does

- **`predict`** (Analysis 1). For each task:
  1. Select lesion-load features by forward/backward stepwise selection. Time post-stroke is always included, and features are scored by 10×10 cross-validated correlation.
  2. Predict **L1** patients with *nested* 10-fold cross-validation, so each patient's features are chosen without them.
  3. Predict **L2** patients from a model selected and fitted on all L1 patients.
  4. Compare errors (predicted − actual) between the groups. **Bias** is a Welch t-test on signed errors. **Accuracy** is a Welch t-test and a Mann-Whitney test on absolute errors.
- **`resample`** (`PermutePreds`). Repeatedly holds out random L1 patients (as many as there are L2 patients), selects and fits on the rest, and compares the held-out L1 patients' errors with the L2 errors. It reports how often L2 errors are larger, and a 95% interval for the difference.
- **`history`** (Analysis 4, `LHget`). Regresses each L2 patient's prediction error on each language-history measure, matching patients by ID. Measures need at least 10 patients.
- **`vlsm`** (Analysis 3, `VLSM_BilMon`, `MakeTMapFromData`, `MakeMapsFromData`, `MakeDMapFromData`). Produces three maps:
  - a lesion-deficit map for L1 and one for L2, using Kruskal-Wallis or t statistics
  - an L1-vs-L2 interaction map
  - All three are FWE-corrected by permutation. Voxels are tested only where at least `min_patients` patients in *each* group are lesioned and spared.
- **`bayes`** (`BayesNullCorr`). Computes the correlation and the JZS Bayes factor for every lesion load × task in each group. This replaces the external `jzs_corbf`; its output matches a Monte Carlo evaluation of the same integral.

## Layout

| Module | Contents |
|---|---|---|
| `config.py` | Settings (TOML) | 
| `data.py` | Patient table, eligibility, first scan per patient, per-task datasets |
| `linear.py` | OLS, fixed repeated folds, stepwise selection | 
| `prediction.py` | L1/L2 analysis, resampling, language history, error plots |
| `vlsm.py` | Group and interaction maps with permutation FWE |
| `bayes.py` | JZS Bayes factor and correlation table |

