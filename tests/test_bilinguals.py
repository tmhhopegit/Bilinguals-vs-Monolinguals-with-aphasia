import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bilinguals import bayes, prediction, vlsm
from bilinguals.__main__ import main
from bilinguals.config import Config, Selection
from bilinguals.data import first_per_patient, load_patients, task_data
from bilinguals.linear import cv_predict, fit_predict, loo_predict, make_folds, stepwise_select

from synthetic import EFFECT_VOXELS, make_study

TMP = Path(tempfile.mkdtemp())
make_study(TMP)
CONFIG = TMP / "config.toml"
CONFIG.write_text("""
patients = "patients.csv"
tasks = ["Naming"]
require_complete = ["Cognitive A"]
language_history = "history.csv"
lesions_npz = "lesions.npz"
results_dir = "results"
[selection]
max_features = 6
repeats = 2
[vlsm]
min_patients = 5
permutations = 200
alpha = 0.05
""")
CFG = Config.load(CONFIG)
TABLE = load_patients(CFG)
TD = task_data(TABLE, CFG, "Naming")
FAST = Selection(max_features=6, repeats=2)


class DataTests(unittest.TestCase):
    def test_eligibility_and_first_scans(self):
        ids = set(TD.ids)
        self.assertFalse({"RIGHT", "EARLY", "NOCOG"} & ids)
        self.assertEqual(len(ids), len(TD.ids))          # one row per patient
        self.assertEqual((TD.n_l1, TD.n_l2), (110, 50))

    def test_first_scan_is_first_in_file_even_if_not_adjacent(self):
        ids = np.array(["a", "b", "a", "c", "b"])
        np.testing.assert_array_equal(first_per_patient(ids, np.ones(5, bool)), [1, 1, 0, 1, 0])
        np.testing.assert_array_equal(first_per_patient(ids, np.array([0, 1, 1, 1, 1], bool)), [0, 1, 1, 1, 0])

    def test_missing_column_is_reported(self):
        cfg = Config.load(CONFIG)
        cfg.tasks = ["Nonexistent"]
        with self.assertRaises(KeyError):
            load_patients(cfg)


class LinearTests(unittest.TestCase):
    def test_ols(self):
        X = np.random.default_rng(0).normal(size=(30, 2))
        y = 1 + 2 * X[:, 0] - X[:, 1]
        np.testing.assert_allclose(fit_predict(X, y, X), y)

    def test_folds_fixed_and_balanced(self):
        f = make_folds(53, 10, 3, rng=1)
        np.testing.assert_array_equal(f, make_folds(53, 10, 3, rng=1))
        for row in f:
            counts = np.bincount(row)
            self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_loo(self):
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(20, 2)), rng.normal(size=20)
        manual = [fit_predict(np.delete(X, i, 0), np.delete(y, i), X[i:i + 1])[0] for i in range(20)]
        np.testing.assert_allclose(loo_predict(X, y), manual)
        # repeated CV averages every repeat (CV10x10 overwrote column 1 each time)
        folds = make_folds(20, 5, 3, rng=2)
        avg = np.mean([cv_predict(X, y, folds[r:r + 1]) for r in range(3)], axis=0)
        np.testing.assert_allclose(cv_predict(X, y, folds), avg)

    def test_stepwise_finds_the_informative_loads(self):
        l1 = TD.l1
        sel = stepwise_select(TD.X[l1].reset_index(drop=True), TD.y[l1], max_features=6, repeats=2)
        self.assertEqual(sel.selected[0], "time_post")    # protected
        self.assertIn("LL_0", sel.selected)
        self.assertIn("LL_1", sel.selected)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = prediction.analyse_task(TD, FAST, outer_folds=5)

    def test_l2_patients_are_predicted_worse(self):
        s = self.res.summary
        self.assertGreater(s["l1_r2"], 0.5)
        self.assertGreater(s["l2_mae"], s["l1_mae"])
        self.assertLess(s["mae_p"], 0.01)

    def test_errors_use_one_sign_convention(self):
        p = self.res.predictions
        np.testing.assert_allclose(p["error"], p["predicted"] - p["actual"])
        # L2 patients score below prediction, so their mean error is positive
        self.assertGreater(p.loc[p["group"] == "L2", "error"].mean(), 3)

    def test_nested_cv_does_not_flatter_noise(self):
        rng = np.random.default_rng(3)
        noise = TD.X.copy()
        noise.iloc[:, 1:] = rng.normal(size=(len(noise), noise.shape[1] - 1))
        fake = type(TD)(TD.task, noise, rng.normal(size=len(TD.y)), TD.l1, TD.ids)
        res = prediction.analyse_task(fake, FAST, outer_folds=5)
        self.assertLess(res.summary["l1_r2"], 0.1)

    def test_history_regression_matches_by_id(self):
        hist = pd.read_csv(TMP / "history.csv")
        res = prediction.history_regression(self.res.predictions, hist).set_index("measure")
        self.assertLess(res.loc["years_since_l2_acquired", "p"], 0.01)
        self.assertGreater(res.loc["years_since_l2_acquired", "slope"], 0)
        self.assertGreater(res.loc["unrelated_measure", "p"], 0.01)
        self.assertEqual(res.loc["mostly_missing", "n"], 0)

    def test_resampling(self):
        df = prediction.resample_comparison(TD, FAST, n_resamples=5)
        s = prediction.summarise_resamples(df)
        self.assertEqual(s["resamples"], 5)
        self.assertGreater(s["share_l2_worse"], 0.5)
        with self.assertRaises(ValueError):
            prediction.resample_comparison(TD, FAST, reselect=False)


class VLSMTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_statistics_match_scipy(self):
        L = self.rng.random((60, 50)) < 0.3
        y = np.round(self.rng.normal(50, 10, 60))
        pats, _ = vlsm._patterns(L, np.ones(50, bool))
        h, p = vlsm._kw(pats, y)
        t, pt = vlsm._t(pats, y)
        for i in range(len(pats)):
            k = stats.kruskal(y[pats[i]], y[~pats[i]])
            tt = stats.ttest_ind(y[pats[i]], y[~pats[i]])
            self.assertAlmostEqual(abs(h[i]), k.statistic)
            self.assertAlmostEqual(p[i], k.pvalue)
            self.assertAlmostEqual(t[i], tt.statistic)

    def test_interaction_t_matches_regression(self):
        n = 70
        L = self.rng.random((n, 30)) < 0.4
        y = self.rng.normal(size=n)
        l1 = self.rng.random(n) < 0.5
        m = vlsm.interaction_map(L, y, l1, permutations=0)
        v = 0
        lv, g = L[:, v].astype(float), (~l1).astype(float)
        X = np.column_stack([np.ones(n), lv, g, lv * g])
        b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        se = np.sqrt(res[0] / (n - 4) * np.linalg.inv(X.T @ X)[3, 3])
        self.assertAlmostEqual(m.stat[v], b[3] / se)

    def test_permutations_really_permute(self):
        L = self.rng.random((60, 80)) < 0.3
        y = self.rng.normal(size=60)
        m = vlsm.group_map(L, y, permutations=200, alpha=0.05, tail="both")
        self.assertLess(m.threshold, np.nanmin(m.p) + 0.2)
        self.assertNotAlmostEqual(m.threshold, np.nanmin(m.p))
        self.assertLessEqual(m.significant.sum(), 5)   # null data: few or no hits

    def test_finds_planted_effect(self):
        lesions, _ = vlsm.load_lesions(CFG, TD.ids)
        keep = vlsm.relevant_voxels(lesions, TD.l1, 5)
        m = vlsm.group_map(lesions[TD.l1], TD.y[TD.l1], permutations=100, alpha=0.05, keep=keep)
        hits = np.flatnonzero(m.significant)
        self.assertTrue(len(hits) > 0)
        self.assertTrue(np.all(hits < EFFECT_VOXELS.stop))


class BayesTests(unittest.TestCase):
    def test_matches_monte_carlo(self):
        for r, n in [(0.0, 20), (0.5, 50), (0.9, 10)]:
            g = stats.invgamma(0.5, scale=n / 2).rvs(1_000_000, random_state=1)
            mc = np.mean(np.exp((n - 2) / 2 * np.log1p(g) - (n - 1) / 2 * np.log1p((1 - r * r) * g)))
            self.assertAlmostEqual(bayes.jzs_bf10(r, n) / mc, 1.0, places=2)
        self.assertLess(bayes.jzs_bf10(0.0, 100), bayes.jzs_bf10(0.0, 20))

    def test_table(self):
        res = bayes.correlation_table(TD.X[["LL_0", "LL_5"]], pd.DataFrame({"Naming": TD.y}), TD.l1)
        big = res.set_index("region").loc["LL_0"]
        self.assertGreater(big["l1_bf10"], 100)


class CLITests(unittest.TestCase):
    def test_all_commands(self):
        for cmd in (["predict", "--outer-folds", "5"], ["resample", "--n", "3"], ["history"], ["vlsm"],
                    ["bayes", "--min-lesioned", "5"]):
            main([cmd[0], "--config", str(CONFIG), *cmd[1:]])
        out = TMP / "results"
        for f in ("prediction_summary.csv", "predictions.csv", "prediction_errors.png",
                  "resample_summary.json", "language_history.csv", "vlsm_summary.csv",
                  "bayes_correlations.csv", "vlsm_Naming_l1_p.npy"):
            self.assertTrue((out / f).exists(), f)


if __name__ == "__main__":
    unittest.main()
