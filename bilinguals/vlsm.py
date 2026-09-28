"""Voxel-based lesion-symptom mapping in L1 and L2 patients.

Replaces VLSM_BilMon, MakeTMapFromData, MakeMapsFromData, MakeDMapFromData
and the (commented-out) Analysis 3 of RebootBilAnalysis.

* Voxels are only tested where at least ``min_patients`` patients in *each*
  group are lesioned and at least that many are spared.
* Voxels with identical lesion patterns are tested once.
* Family-wise error is controlled by permutation: the threshold is the
  alpha-quantile of the smallest p-value per permutation. (VLSM_BilMon never
  actually permuted, so its threshold was the observed minimum p.)
* The L1-vs-L2 interaction map uses Freedman-Lane permutation of the
  residuals of the lesion + group model, which tests the interaction itself.
  (MakeDMapFromData shuffled the scores across everyone, which tests "any
  effect".)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Map:
    stat: np.ndarray        # per voxel; NaN where not tested
    p: np.ndarray
    threshold: float        # FWE-corrected p threshold (NaN if no permutations)
    tested: int             # voxels tested

    @property
    def significant(self):
        return np.nan_to_num(self.p, nan=1.0) < self.threshold


def relevant_voxels(lesions, l1, min_patients) -> np.ndarray:
    lesions = np.asarray(lesions, bool)
    keep = np.ones(lesions.shape[1], bool)
    for g in (l1, ~l1):
        n_les = lesions[g].sum(axis=0)
        keep &= (n_les >= min_patients) & (n_les <= g.sum() - min_patients)
    return keep


def _patterns(lesions, keep):
    pats, inverse = np.unique(lesions[:, keep].T, axis=0, return_inverse=True)
    return pats.astype(bool), inverse.ravel()


def _expand(values, inverse, keep):
    out = np.full(keep.size, np.nan)
    out[np.flatnonzero(keep)] = values[inverse]
    return out


# ------------------------------------------------------------ two-group tests
def _kw(pats, y):
    """Kruskal-Wallis H (2 groups) per pattern, signed negative when lesioned
    patients rank lower. Matches scipy.stats.kruskal, including tie correction."""
    n = y.size
    ranks = stats.rankdata(y)
    n1 = pats.sum(axis=1)
    r1 = pats @ ranks
    r2 = ranks.sum() - r1
    n2 = n - n1
    h = 12.0 / (n * (n + 1)) * (r1 ** 2 / n1 + r2 ** 2 / n2) - 3 * (n + 1)
    _, counts = np.unique(y, return_counts=True)
    h = h / (1 - np.sum(counts ** 3 - counts) / (n ** 3 - n))
    p = stats.chi2.sf(h, 1)
    sign = np.where(r1 / n1 < r2 / n2, -1.0, 1.0)
    return sign * h, p


def _t(pats, y):
    """Pooled-variance t (lesioned minus spared) per pattern, as ttest2."""
    n1 = pats.sum(axis=1)
    n2 = y.size - n1
    s1, s2 = pats @ y, (~pats) @ y
    q1, q2 = pats @ y ** 2, (~pats) @ y ** 2
    m1, m2 = s1 / n1, s2 / n2
    ss = (q1 - n1 * m1 ** 2) + (q2 - n2 * m2 ** 2)
    df = y.size - 2
    t = (m1 - m2) / np.sqrt(ss / df * (1 / n1 + 1 / n2))
    return t, 2 * stats.t.sf(np.abs(t), df)


def _tail(stat, p, tail):
    if tail == "deficit":  # only patterns where lesioned patients do worse count
        return np.where(stat < 0, p, np.nan)
    return p


def group_map(lesions, y, statistic="kw", permutations=1000, alpha=0.10, tail="deficit",
              keep=None, rng=0) -> Map:
    """Lesion-deficit map for one group of patients."""
    lesions = np.asarray(lesions, bool)
    y = np.asarray(y, float)
    if keep is None:
        keep = np.ones(lesions.shape[1], bool)
    pats, inverse = _patterns(lesions, keep)
    test = _kw if statistic == "kw" else _t
    stat, p = test(pats, y)
    p = _tail(stat, p, tail)
    threshold = np.nan
    if permutations:
        rng = np.random.default_rng(rng)
        mins = np.array([np.nanmin(np.append(_tail(*test(pats, rng.permutation(y)), tail), 1.0))
                         for _ in range(permutations)])
        threshold = float(np.quantile(mins, alpha))
    return Map(_expand(stat, inverse, keep), _expand(p, inverse, keep), threshold, int(keep.sum()))


# ------------------------------------------------------------- interaction
def _interaction_t(X, pinv, cvar, y):
    """t for the lesion x group coefficient; y is (P, n)."""
    beta = np.einsum("pkn,pn->pk", pinv, y)
    resid = y - np.einsum("pnk,pk->pn", X, beta)
    df = X.shape[1] - X.shape[2]
    sigma2 = np.sum(resid ** 2, axis=1) / df
    return beta[:, 3] / np.sqrt(sigma2 * cvar), df


def interaction_map(lesions, y, l1, permutations=1000, alpha=0.10, keep=None, rng=0) -> Map:
    """Does the lesion effect differ between L1 and L2 patients?
    Model per voxel: score ~ lesion + group + lesion:group."""
    lesions = np.asarray(lesions, bool)
    y = np.asarray(y, float)
    g = (~np.asarray(l1, bool)).astype(float)  # 1 = L2
    if keep is None:
        keep = np.ones(lesions.shape[1], bool)
    pats, inverse = _patterns(lesions, keep)
    L = pats.astype(float)
    ones = np.ones_like(L)
    G = np.broadcast_to(g, L.shape)
    X = np.stack([ones, L, G, L * G], axis=2)          # (P, n, 4)
    Xr = X[:, :, :3]                                   # reduced model
    pinv = np.linalg.pinv(X)                           # (P, 4, n)
    cvar = np.linalg.inv(np.einsum("pnk,pnj->pkj", X, X))[:, 3, 3]
    Y = np.broadcast_to(y, L.shape)
    t, df = _interaction_t(X, pinv, cvar, Y)
    p = 2 * stats.t.sf(np.abs(t), df)
    threshold = np.nan
    if permutations:
        rng = np.random.default_rng(rng)
        beta_r = np.einsum("pkn,pn->pk", np.linalg.pinv(Xr), Y)
        fitted = np.einsum("pnk,pk->pn", Xr, beta_r)
        resid = Y - fitted
        mins = np.empty(permutations)
        for i in range(permutations):
            perm = rng.permutation(y.size)
            tp, _ = _interaction_t(X, pinv, cvar, fitted + resid[:, perm])
            mins[i] = np.min(2 * stats.t.sf(np.abs(tp), df))
        threshold = float(np.quantile(mins, alpha))
    return Map(_expand(t, inverse, keep), _expand(p, inverse, keep), threshold, int(keep.sum()))


def compare_maps(l1_map: Map, l2_map: Map, diff_map: Map) -> dict:
    """Voxel counts as in Table 3 of RebootBilAnalysis."""
    b, m, d = l2_map.significant, l1_map.significant, diff_map.significant
    return {"l2_voxels": int(b.sum()), "l1_voxels": int(m.sum()), "interaction_voxels": int(d.sum()),
            "shared_and_interaction": int((m & b & d).sum()),
            "l2_only_and_interaction": int((b & ~m & d).sum())}


def load_lesions(cfg, ids, table=None):
    """Binary lesion matrix (n, voxels) in the order of ``ids``.

    From ``lesions_npz`` (arrays 'ids' and 'lesions'), or from per-scan image
    paths in ``lesion_image_column`` restricted to ``brain_mask`` (needs nibabel).
    """
    if cfg.lesions_npz is not None:
        data = np.load(cfg.lesions_npz, allow_pickle=False)
        index = {str(i): k for k, i in enumerate(data["ids"])}
        missing = [i for i in ids if str(i) not in index]
        if missing:
            raise KeyError(f"no lesion data for {len(missing)} patients, e.g. {missing[:3]}")
        return data["lesions"][[index[str(i)] for i in ids]].astype(bool), None
    if cfg.lesion_image_column and table is not None:
        import nibabel as nib
        mask_img = nib.load(str(cfg.brain_mask))
        mask = mask_img.get_fdata() > 0
        paths = table.drop_duplicates("id").set_index("id")[cfg.lesion_image_column]
        lesions = np.stack([nib.load(str(paths[i])).get_fdata()[mask] > 0 for i in ids])
        return lesions, (mask_img, mask)
    raise ValueError("config: set lesions_npz, or lesion_image_column and brain_mask")


def save_map(values, space, path):
    """Write a map as NIfTI when the lesions came from images, else .npy."""
    if space is None:
        np.save(str(path) + ".npy", values)
        return str(path) + ".npy"
    import nibabel as nib
    img, mask = space
    vol = np.full(mask.shape, np.nan, dtype=np.float32)
    vol[mask] = values
    nib.save(nib.Nifti1Image(vol, img.affine), str(path) + ".nii.gz")
    return str(path) + ".nii.gz"
