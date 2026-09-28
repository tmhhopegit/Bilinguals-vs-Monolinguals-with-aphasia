"""Made-up patients for tests. No real patient data."""
from __future__ import annotations

import numpy as np
import pandas as pd

N_LOADS = 12
N_VOXELS = 400
EFFECT_VOXELS = slice(0, 40)   # lesioning these lowers scores


def make_study(folder, n_l1=110, n_l2=50, seed=0, l2_penalty=6.0):
    """Write patients.csv, history.csv and lesions.npz; return a config dict.

    Scores depend on three lesion loads (LL_0, LL_1, LL_2 = the effect voxels) and time post-stroke.
    L2 patients score lower than predicted by ``l2_penalty`` points on average,
    and that shortfall grows with 'years_since_l2_acquired'.
    """
    rng = np.random.default_rng(seed)
    rows, lesion_rows, hist_rows = [], [], []
    for i in range(n_l1 + n_l2):
        pid = f"S{i:03d}"
        l1 = i < n_l1
        loads = rng.beta(0.7, 3, N_LOADS)
        lesion = rng.random(N_VOXELS) < 0.25
        lesion[EFFECT_VOXELS] = rng.random() < 0.4
        loads[2] = lesion[EFFECT_VOXELS].mean()   # the region's lesion load reflects those voxels
        years = rng.uniform(0, 40)
        base = 60 - 30 * loads[0] - 15 * loads[1] - 8 * lesion[0]
        for scan in range(2 if i % 7 == 0 else 1):
            t = rng.uniform(4, 60) + 12 * scan
            score = base + 3 * np.log(t) + rng.normal(0, 2)
            if not l1:
                score -= l2_penalty * years / 20   # averages l2_penalty over 0-40 years
            rows.append({"ID": pid, "English first language": "English" if l1 else "Other",
                         "Time post stroke": t, "Lesion volume": 1000 + loads.sum() * 1e4,
                         "Right hemisphere volume": 0.0, "Cognitive A": rng.normal(),
                         "Naming": score, **{f"LL_{k}": loads[k] for k in range(N_LOADS)}})
        lesion_rows.append((pid, lesion))
        if not l1:
            hist_rows.append({"ID": pid, "years_since_l2_acquired": years,
                              "unrelated_measure": rng.uniform(0, 10), "mostly_missing": -1})
    # rows the eligibility rules must drop
    rows.append({**rows[0], "ID": "RIGHT", "Right hemisphere volume": 50.0})
    rows.append({**rows[0], "ID": "EARLY", "Time post stroke": 2.0})
    rows.append({**rows[0], "ID": "NOCOG", "Cognitive A": np.nan})
    df = pd.DataFrame(rows)
    # move one repeat scan away from its patient's first scan (non-adjacent repeat)
    df = pd.concat([df.iloc[[1]], df.drop(index=1)], ignore_index=True)
    df.to_csv(folder / "patients.csv", index=False)
    hist = pd.DataFrame(hist_rows).sample(frac=1, random_state=0)  # history in a different order
    hist.to_csv(folder / "history.csv", index=False)
    np.savez(folder / "lesions.npz", ids=np.array([p for p, _ in lesion_rows]),
             lesions=np.array([l for _, l in lesion_rows]))
    return folder
