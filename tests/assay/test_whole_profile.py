"""The whole-profile score comparison (``reproduce/assay/whole_profile.py``) on small hand-made tables; no study data.

    python -m pytest tests/assay/test_whole_profile.py
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reproduce.assay import scores, whole_profile

SAVED = ["fit", "model", "seed", "level", "total.mse", "systema.pearson_ref_train", "mz.r_gamma", "dbd.r2w_delta",
         "systema.centroid_accuracy"]


def fit_rows(fold, model, seed, level, value):
    """A kept fit's 25 rows: recovery ``value`` on every non-null planted state."""
    return [{"screen": "Tahoe-100M", "fold": fold, "model": model, "seed": seed, "level": level,
             "learning_rate": "standard", "fit_status": "kept", **scores.describe(a),
             "recovery": None if a % 5 == 4 else value * (1 + a % 5)} for a in range(25)]


class ScoresTable(unittest.TestCase):
    def test_saved_scores_join_the_recovery_sums_of_the_five_shared_and_five_selective_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, model in (("BUILT_MODELS", "state"), ("PUBLISHED_MODELS", "state_pub")):
                path = Path(tmp) / scores.SCORES_FOLDER / "tahoe-100m/F1" / f"WHOLE_PROFILE_SCORES_{name}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(SAVED)
                    writer.writerow([f"{model}_31002_L2", model, 31002, "L2", .5, .1, .2, .3, .4])
                    writer.writerow([f"{model}_31002_L0", model, 31002, "L0", 9, 9, 9, 9, 9])
            fits = pd.DataFrame(fit_rows("F1", "attention model", 31002, "+Var", .1)
                                + fit_rows("F1", "State", 31002, "+Var", .2))
            table = whole_profile.scores_table(tmp, fits)
        got = table.set_index("model")
        self.assertEqual(sorted(got.index), ["State", "attention model"])
        self.assertEqual(got.loc["attention model", "total_mse"], .5)
        self.assertEqual(got.loc["attention model", "centroid_accuracy"], .4)
        self.assertAlmostEqual(got.loc["attention model", "shared_recovery"], 5 * .1)
        self.assertAlmostEqual(got.loc["State", "selective_recovery"], 5 * .2 * 2)


class Correlations(unittest.TestCase):
    def test_spearman_with_average_ranks_within_each_fold_then_the_fold_mean(self):
        rng = np.random.default_rng(3)
        rows = []
        for fold in ("F1", "F2"):
            for i in range(12):
                values = rng.normal(size=7)
                rows.append({"fold": fold, "model": "CPA" if i % 2 else "State",
                             **dict(zip(whole_profile.PROFILE_COLUMNS, values))})
        table = pd.DataFrame(rows)
        table.loc[table.index[:3], "centroid_accuracy"] = 1.          # ties
        got = whole_profile.correlations(table).set_index(["models", "score_row", "score_column"])

        def expected(models, row, column):
            per_fold = []
            for _, g in table[table.model.isin(models)].groupby("fold"):
                x = -g[row] if row == "total_mse" else g[row]
                y = -g[column] if column == "total_mse" else g[column]
                per_fold.append(spearmanr(x, y).statistic)
            return np.mean(per_fold)

        labels = dict(zip(whole_profile.PROFILE_COLUMNS, whole_profile.LABELS))
        for models, name in ((("CPA", "State"), whole_profile.ALL), (("CPA",), "CPA")):
            for row, column in (("centroid_accuracy", "total_mse"), ("selective_recovery", "pearson_delta")):
                value = got.loc[(name, labels[row], labels[column]), "spearman"]
                self.assertAlmostEqual(value, expected(models, row, column), places=12)
        self.assertEqual(len(got.loc[whole_profile.ALL]), 21)       # each pair once


if __name__ == "__main__":
    unittest.main()
