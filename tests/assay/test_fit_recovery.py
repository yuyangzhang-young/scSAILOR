"""The recovery tables (``reproduce/assay/recovery.py``) on small hand-made fits; no study data.

    python -m pytest tests/assay/test_fit_recovery.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce.assay import recovery, scores


def fit(model="attention model", seed=31002, fold="F1", level="Mean", recovery_of=None, predicted=.5, true=1.,
        learning_rate="standard", screen="Tahoe-100M"):
    """One fit's 25 rows as ``scores.fit_scores`` gives them; ``recovery_of(a)`` sets planted state a's recovery."""
    rows = []
    for a in range(25):
        E = 0. if a % 5 == 4 else true
        Q = None if E == 0 else (recovery_of(a) if recovery_of else .5)
        rows.append({"screen": screen, "fold": fold, "model": model, "seed": seed, "level": level,
                     "learning_rate": learning_rate, "planted_state": a, "cue": list(scores.CUES.values())[a // 5],
                     "response": scores.RESPONSES[a % 5], "true_energy": E,
                     "squared_error": None if Q is None else E * (1 - Q), "predicted_energy": predicted * true,
                     "recovery": Q, "projection_slope": None if Q is None else .5})
    return rows


class Collapse(unittest.TestCase):
    def test_an_attention_fit_below_one_percent_on_every_state_is_excluded(self):
        fits = pd.DataFrame(fit(predicted=.0099) + fit(seed=31003, predicted=.01) + fit(model="DeepSets model",
                                                                                          predicted=0.))
        table = recovery.fits_table(fits)
        status = table.groupby(["model", "seed"]).fit_status.first().to_dict()
        self.assertEqual(status, {("DeepSets model", 31002): "kept", ("attention model", 31002): recovery.COLLAPSED,
                                  ("attention model", 31003): "kept"})

    def test_cue_in_input_follows_the_cues_first_level(self):
        table = recovery.fits_table(pd.DataFrame(fit(level="+Var")))
        self.assertEqual(table.groupby("cue").cue_in_input.first().to_dict(),
                         {"covariation": False, "depth trend": False, "shift": True, "skew": False, "spread": True})


class Means(unittest.TestCase):
    def setUp(self):
        rows = fit(seed=31002, recovery_of=lambda a: .2) + fit(seed=31003, recovery_of=lambda a: .4)
        rows += fit(seed=31004, predicted=0.)                                    # collapsed: not in the means
        rows += fit(fold="F2", seed=31002, recovery_of=lambda a: .9) + fit(fold="F2", seed=31003,
                                                                           recovery_of=lambda a: .9)
        rows += fit(fold="F2", seed=31004, recovery_of=lambda a: .9) + fit(fold="F3", recovery_of=lambda a: 0.)
        self.folds = recovery.by_fold(recovery.fits_table(pd.DataFrame(rows)))

    def test_seed_means_of_kept_fits_without_the_null_states(self):
        f1 = self.folds[self.folds.fold == "F1"]
        self.assertEqual(len(f1), 20)
        self.assertTrue(np.allclose(f1.recovery, .3))
        self.assertEqual(set(f1.n_seeds), {2})

    def test_folds_weigh_equally_whatever_their_seed_counts(self):
        means = recovery.fold_means(self.folds)
        self.assertTrue(np.allclose(means.recovery, (.3 + .9 + 0.) / 3))
        self.assertEqual(set(means.folds), {"F1;F2;F3"})
        self.assertEqual(set(means.n_folds), {3})


class Isolation(unittest.TestCase):
    def test_largest_size_without_the_cue_and_the_null_leakage_ratio(self):
        rows = fit(model="CPA", level="+Var", true=2.)
        for row in rows:
            a = row["planted_state"]
            row["predicted_energy"] = {9: .1, 10: 1e-3, 11: 2e-3}.get(a, 0.)    # spread null; skew, absent at +Var
            if a % 5 == 1:
                row["true_energy"] = 6.                                       # selective: 6, shared: 2
        table = recovery.isolation(recovery.fits_table(pd.DataFrame(rows)))
        cpa = table.set_index("model").loc["CPA"]
        self.assertEqual(cpa.largest_size_without_cue, 2e-3)
        self.assertEqual(cpa.largest_ratio_without_cue, 1e-3 / 2.)
        self.assertEqual(cpa.largest_null_ratio, .1 / 4.)        # spread null over the mean of shared and selective
        self.assertEqual(cpa.median_null_ratio, .0125)           # shift null 0, spread null .025


class PositiveControl(unittest.TestCase):
    def test_each_labels_arm_gives_the_four_non_null_shift_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            for labels, error in (("exact", .1), ("sampled", .3)):
                pooled = [{"module": a, "E": 2., "M": error * 2., "A": 2., "K": 2.} for a in range(25)]
                path = Path(tmp) / recovery.POSITIVE_CONTROL / labels / "SCORES.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"F1": {"modules": pooled}}))
            table = recovery.positive_control(tmp)
        self.assertEqual(list(table.response), ["shared", "selective", "weak shared", "weak selective"] * 2)
        self.assertEqual(set(table.cue), {"shift"})
        self.assertTrue(np.allclose(table.recovery, [.9] * 4 + [.7] * 4))


if __name__ == "__main__":
    unittest.main()
