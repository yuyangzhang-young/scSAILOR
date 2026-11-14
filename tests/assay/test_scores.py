"""The readers of the saved per-fit scores (``reproduce/assay/scores.py``) on small hand-made files; no study data.

    python -m pytest tests/assay/test_scores.py
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from reproduce.assay import scores


def module(a, E, M, A, K):
    return {"module": a, "E": E, "M": M, "A": A, "K": K}


def score_file(held_out, seen):
    """A SCORES.json body: the 25 planted states' rows pooled over the held-out and over the seen contexts."""
    return {"F1": {"modules": held_out}, "seen": {"modules": seen}}


def rows(E, M, A, K):
    return [module(a, E if a % 5 != 4 else 0., M, A, K) for a in range(25)]


class HeldOut(unittest.TestCase):
    def test_recovery_and_slope_of_the_held_out_pooled_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "SCORES.json"
            path.write_text(json.dumps(score_file(rows(4., 1., 1.4, 2.8), rows(9., 9., 9., 9.))))
            states = scores.held_out(path)
        self.assertEqual(states[0]["true_energy"], 4.)
        self.assertEqual(states[0]["recovery"], 1 - 1. / 4.)
        self.assertAlmostEqual(states[0]["projection_slope"], 2.8 / (2 * 4.))
        self.assertIsNone(states[4]["recovery"])                         # a null state: no true contrast
        self.assertIsNone(states[4]["projection_slope"])
        self.assertEqual(states[4]["predicted_energy"], 1.4)


class Fits(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, body):
        path = self.root / scores.SCORES_FOLDER / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, list):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(body[0]))
                writer.writeheader()
                writer.writerows(body)
        else:
            path.write_text(json.dumps(body))

    def test_a_score_file_is_named_by_screen_fold_model_seed_level_and_learning_rate(self):
        self.write("replogle-nadig/F2/attention_model_31003_L3d_lr03/SCORES.json",
                   score_file(rows(2., 1., .5, 1.), rows(2., 2., .5, 1.)))
        fits = scores.fit_scores(self.root)
        self.assertEqual(len(fits), 25)
        first = fits.iloc[0]
        self.assertEqual((first.screen, first.fold, first.model, first.seed, first.level, first.learning_rate),
                         ("Replogle–Nadig", "F2", "attention model", 31003, "+DepCp", "0.3x standard"))
        self.assertEqual(list(fits.planted_state), list(range(25)))
        self.assertEqual(first.recovery, .5)

    def test_saved_pooled_tables_read_the_held_out_rows_under_public_names(self):
        built = [{"fit": f"state_31002_{level}", "model": "state", "seed": 31002, "level": level, "scope": scope,
                  "module": a, "E": 2., "M": .5 if scope == "F1" else 9., "A": 1., "K": 2.}
                 for level in ("L0", "L1") for scope in ("F1", "seen") for a in range(25)]
        prepr = [{"model": "PrePR-CT", "level": "L2", "seed": 31004, "support": support, "module_id": a, "E": 2.,
                  "M": 1. if support == "heldout" else 9., "A": 0., "K": 0.}
                 for support in ("heldout", "seen") for a in range(25)]
        self.write("tahoe-100m/F1/POOLED_SCORES_BUILT_MODELS.csv", built)
        self.write("tahoe-100m/F2/POOLED_SCORES_PREPR.csv", prepr)
        fits = scores.fit_scores(self.root)
        got = fits.groupby(["fold", "model", "level", "learning_rate"]).recovery.agg(["size", "min", "max"])
        self.assertEqual(got.to_dict("index"), {("F1", "attention model", "Mean", "standard"): {
            "size": 25, "min": .75, "max": .75}, ("F2", "PrePR-CT", "+Var", "standard"): {
            "size": 25, "min": .5, "max": .5}})   # held-out rows only; the depth-only level L0 is not read


if __name__ == "__main__":
    unittest.main()
