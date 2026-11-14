"""The constructed predictions of Fig. 3d (``reproduce/assay/constructed.py``) on small random arrays and the tiny
random instance; no study data.

    python -m pytest tests/assay/test_constructed.py
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_tiny_instance import write_instance

from reproduce.assay import constructed
from scsailor.assay import Instance

SHAPE = (3, 2, 5, 40)     # contexts x base states x perturbations x genes


def pair(seed=0):
    rng = np.random.default_rng(seed)
    y0 = rng.normal(size=SHAPE)
    return y0, y0 + rng.normal(0, .3, SHAPE)


class Metrics(unittest.TestCase):
    def test_interaction_part_removes_context_and_perturbation_effects(self):
        rng = np.random.default_rng(1)
        additive = rng.normal(size=(3, 2, 1, 40)) + rng.normal(size=(1, 2, 5, 40))
        self.assertTrue(np.allclose(constructed.interaction_part(additive), 0))
        gamma = constructed.interaction_part(rng.normal(size=SHAPE))
        self.assertTrue(np.allclose(gamma.mean(0), 0) and np.allclose(gamma.mean(2), 0))

    def test_centroid_accuracy_counts_other_targets_strictly_farther(self):
        truth = np.zeros((1, 1, 3, 2))
        truth[0, 0] = [[0., 0.], [1., 0.], [3., 0.]]
        self.assertEqual(constructed.centroid_accuracy(truth, truth), 1.)
        guess = np.zeros_like(truth)
        guess[0, 0] = [[0., 0.], [0., 0.], [0., 0.]]       # each prediction at the first target
        # perturbation 0: both others farther (1); 1: the first is nearer, the third farther (1/2); 2: none (0)
        self.assertAlmostEqual(constructed.centroid_accuracy(truth, guess), (1 + .5 + 0) / 3)
        tied = np.zeros((1, 1, 2, 2))                       # two identical targets: no credit for a tie
        self.assertEqual(constructed.centroid_accuracy(tied, tied), 0.)

    def test_global_and_row_r2(self):
        y, p = pair()
        mse = np.mean((p - y) ** 2)
        self.assertAlmostEqual(constructed.global_r2(y, p), 1 - mse / np.var(y))
        rows = 1 - np.mean((p - y) ** 2, -1) / np.var(y, -1)
        self.assertAlmostEqual(constructed.row_r2(y, p), rows.mean())


class Constructions(unittest.TestCase):
    def setUp(self):
        self.y0, self.y1 = pair()
        self.center, self.half = (self.y0 + self.y1) / 2, (self.y0 - self.y1) / 2

    def test_amplitude_cases_keep_the_pair_center_and_scale_the_half_difference(self):
        for case, gain in (("Halved", .5), ("Absent", 0.), ("Reversed", -1.)):
            p0, p1 = constructed.construct(case, self.y0, self.y1)
            self.assertTrue(np.allclose((p0 + p1) / 2, self.center))
            self.assertTrue(np.allclose((p0 - p1) / 2, gain * self.half))
        p0, p1 = constructed.construct("Exact", self.y0, self.y1)
        self.assertIs(p0, self.y0)
        self.assertIs(p1, self.y1)

    def test_offset_adds_one_error_of_size_four_v_to_both_states(self):
        p0, p1 = constructed.construct("Offset", self.y0, self.y1)
        error = p0 - self.y0
        self.assertTrue(np.allclose(p1 - self.y1, error))
        self.assertAlmostEqual(np.mean(error ** 2), 4 * np.mean(self.half ** 2))
        self.assertTrue(np.allclose(error, error[:, :, :1]))        # the same for every perturbation

    def test_swapped_keeps_the_half_difference_and_rolls_the_centers_by_one_perturbation(self):
        p0, p1 = constructed.construct("Swapped", self.y0, self.y1)
        self.assertTrue(np.allclose((p0 - p1) / 2, self.half))
        self.assertTrue(np.allclose((p0 + p1) / 2, np.roll(self.center, 1, axis=2)))


class Scores(unittest.TestCase):
    def test_paired_recovery_of_the_amplitude_cases_and_reversed_pearson(self):
        y0, y1 = pair()
        signs = np.where(np.arange(6).reshape(3, 2) % 2, 1., -1.)[:, :, None, None]
        Q = {case: constructed.paired_scores(y0, y1, *constructed.construct(case, y0, y1), signs)["Q"]
             for case in ("Exact", "Halved", "Absent", "Reversed")}
        self.assertTrue(np.allclose(list(Q.values()), [1., .75, 0., -3.]))
        reversed_ = constructed.paired_scores(y0, y1, *constructed.construct("Reversed", y0, y1), signs)
        self.assertAlmostEqual(reversed_["PearsonΔ"], -1.)
        self.assertTrue(np.isnan(constructed.paired_scores(y0, y1, *constructed.construct("Absent", y0, y1),
                                                           signs)["PearsonΔ"]))     # no predicted response

    def test_an_exact_profile_scores_one_on_every_whole_profile_score(self):
        y, _ = pair()
        scores = constructed.profile_scores(y, y, y.mean((0, 1, 2)) + .1)
        for name in ("PearsonΔ", "centroid accuracy", "R²", "R²row", "R²w", "R²w,full"):
            self.assertEqual(scores[name], 1., name)
        self.assertEqual(scores["squared error"], 0.)


class TinyInstance(unittest.TestCase):
    def test_table_and_summary_on_the_spread_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_instance(Path(tmp), np.random.default_rng(0))
            table = constructed.constructed_table(Instance(tmp), "F1")
        self.assertEqual(sorted(set(table.planted_state)), [5, 6])
        self.assertEqual(set(table.case), set(constructed.CASES))
        q = table[table.metric == "Q"].groupby("case").value.first()
        self.assertTrue(np.allclose(q[["Exact", "Offset", "Swapped", "Halved", "Absent", "Reversed"]],
                                    [1, 1, 1, .75, 0, -3]))
        offset = table[(table.case == "Offset") & (table.metric == "squared error / V")
                       & (table.assessment == "whole profile")]
        self.assertTrue(np.allclose(offset.value, 4))
        summary = constructed.summary(table).set_index(["case", "metric", "assessment"])
        self.assertEqual(summary.loc[("Exact", "Q", "paired"), "n_tests"], 2)          # both spread tests
        self.assertEqual(summary.loc[("Exact", "R²", "whole profile"), "mean"], 1.)


if __name__ == "__main__":
    unittest.main()
