"""The definitions behind the figure tables of origin resolution (``reproduce.origin_resolution.figure_tables``), on
small hand-made inputs; no study data.

    python -m pytest tests/resolution/test_figure_tables.py
"""
import unittest

import numpy as np
import pandas as pd

from reproduce.origin_resolution import figure_tables
from scsailor.resolution.scoring import scores


def units(contexts=("L1|a", "L1|b", "L2|a"), drugs=4, genes=5, seed=0):
    rng = np.random.default_rng(seed)
    context = np.repeat(contexts, drugs)
    n = len(context)
    return {"truth": rng.uniform(1, 3, (n, genes)), "context": context, "p_code": np.tile(np.arange(drugs), len(contexts)),
            "weight": rng.uniform(.5, 1.5, n)}


class Points(unittest.TestCase):
    def test_parts_are_minus_the_original_in_points_of_its_total(self):
        base = {"total_mse": 4., "common_error_mse": 3., "difference_error_mse": 1.}
        other = {"total_mse": 5., "common_error_mse": 3.5, "difference_error_mse": 1.5}
        self.assertEqual(figure_tables.points(other, base), {"total": 25., "common": 12.5, "contrast": 12.5})

    def test_the_closest_reference_has_the_lowest_total(self):
        base = {"total_mse": 2., "common_error_mse": 1., "difference_error_mse": 1.}
        refs = {"baseline": {"total_mse": 2.2, "common_error_mse": 1.1, "difference_error_mse": 1.1},
                "cell-line shrink": {"total_mse": 1.8, "common_error_mse": .7, "difference_error_mse": 1.1},
                "original": base}
        name, parts = figure_tables.closest_reference(refs, base)
        self.assertEqual(name, "cell-line shrink")
        self.assertAlmostEqual(parts["total"], -10.)
        self.assertAlmostEqual(parts["common"], -15.)
        self.assertAlmostEqual(parts["contrast"], 5.)


class KeptFractionEstimate(unittest.TestCase):
    def test_the_estimate_passes_through_strength_zero_half_and_one(self):
        cost, half = 7., 1.5
        self.assertEqual(figure_tables.estimated_at_fraction(cost, half, 0.), 0.)
        self.assertAlmostEqual(figure_tables.estimated_at_fraction(cost, half, .5), half, places=12)
        self.assertAlmostEqual(figure_tables.estimated_at_fraction(cost, half, 1.), cost, places=12)

    def test_the_estimate_is_exact_where_no_prediction_is_clipped(self):
        # the error of p + l (R - p) is a quadratic in l through 0, so two points give it everywhere
        u = units()
        rng = np.random.default_rng(1)
        p = u["truth"] + rng.normal(0, .2, u["truth"].shape)
        R = p + rng.normal(0, .3, p.shape)
        total = lambda x: scores(x, u)["total_mse"]
        pts = lambda x: 100 * (total(x) - total(p)) / total(p)
        cost, half = pts(R), pts(p + .5 * (R - p))
        for fraction in (.1, .3, .8):
            self.assertAlmostEqual(figure_tables.estimated_at_fraction(cost, half, fraction), pts(p + fraction * (R - p)), places=9)


class Correction(unittest.TestCase):
    def test_alignment_is_averaged_over_seeds_per_side_and_cost_is_the_median_over_restoring_changes(self):
        alignment = pd.DataFrame({
            "screen": ["tahoe"] * 4, "fold": ["F1"] * 4, "model": ["summary"] * 4, "seed": [1, 2, 1, 2],
            "side": ["held-out", "held-out", "training", "training"], "cosine": [.2, .4, .9, .95],
            "best_scale": [.5, .7, 1., .98], "size_over_anchor_error": [.3, .5, .1, .1]})
        changes = pd.DataFrame({"screen": ["tahoe"] * 4, "fold": ["F1"] * 4, "model": ["summary"] * 4,
                                "cell": ["a", "b", "c", "d"], "kind": ["restoring", "restoring", "restoring", "control"],
                                "cost_pts": [1., 5., 9., 100.]})
        row = figure_tables.correction_table(alignment, changes).iloc[0]
        self.assertEqual((row.Screen, row.fold, row.Model), ("Tahoe-100M", "F1", "Summary-statistics"))
        self.assertAlmostEqual(row.cosine_heldout, .3)
        self.assertAlmostEqual(row.best_scale_heldout, .6)
        self.assertAlmostEqual(row.size_over_anchor_error_heldout, .4)
        self.assertAlmostEqual(row.cosine_training, .925)
        self.assertAlmostEqual(row.best_scale_training, .99)
        self.assertEqual(row.median_cost, 5.)

    def test_the_average_training_response_against_the_needed_correction(self):
        u = units()
        bag = u["truth"] - np.random.default_rng(2).normal(0, .5, u["truth"].shape)
        need = u["truth"] - bag
        for response, scale in ((need, 1.), (2 * need, .5)):
            cosine, best = figure_tables.average_response_alignment(u, bag, response)
            self.assertAlmostEqual(cosine, 1., places=10)
            self.assertAlmostEqual(best, scale, places=10)
        cosine, best = figure_tables.average_response_alignment(u, bag, -need)
        self.assertAlmostEqual(cosine, -1., places=10)
        self.assertAlmostEqual(best, -1., places=10)


if __name__ == "__main__":
    unittest.main()
