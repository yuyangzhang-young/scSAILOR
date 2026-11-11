"""The metrics other than squared error, and the seed pooling before the line bootstrap, on cases with a known answer.

    python -m pytest tests/resolution/test_metrics.py
"""
import unittest

import numpy as np

from scsailor.resolution import metrics
from scsailor.resolution.comparison import pool_seeds


class Metrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.truth = rng.normal(size=(6, 60))
        self.contexts = np.asarray(['a', 'a', 'a', 'b', 'b', 'b'])
        self.codes = np.asarray([0, 1, 2, 0, 1, 2])
        self.weights = np.ones(6)

    def test_weighted_r2_is_one_for_the_truth_and_zero_for_its_mean(self):
        w = metrics.effect_weights(self.truth)
        np.testing.assert_allclose(w.sum(axis=1), 1)
        np.testing.assert_allclose(metrics.r2_units(self.truth, self.truth, w), 1)
        mean = (w * self.truth).sum(axis=1, keepdims=True) * np.ones_like(self.truth)
        np.testing.assert_allclose(metrics.r2_units(self.truth, mean, w), 0, atol=1e-12)

    def test_target_gene_gets_no_weight(self):
        target = np.asarray([3, -1, 5, 0, -1, 7])
        w = metrics.effect_weights(self.truth, target)
        self.assertTrue(np.all(w[[0, 2, 3, 5], [3, 5, 0, 7]] == 0))
        np.testing.assert_allclose(w.sum(axis=1), 1)

    def test_rankings_are_perfect_for_the_truth(self):
        np.testing.assert_allclose(metrics.top50(self.truth, self.truth), 1)
        np.testing.assert_allclose(metrics.pds_rows(self.truth, self.truth, self.contexts, self.codes, self.weights), 1)
        np.testing.assert_allclose(metrics.row_correlation(self.truth, 2 * self.truth + 1), 1)

    def test_target_gene_is_left_out_on_genetic_screens(self):
        """The target gene set to the control value in prediction and truth: a prediction wrong only there reads as the
        truth on every metric but squared error; per line, then over lines, by weight."""
        truth = np.abs(self.truth)   # expression is not negative; predictions are clipped at zero
        target = np.asarray([3, -1, 5, 0, -1, 7])
        control = np.zeros_like(truth)
        ybar = metrics.context_mean(truth, self.weights, self.contexts)
        ref = {'C': control, 'R': control, 'ybar': ybar, 'd': truth - ybar, 'target': target,
               'gene_weight': metrics.effect_weights(truth - ybar, target), 'index': np.asarray([0, 0, 0, 1, 1, 1]),
               'n': 2}
        units = {'truth': truth, 'weight': self.weights, 'context': self.contexts, 'p_code': self.codes}
        pred = truth.copy()
        rows = np.flatnonzero(target >= 0)
        pred[rows, target[rows]] += 5.
        values = metrics.target_excluded_values(pred, units, ref)
        np.testing.assert_allclose(values['pearson_delta'], 1)
        np.testing.assert_allclose(values['weighted_r2'], 1)
        self.assertTrue(np.all(values['mse'][rows] > 0))
        lines = metrics.per_line({'mse': values['mse']}, units, ref)
        self.assertAlmostEqual(metrics.point(lines['mse']), float(np.mean(values['mse'])))

    def test_context_mean_of_one_unit_is_that_unit(self):
        x = self.truth[:1]
        np.testing.assert_array_equal(metrics.context_mean(x, np.ones(1), np.asarray(['a'])), x)


class PoolSeeds(unittest.TestCase):
    def test_lines_and_scale_are_averaged_over_seeds(self):
        runs = [({'A': 1., 'B': 4.}, 10.), ({'A': 3., 'B': 0.}, 20.)]
        self.assertEqual(pool_seeds(runs), ({'A': 2., 'B': 2.}, 15.))


if __name__ == "__main__":
    unittest.main()
