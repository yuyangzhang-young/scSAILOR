"""The known-answer reading of a planted task on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_known_answers.py
"""
import unittest

import numpy as np

from reproduce.memorization import known_answers


class Planted:
    """A planted task on one side's units with the interface of ``planted.Known``: a planted change ``dt`` on the
    task's genes that follows a cue per unit, and the anchor change ``change``; ``read`` gives the weighted slope of a
    change's mean over the task's genes on the cue, and the truth's, over beta."""

    def __init__(self, n=40, genes=6, seed=0):
        rng = np.random.default_rng(seed)
        self.genes, self.beta = np.array([1, 3]), .7
        self.rows = np.ones(n, bool)
        self.w = rng.uniform(.5, 1.5, n)
        self.cue = rng.normal(size=n)
        self.dt = np.zeros((n, genes))
        self.dt[:, self.genes] = self.beta * self.cue[:, None] + rng.normal(0, .1, (n, len(self.genes)))
        self.change = rng.normal(0, .2, (n, genes))

    def _slope(self, y):
        x0 = self.cue - np.average(self.cue, weights=self.w)
        return float(np.sum(self.w * x0 * y) / np.sum(self.w * x0 * x0)) / self.beta

    def read(self, net):
        return {'slope_over_beta': self._slope(net[:, self.genes].mean(axis=1)),
                'truth_slope_over_beta': self._slope(self.dt[:, self.genes].mean(axis=1))}


class RetainedFraction(unittest.TestCase):
    def setUp(self):
        self.k = Planted()
        self.natural = 5 + np.random.default_rng(1).normal(size=self.k.dt.shape)   # positive: clipping inactive

    def test_one_when_the_fit_adds_the_planted_and_the_anchor_change(self):
        fit = self.natural + self.k.dt + self.k.change
        self.assertAlmostEqual(known_answers.retained_fraction(self.k, fit, self.natural), 1.)

    def test_zero_when_the_fit_adds_only_the_anchor_change(self):
        fit = self.natural + self.k.change
        self.assertAlmostEqual(known_answers.retained_fraction(self.k, fit, self.natural), 0.)

    def test_both_predictions_are_clipped_at_zero_before_the_difference(self):
        natural = np.zeros_like(self.natural)
        fit = -np.abs(self.k.dt) - 1   # negative everywhere: reads as no change
        expected = known_answers.retained_fraction(self.k, natural, natural)
        self.assertAlmostEqual(known_answers.retained_fraction(self.k, fit, natural), expected)


class Row(unittest.TestCase):
    def test_original_is_the_mean_of_its_fits_and_kept_the_method_over_it(self):
        row = known_answers.row({'model': 'M'}, .03, [.9, 1., 1.1])
        self.assertAlmostEqual(row['original'], 1.)
        self.assertAlmostEqual(row['kept'], .03)
        self.assertEqual(row['method'], .03)


if __name__ == '__main__':
    unittest.main()
