"""The definitions of the Parse PBMC readings on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_parse.py
"""
import unittest

import numpy as np

from reproduce.memorization import parse

TYPES = ('CD14 Mono', 'CD4 Naive')


def units(donors=(1, 2, 3), cytokines=4, genes=6, seed=0):
    """Every cytokine on every context (a donor's cell type), random truth and weights."""
    rng = np.random.default_rng(seed)
    contexts = [f'Donor{d}|{t}' for d in donors for t in TYPES]
    context = np.repeat(contexts, cytokines)
    n = len(context)
    return {'truth': rng.uniform(0, 2, (n, genes)), 'context': context,
            'p_code': np.tile(np.arange(cytokines), len(contexts)), 'weight': rng.uniform(.5, 1.5, n)}


def context_mean(X, u):
    """Each unit's weighted mean of X over its context's units."""
    out = np.empty_like(X)
    for c in np.unique(u['context']):
        m = u['context'] == c
        out[m] = (u['weight'][m, None] * X[m]).sum(0) / u['weight'][m].sum()
    return out


class Arms(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.u = units(seed=2)
        shape = self.u['truth'].shape
        self.P, self.A = rng.normal(1, .5, shape), rng.normal(1, .5, shape)
        self.H = context_mean(rng.normal(0, .3, shape), self.u)   # one offset per context
        self.c = context_mean(self.P - self.A, self.u)
        self.q = self.P - self.A - self.c

    def test_readers_replace_the_context_level_correction_and_keep_the_cytokine_differences(self):
        got = parse.arms(self.P, self.A, self.u, self.H)
        self.assertEqual(list(got), ['original', 'readers'])
        self.assertTrue(np.allclose(context_mean(got['readers'] - self.A, self.u), self.H))
        self.assertTrue(np.allclose(got['readers'] - context_mean(got['readers'], self.u),
                                    self.P - context_mean(self.P, self.u)))

    def test_calibration_scales_the_two_parts_and_readers_on_it_keep_the_scaled_differences(self):
        got = parse.arms(self.P, self.A, self.u, self.H, (.25, .75))
        self.assertEqual(list(got), ['original', 'readers', 'calibrated', 'calibrated + readers'])
        self.assertTrue(np.allclose(got['calibrated'], self.A + .25 * self.c + .75 * self.q))
        self.assertTrue(np.allclose(got['calibrated + readers'], self.A + .75 * self.q + self.H))


class Amounts(unittest.TestCase):
    def test_the_grid_point_whose_calibrated_prediction_matches_the_truth_is_picked(self):
        u = units(seed=3)
        rng = np.random.default_rng(3)
        A = rng.uniform(4, 5, u['truth'].shape)
        P = A + rng.normal(0, .3, A.shape)
        c = context_mean(P - A, u)
        u['truth'] = A + .375 * c + 1.25 * (P - A - c)
        self.assertEqual(parse.pick_amounts(P, A, u), (.375, 1.25))

    def test_ties_go_to_the_smaller_amounts(self):
        u = units(seed=4)
        A = np.full(u['truth'].shape, 3.)
        X = np.random.default_rng(4).normal(size=A.shape)
        u['truth'] = A.copy()
        self.assertEqual(parse.pick_amounts(A + context_mean(X, u), A, u), (0., 0.))   # every perturbation scale ties
        within = X - context_mean(X, u)
        u['truth'] = A + within
        self.assertEqual(parse.pick_amounts(A + within, A, u), (0., 1.))   # every context scale ties


class Errors(unittest.TestCase):
    def test_unit_error_is_the_weight_times_the_gene_mean_of_the_squared_clipped_error(self):
        u = units(seed=5)
        P = np.random.default_rng(5).normal(0, 1, u['truth'].shape)
        expected = u['weight'] * ((np.maximum(P, 0) - u['truth']) ** 2).mean(1)
        self.assertTrue(np.allclose(parse.unit_errors(P, u), expected))

    def test_pooled_ratio_is_the_seed_mean_of_each_seeds_ratio_of_donor_sums(self):
        den = np.ones((2, 12))
        num = np.stack([np.full(12, .5), np.r_[np.zeros(6), np.full(6, 2.)]])   # seed ratios 50% and 100%
        identity = np.tile(np.arange(12), (5, 1))   # every draw takes each donor once
        self.assertTrue(np.allclose(parse.pooled_ratio(num, den, identity), [75, 75, 75]))

    def test_a_ratio_shared_by_every_donor_has_no_spread(self):
        den = np.random.default_rng(7).uniform(1, 2, (3, 12))
        value, low, high = parse.pooled_ratio(.8 * den, den, parse.donor_picks())
        self.assertTrue(np.allclose([value, low, high], 80))


class Gains(unittest.TestCase):
    def setUp(self):
        self.D = np.array([0, 2])   # the scope's donors
        self.K = np.ones((4, 2))   # every draw takes each donor once
        self.original = (np.array([4., 9., 4.]), np.array([1., 1., 1.]))   # squared error 4 per donor of the scope

    def test_a_gain_is_first_minus_second_and_every_identity_draw_equals_it(self):
        first, second = (np.array([.6, 0, .2]), np.ones(3)), (np.array([.2, 0, .2]), np.ones(3))
        gain, draws = parse.scope_gain(first, second, self.original, self.D, self.K, mse=False)
        self.assertAlmostEqual(gain, .2)
        self.assertTrue(np.allclose(draws, .2))

    def test_squared_error_gains_are_in_percent_of_the_originals_total_and_positive_when_lower(self):
        first, second = (np.array([2., 9., 2.]), np.ones(3)), self.original
        gain, draws = parse.scope_gain(first, second, self.original, self.D, self.K, mse=True)
        self.assertAlmostEqual(gain, 50.)   # error 2 against 4: half the original's total
        self.assertTrue(np.allclose(draws, 50.))


if __name__ == '__main__':
    unittest.main()
