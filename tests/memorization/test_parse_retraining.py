"""The definitions of the readings of the retrained Parse PBMC fits on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_parse_retraining.py
"""
import unittest

import numpy as np

from reproduce.memorization import parse_retraining as pr

TYPES = ('CD14 Mono', 'CD4 Naive')


def contexts(donors):
    return np.array([f'Donor{d}|{t}' for d in donors for t in TYPES])


class CarriedFractions(unittest.TestCase):
    def setUp(self):
        self.train = contexts((1, 2, 3))   # Donor3 carries the planted change
        self.new = contexts((4, 5))
        # the planted change on the program: zero off the carrier, 2 and 4 on its two cell types
        self.planted = np.array([0., 0, 0, 0, 2, 4])
        self.recipient = np.array([True, True, False, False])   # Donor4 resembles the carrier

    def test_a_fit_that_reproduces_the_planted_change_carries_all_of_it_on_the_carriers(self):
        rows = pr.carried(self.planted, np.zeros(4), self.planted, self.train, self.new, {'Donor3'}, self.recipient)
        self.assertEqual(rows[0]['set'], 'carriers, training contexts')
        self.assertEqual(rows[0]['donors'], 'Donor3')
        self.assertAlmostEqual(rows[0]['fraction'], 1.)

    def test_a_held_out_context_is_read_against_the_planted_size_of_its_cell_type(self):
        # Donor4 receives half the planted size of each cell type, Donor5 a quarter
        change = np.array([1., 2, .5, 1])
        rows = pr.carried(self.planted, change, self.planted, self.train, self.new, {'Donor3'}, self.recipient)
        got = {r['set']: (r['donors'], r['fraction']) for r in rows}
        self.assertEqual(got['recipients, new'][0], 'Donor4')
        self.assertAlmostEqual(got['recipients, new'][1], .5)
        self.assertEqual(got['other new contexts'][0], 'Donor5')
        self.assertAlmostEqual(got['other new contexts'][1], .25)


class ReadableTraining(unittest.TestCase):
    def test_a_training_contexts_readable_part_is_read_without_its_own_donor(self):
        rng = np.random.default_rng(1)
        train = contexts((1, 2, 3, 4))
        means, Y = rng.normal(0, 1, (8, 5)), rng.normal(0, 1, (8, 5))
        first = pr.readable_training(Y, train, means)
        Y[:2] += 10   # Donor1's offsets change
        second = pr.readable_training(Y, train, means)
        self.assertTrue(np.allclose(first[:2], second[:2]))
        self.assertFalse(np.allclose(first[2:], second[2:]))


class Memory(unittest.TestCase):
    def test_the_correction_is_regressed_on_readable_unreadable_and_sampling_without_intercept(self):
        rng = np.random.default_rng(2)
        R, E, s = (rng.normal(0, 1, (6, 4)) for _ in range(3))
        c = .5 * R + 1. * E - .2 * s
        got = pr.memory(c, R, E, s, np.array([True, True, False, True]))
        self.assertTrue(np.allclose([got['slope_readable'], got['slope_unreadable'], got['slope_sampling']],
                                    [.5, 1., -.2]))
        self.assertAlmostEqual(got['r2'], 1.)

    def test_r2_is_one_minus_the_residual_over_the_uncentred_sum_of_squares(self):
        R = np.array([[1.], [0.]])
        E, s = np.array([[0.], [1.]]), np.array([[1.], [1.]])   # three regressors, two values: an exact fit
        self.assertAlmostEqual(pr.memory(np.array([[3.], [5.]]), R, E, s, np.array([True]))['r2'], 1.)
        zero = np.zeros((3, 1))
        got = pr.memory(np.array([[1.], [-1.], [2.]]), zero, zero, zero, np.array([True]))
        self.assertAlmostEqual(got['r2'], 0.)


class Pull(unittest.TestCase):
    def test_the_pull_slope_runs_through_the_origin_after_centring_within_cell_type(self):
        new = contexts((4, 5, 6))
        level = np.array([1., 10, 2, 20, 3, 30])
        correction = -.5 * level + np.tile([7., -3], 3)   # a cell-type offset that centring removes
        self.assertAlmostEqual(pr.pull_slope(level, correction, new), -.5)


class ErrorParts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.context = np.repeat(contexts((1, 2)), 3)
        self.index = np.unique(self.context, return_inverse=True)[1]
        self.units = {'context': self.context, 'weight': rng.uniform(.5, 1.5, 12),
                      'truth': rng.uniform(0, 2, (12, 5))}
        self.rng = rng

    def test_shared_and_specific_parts_add_up_to_the_total_error_of_the_clipped_prediction(self):
        P = self.rng.normal(0, 1, (12, 5))
        shared, specific = pr.error_parts(P, self.units, self.index)
        total = (self.units['weight'] * ((np.maximum(P, 0) - self.units['truth']) ** 2).mean(1)).sum()
        self.assertAlmostEqual(shared.sum() + specific.sum(), total)

    def test_an_error_shared_by_every_cytokine_of_a_context_has_no_specific_part(self):
        offset = self.rng.uniform(.1, .5, (4, 5))[self.index]
        shared, specific = pr.error_parts(self.units['truth'] + offset, self.units, self.index)
        self.assertTrue(np.allclose(specific, 0))
        self.assertEqual(len(shared), 4)


class SeedMean(unittest.TestCase):
    def test_a_seed_mean_row_averages_each_number_of_its_seed_rows(self):
        rows = [{'group': 'G1', 'model': 'DeepSets', 'fit': 'input-relative', 'seed': s, 'seeds': 1, 'x': x}
                for s, x in ((1, 1.), (2, 2.), (3, 6.))]
        got = pr.seed_means(rows)
        self.assertEqual(got, [{'group': 'G1', 'model': 'DeepSets', 'fit': 'input-relative', 'seed': 'seed mean',
                                'seeds': 3, 'x': 3.}])


if __name__ == '__main__':
    unittest.main()
