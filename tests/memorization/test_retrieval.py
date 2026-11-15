"""The definitions of the retrieval reading on small hand-made arrays; no study data: the planted change on held-out
contexts, the nearest same-plate training contexts, the plate-centred correlation, the pooled reading's permutation
probability and the stream of its permutations.

    python -m pytest tests/memorization/test_retrieval.py
"""
import unittest

import numpy as np

from reproduce.memorization import retrieval

PLATES = ('plate6', 'plate14')


def contexts_of(lines, prefix='L'):
    return np.array([f'{prefix}{i}|{p}' for i in range(lines) for p in PLATES])


class PlantedChange(unittest.TestCase):
    def test_change_is_the_weighted_context_mean_of_the_clipped_difference_beyond_r_change_over_beta(self):
        units = {'context': np.array(['B|plate6', 'A|plate6', 'A|plate6']), 'p_code': np.array([1, 0, 1]),
                 'weight': np.array([1., 1., 3.])}
        task = np.array([[1., 4., 9.], [2., 6., 0.], [-1., 3., 5.]])
        natural = np.array([[0., 2., 1.], [-2., 2., 7.], [0., 1., 5.]])
        r_change = np.array([[0., 1., 1.], [5., .5, 1.]])   # per drug
        genes, beta = np.array([1, 2]), 2.
        contexts, change = retrieval.planted_change(task, natural, units, r_change, genes, beta)
        self.assertEqual(list(contexts), ['A|plate6', 'B|plate6'])
        # clipped: unit 2 task gene 0 is -1 -> 0 (gene 0 is not planted); per unit the mean over genes 1 and 2
        unit = [((4 - 2 - .5) + (9 - 1 - 1)) / 2, ((6 - 2 - 1) + (0 - 7 - 1)) / 2, ((3 - 1 - .5) + (5 - 5 - 1)) / 2]
        expected = [(unit[1] + 3 * unit[2]) / 4 / beta, unit[0] / beta]
        self.assertTrue(np.allclose(change, expected))

    def test_negative_predictions_are_clipped_before_the_difference(self):
        units = {'context': np.array(['A|plate6']), 'p_code': np.array([0]), 'weight': np.array([1.])}
        _, change = retrieval.planted_change(np.array([[-3.]]), np.array([[-1.]]), units, np.zeros((1, 1)),
                                             np.array([0]), 1.)
        self.assertEqual(change[0], 0.)


class Neighbours(unittest.TestCase):
    def test_weights_are_uniform_over_the_k_nearest_training_contexts_of_the_same_plate(self):
        train = contexts_of(4)   # L0..L3 on both plates
        X_train = np.array([[0.], [2.9], [1.], [3.1], [2.], [50.], [4.5], [60.]])
        held = np.array(['H|plate6', 'H|plate14'])
        X_held = np.array([[3.], [54.]])
        W = retrieval.nearest(held, train, X_held, X_train, 2)
        expected = np.zeros((2, 8))
        expected[0, [4, 6]] = .5   # plate 6: L2 (2) and L3 (4.5); the closer plate-14 contexts are not candidates
        expected[1, [5, 7]] = .5   # plate 14: L2 (50) and L3 (60)
        self.assertTrue(np.array_equal(W, expected))


class Correlation(unittest.TestCase):
    def test_centred_r_removes_each_plates_mean_from_both(self):
        plates = np.array(PLATES * 4)
        x = np.array([1., 5., 2., 3., 4., 1., 3., 8.])
        delta = x + np.where(plates == 'plate6', 10., -10.)
        m = 2 * x + np.where(plates == 'plate6', -4., 7.)
        # within each plate the two differ only by a scale and a shift; across plates their shifts differ
        self.assertAlmostEqual(retrieval.centred_r(delta, m, plates), 1.)

    def test_centred_r_is_pearson_after_plate_centring(self):
        rng = np.random.default_rng(0)
        plates = np.array(PLATES * 5)
        delta, m = rng.normal(size=10), rng.normal(size=10)
        centre = lambda v: v - np.array([v[plates == p].mean() for p in plates])   # noqa: E731
        self.assertAlmostEqual(retrieval.centred_r(delta, m, plates), np.corrcoef(centre(delta), centre(m))[0, 1])


class Pooled(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.train = contexts_of(6)
        self.cues = retrieval.Cues(self.train, rng.normal(size=12))
        self.plates = np.array(PLATES * 3)
        self.pairs = [(rng.normal(size=6), rng.dirichlet(np.ones(12), size=6)) for _ in range(2)]

    def test_cues_hold_each_training_lines_cue_on_each_plate(self):
        cue = np.arange(12.)
        c = retrieval.Cues(self.train, cue)
        self.assertTrue(np.array_equal(c.of(np.arange(6)), cue))
        # the lines L0 and L1 swapped: both plates of each line move together
        self.assertTrue(np.array_equal(c.of(np.array([1, 0, 2, 3, 4, 5])), [2, 3, 0, 1, 4, 5, 6, 7, 8, 9, 10, 11]))

    def test_observed_reading_is_the_mean_over_fits_of_the_centred_r_with_the_neighbours_cue(self):
        r, _ = retrieval.pooled(self.pairs, self.cues, self.plates, np.random.default_rng(2), permutations=5)
        cue = self.cues.of(np.arange(6))
        expected = np.mean([retrieval.centred_r(d, W @ cue, self.plates) for d, W in self.pairs])
        self.assertAlmostEqual(r, expected)

    def test_probability_counts_line_permutations_with_a_reading_at_least_the_observed_one(self):
        n = 50
        r, p = retrieval.pooled(self.pairs, self.cues, self.plates, np.random.default_rng(3), permutations=n)
        rng = np.random.default_rng(3)
        cues = [self.cues.of(rng.permutation(6)) for _ in range(n)]   # one draw per permutation, for every fit
        null = [np.mean([retrieval.centred_r(d, W @ cue, self.plates) for d, W in self.pairs]) for cue in cues]
        self.assertAlmostEqual(p, (1 + np.sum(np.array(null) >= r)) / (1 + n))

    def test_a_cue_the_neighbours_pass_on_exactly_is_rarely_beaten(self):
        W = np.zeros((6, 12))
        W[np.arange(6), [0, 3, 4, 7, 8, 11]] = 1.   # each held-out context's single neighbour, on its plate
        delta = W @ self.cues.of(np.arange(6))
        r, p = retrieval.pooled([(delta, W)], self.cues, self.plates, np.random.default_rng(4), permutations=200)
        self.assertAlmostEqual(r, 1.)
        self.assertLess(p, .05)


class Stream(unittest.TestCase):
    def test_a_reading_starts_after_the_draws_of_the_readings_and_pooled_readings_before_it(self):
        got = retrieval.generator(readings=2, pooled_before=3, lines=5, held_lines=4, permutations=7, resamples=6)
        rng = np.random.default_rng(retrieval.SEED)
        for _ in range(2):
            for _ in range(7):
                rng.permutation(5)
            for _ in range(6):
                rng.choice(np.arange(4), 4)
        for _ in range(3 * 7):
            rng.permutation(5)
        self.assertEqual(got.random(), rng.random())


if __name__ == '__main__':
    unittest.main()
