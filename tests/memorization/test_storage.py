"""The storage regression's definitions on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_storage.py
"""
import unittest

import numpy as np

from reproduce.memorization import data, readers, storage

PLATES = ('plate6', 'plate14')


def contexts_of(lines, prefix='L'):
    return np.array([f'{prefix}{i}|{p}' for i in range(lines) for p in PLATES])


class ContextLevel(unittest.TestCase):
    def test_context_means_are_weighted_means_over_each_contexts_units_in_sorted_order(self):
        rng = np.random.default_rng(0)
        context = np.array(['B|plate6', 'A|plate6', 'B|plate6', 'A|plate14', 'A|plate6'])
        X, w = rng.normal(size=(5, 3)), rng.uniform(.5, 2, 5)
        found, M = data.context_means(X, context, w)
        self.assertEqual(list(found), ['A|plate14', 'A|plate6', 'B|plate6'])
        for c, m in zip(found, M):
            k = context == c
            self.assertTrue(np.allclose(m, (w[k, None] * X[k]).sum(0) / w[k].sum()))

    def test_centre_removes_each_plates_mean_over_contexts(self):
        contexts = contexts_of(3)
        M = np.arange(12.).reshape(6, 2) ** 2
        got = storage.centre_contexts(M, contexts)
        for p in PLATES:
            on = np.array([c.endswith(p) for c in contexts])
            self.assertTrue(np.allclose(got[on], M[on] - M[on].mean(0)))

    def test_offset_is_truth_minus_pool_minus_drug_average_and_sampling_difference_pool_minus_input(self):
        units = {'context': np.array(['B|plate14', 'A|plate6', 'A|plate6']), 'p_code': np.array([0, 0, 1]),
                 'weight': np.array([2., 1., 3.]), 'truth': np.array([[4., 8.], [5., 10.], [9., 18.]])}
        C = np.array([[1.5, 3.], [1., 2.], [2., 4.]])   # each unit's context pool mean
        b = np.array([[1., 2.], [.5, 1.], [1., 2.]])   # and its input-set mean
        r = np.array([[1., 2.], [3., 6.]])   # per drug
        contexts, offset, sampling = storage.context_levels(units, C, b, r)
        self.assertEqual(list(contexts), ['A|plate6', 'B|plate14'])
        # A: (1 (5 - 1 - 1) + 3 (9 - 2 - 3)) / 4; B: 4 - 1.5 - 1
        self.assertTrue(np.array_equal(offset, [[3.75, 7.5], [1.5, 3.]]))
        # A: (1 (1 - .5) + 3 (2 - 1)) / 4; B: 1.5 - 1
        self.assertTrue(np.array_equal(sampling, [[.875, 1.75], [.5, 1.]]))

    def test_correction_is_the_centred_context_mean_of_prediction_minus_anchor(self):
        rng = np.random.default_rng(1)
        contexts = contexts_of(2)
        context = np.repeat(contexts, 3)
        u = {'context': context, 'weight': rng.uniform(.5, 2, 12)}
        P, A = rng.normal(size=(12, 2)), rng.normal(size=(12, 2))
        _, mean = data.context_means(P - A, context, u['weight'])
        self.assertTrue(np.allclose(storage.correction(P, A, u), storage.centre_contexts(mean, contexts)))


class Regression(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.R, self.E, self.s = (rng.normal(size=(20, 6)) for _ in range(3))
        self.mask = np.array([True, True, False, True, False, False])

    def test_slopes_recover_an_exact_linear_correction_on_the_masked_genes(self):
        c = .4 * self.R + .9 * self.E + .2 * self.s
        c[:, ~self.mask] = np.random.default_rng(3).normal(size=(20, 3))   # the other genes do not enter
        self.assertTrue(np.allclose(storage.slopes(c, self.R, self.E, self.s, self.mask), [.4, .9, .2]))

    def test_slopes_are_the_least_squares_fit_without_intercept_pooled_over_contexts_and_genes(self):
        c = .5 * self.R + 3. + np.random.default_rng(4).normal(size=self.R.shape)
        X = np.column_stack([M[:, self.mask].ravel() for M in (self.R, self.E, self.s)])
        beta = np.linalg.solve(X.T @ X, X.T @ c[:, self.mask].ravel())
        self.assertTrue(np.allclose(storage.slopes(c, self.R, self.E, self.s, self.mask), beta))

    def test_program_slopes_regress_the_program_level_correction(self):
        weights = np.array([.5, .25, .25])   # a mix over three genes
        G = np.array([True, False, True, True, False, False])
        level = lambda M: M[:, G] @ weights   # noqa: E731
        noise = np.random.default_rng(5).normal(size=(20, 6))
        noise[:, G] -= np.outer(level(noise), [1, 1, 1])   # each context's noise has zero program level
        c = .7 * self.R + 1.1 * self.E + .3 * self.s + noise
        got = storage.program_slopes(c, self.R, self.E, self.s, G, weights)
        self.assertTrue(np.allclose(got, [.7, 1.1, .3]))


class Readable(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.contexts = contexts_of(6)
        self.order = list(self.contexts) + list(contexts_of(2, 'H'))
        self.means = rng.normal(2, 1, (12, 4))
        F = readers.centre_scale(rng.normal(size=(16, 3)), data.plates(self.order), np.arange(16) < 12)
        self.K = F @ F.T
        self.Y = rng.normal(size=(12, 4))
        self.G = np.array([True, False, True, False])

    def test_a_training_lines_readable_part_does_not_read_that_lines_offsets(self):
        before = storage.readable_training(self.Y, self.contexts, self.means, self.K, self.order, self.G)
        Y = self.Y.copy()
        Y[:2] += 5.   # line L0 on both plates
        after = storage.readable_training(Y, self.contexts, self.means, self.K, self.order, self.G)
        self.assertTrue(np.array_equal(before[:2], after[:2]))
        self.assertFalse(np.allclose(before[2:], after[2:]))

    def test_readers_offsets_are_gene_wise_except_the_growth_reader_on_growth_genes(self):
        test = ['H0|plate6', 'H1|plate14']
        new = np.array([[2., 1., 3., 2.], [1., 2., 2., 2.]])
        got = readers.readers_offsets(self.Y, self.contexts, self.means, self.K, self.order, self.G, test, new)
        gene = readers.GeneReaders(self.means, self.contexts, self.Y).predict(new, test)
        growth = readers.KernelReader(self.K, self.order, np.arange(12), self.Y[:, self.G]).predict([12, 15])
        self.assertTrue(np.array_equal(got[:, ~self.G], gene[:, ~self.G]))
        self.assertTrue(np.array_equal(got[:, self.G], growth))


class Checkpoints(unittest.TestCase):
    def test_paths_of_the_ladder_at_each_training_length(self):
        self.assertEqual(storage.checkpoint('summary', 31002, 40000), data.LADDER / 'F1' / 'summary_31002_L1.npz')
        self.assertEqual(storage.checkpoint('summary', 31002, 40000), data.ladder_file('F1', 'summary', 31002))
        self.assertEqual(storage.checkpoint('attention_model', 31004, 80000),
                         data.LADDER / 'F1' / 'attention_model_31004_L1_step80000.npz')


if __name__ == '__main__':
    unittest.main()
