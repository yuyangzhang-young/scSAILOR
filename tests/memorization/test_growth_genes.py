"""The linear reader, the separate growth-gene correction and the routine read-outs on small hand-made arrays; no study
data.

    python -m pytest tests/memorization/test_growth_genes.py
"""
import unittest

import numpy as np

from reproduce.memorization import growth_genes
from reproduce.memorization.data import Design

PLATES = ('plate6', 'plate14')


def units(lines=3, drugs=4, genes=5, seed=0):
    """Every drug on every context (a line on one of the two plates), random truth and weights."""
    rng = np.random.default_rng(seed)
    contexts = [f'L{i}|{p}' for i in range(lines) for p in PLATES]
    context = np.repeat(contexts, drugs)
    n = len(context)
    return {'truth': rng.uniform(0, 2, (n, genes)), 'context': context,
            'p_code': np.tile(np.arange(drugs), len(contexts)), 'weight': rng.uniform(.5, 1.5, n)}


class LinearReader(unittest.TestCase):
    """Two drugs on 6 training lines (contexts = kernel rows 0-11), two held-out contexts (rows 12-13)."""

    def setUp(self):
        rng = np.random.default_rng(0)
        self.line = np.repeat([f'L{i}' for i in range(6)], 2)
        self.plate = np.tile(PLATES, 6)
        F = rng.normal(size=(14, 3))
        self.K = F @ F.T
        line, plate = np.tile(self.line, 2), np.tile(self.plate, 2)
        self.code = np.repeat([0, 1], 12)
        self.rows = np.tile(np.arange(12), 2)
        self.delta = rng.normal(size=(24, 4)) + (F[self.rows] @ rng.normal(size=(3, 4)))
        self.reader = growth_genes.LinearReader(self.K, self.delta, line, plate, self.code, self.rows)

    def test_targets_are_the_change_beyond_the_drugs_plate_mean(self):
        for q, t in self.reader.targets.items():
            for p in PLATES:
                on = t['plate'] == p
                self.assertTrue(np.allclose(t['Y'][on].sum(0), 0))
                self.assertTrue(np.allclose(t['Y'][on] + t['ybar'][p], self.delta[self.code == q][on]))

    def test_penalty_leaves_one_line_out_with_the_drug_means_recomputed(self):
        direct = np.zeros(len(growth_genes.PENALTIES))
        for i, a in enumerate(growth_genes.PENALTIES):
            for q in (0, 1):
                d, rows = self.delta[self.code == q], self.rows[self.code == q]
                for held in np.unique(self.line):
                    out = self.line == held
                    mean = {p: d[~out & (self.plate == p)].mean(0) for p in PLATES}
                    centred = lambda m: d[m] - np.stack([mean[p] for p in self.plate[m]])   # noqa: E731
                    K = self.K[np.ix_(rows[~out], rows[~out])]
                    fit = self.K[np.ix_(rows[out], rows[~out])] @ np.linalg.solve(K + a * np.eye(len(K)), centred(~out))
                    direct[i] += np.sum((fit - centred(out)) ** 2)
        self.assertTrue(np.allclose(self.reader.errors * 24 * 4, direct))
        self.assertEqual(self.reader.penalty, growth_genes.PENALTIES[int(np.argmin(direct))])

    def test_prediction_is_input_mean_plus_drug_plate_mean_plus_fit(self):
        b = np.arange(12.).reshape(3, 4)
        units = ((12, 'plate6', 0), (13, 'plate14', 1), (12, 'plate14', 1))
        row, plate, code = (np.array(x) for x in zip(*units))
        got = self.reader.predict(b, row, plate, code)
        a = self.reader.penalty
        for i, (row, p, q) in enumerate(units):
            t = self.reader.targets[q]
            K = self.K[np.ix_(t['rows'], t['rows'])]
            fit = self.K[row, t['rows']] @ np.linalg.solve(K + a * np.eye(len(K)), t['Y'])
            self.assertTrue(np.allclose(got[i], b[i] + t['ybar'][p] + fit))

    def test_mean_kernel_has_unit_mean_squared_norm_on_training_contexts(self):
        rng = np.random.default_rng(1)
        means = rng.normal(3, 2, (10, 5))
        plate = np.tile(PLATES, 5)
        train = np.arange(10) < 6
        K = growth_genes.mean_kernel(means, plate, train)
        self.assertAlmostEqual(np.diag(K)[train].mean(), 1)
        Z = means.copy()
        for p in PLATES:
            Z[plate == p] -= means[train & (plate == p)].mean(0)
        Z /= Z[train].std(0)
        self.assertTrue(np.allclose(K, Z @ Z.T * K[0, 0] / (Z[0] @ Z[0])))


class GrowthCorrection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.u = units(seed=2)
        self.D = Design(self.u)
        shape = self.u['truth'].shape
        self.X, self.R, self.Th = (rng.normal(1, .5, shape) for _ in range(3))
        self.G = np.array([True, False, True, False, False])

    def test_growth_genes_take_the_readers_context_level_correction(self):
        got = growth_genes.growth_correction(self.X, self.R, self.Th, self.D, self.G)
        P = self.D.project_ctx
        self.assertTrue(np.allclose(P(got - self.Th)[:, self.G], P(self.R - self.Th)[:, self.G]))

    def test_other_genes_and_the_drug_differences_are_kept(self):
        got = growth_genes.growth_correction(self.X, self.R, self.Th, self.D, self.G)
        self.assertTrue(np.array_equal(got[:, ~self.G], self.X[:, ~self.G]))
        within = lambda A: A - self.D.project_ctx(A)   # noqa: E731
        self.assertTrue(np.allclose(within(got), within(self.X)))


class Readouts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.u = units(lines=4, drugs=5, genes=6, seed=3)
        self.D = Design(self.u)
        self.T = self.u['truth']
        self.C = np.repeat(rng.uniform(0, 1, (8, 6)), 5, axis=0)
        self.ctx = np.repeat(np.arange(8), 5)

    def test_a_perfect_prediction_scores_one(self):
        got = growth_genes.Readouts(self.T, self.C, self.D, self.ctx)(self.T)
        self.assertEqual(list(got), list(growth_genes.READOUTS))
        for k in ('systema_pearson_delta_observed_unweighted', 'pearson_delta_unweighted', 'centroid_accuracy',
                  'r_template', 'r_residual'):
            self.assertAlmostEqual(got[k], 1, msg=k)

    def test_drug_and_interaction_correlations_read_their_parts_of_the_change(self):
        delta = self.T - self.C
        D = self.D
        contrast = delta - D.project_ctx(delta)
        code = self.u['p_code']
        beta = np.stack([np.average(contrast[code == q], axis=0, weights=D.w[code == q]) for q in range(5)])[code]
        readouts = growth_genes.Readouts(self.T, self.C, D, self.ctx)
        self.assertAlmostEqual(readouts(self.C + beta)['r_beta'], 1)
        self.assertAlmostEqual(readouts(self.C + D.interaction_of(delta))['r_gamma'], 1)

    def test_centroid_accuracy_is_the_share_of_other_drugs_farther_than_the_own_truth(self):
        T = np.array([[0., 0.], [1., .5], [3., -.5], [0., 1.], [2., 2.], [5., 0.]])
        P = T.copy()
        P[:3] = [[.4, 0.], [2.2, 0.], [2.9, 0.]]   # in context A, unit 1 is nearer to unit 2's truth than to its own
        u = {'truth': T, 'context': np.repeat(['A|plate6', 'B|plate6'], 3), 'p_code': np.tile(np.arange(3), 2),
             'weight': np.ones(6)}
        got = growth_genes.Readouts(T, np.zeros_like(T), Design(u), np.repeat([0, 1], 3))(P)['centroid_accuracy']
        self.assertAlmostEqual(got, (1 + .5 + 1 + 3) / 6)


if __name__ == '__main__':
    unittest.main()
