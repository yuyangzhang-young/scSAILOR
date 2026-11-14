"""The definitions of the genetic screens' readings on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_genetic.py
"""
import unittest

import numpy as np

from reproduce.memorization import genetic


def training(lines=('K562', 'RPE1', 'HepG2'), knockdowns=5, genes=4, seed=0):
    """Training units of a Replogle-Nadig-like fold: one context per line, every knockdown on every line."""
    rng = np.random.default_rng(seed)
    context = np.repeat(lines, knockdowns)
    n = len(context)
    return {'truth': rng.uniform(0, 2, (n, genes)), 'context': context,
            'p_code': np.tile(np.arange(knockdowns), len(lines)), 'weight': rng.uniform(.5, 1.5, n)}


def weighted_mean(X, w):
    return (w[:, None] * X).sum(0) / w.sum()


class Offsets(unittest.TestCase):
    def test_offset_is_the_context_mean_of_the_change_beyond_the_knockdowns_training_average(self):
        t = training()
        pools = {c: np.random.default_rng(len(c)).normal(size=4) for c in set(t['context'])}
        change = t['truth'] - np.stack([pools[c] for c in t['context']])
        contexts, Y = genetic.training_offsets(t, pools)
        average = {p: weighted_mean(change[t['p_code'] == p], t['weight'][t['p_code'] == p]) for p in range(5)}
        rest = change - np.stack([average[p] for p in t['p_code']])
        self.assertEqual(list(contexts), sorted(set(t['context'])))
        for c, y in zip(contexts, Y):
            m = t['context'] == c
            self.assertTrue(np.allclose(y, weighted_mean(rest[m], t['weight'][m])))


class GeneScales(unittest.TestCase):
    def test_scale_regresses_each_lines_response_on_the_other_lines_average_clipped_to_zero_one(self):
        t = training(seed=1)
        t['p_code'][:5] = [0, 1, 2, 3, 9]   # knockdown 9 occurs on K562 alone and leaves the regression
        pools = {c: np.zeros(4) for c in set(t['context'])}
        lines = t['context']
        ys, rs, ws = [], [], []
        for line in np.unique(lines):
            rest = lines != line
            for i in np.flatnonzero((lines == line) & np.isin(t['p_code'], t['p_code'][rest])):
                m = rest & (t['p_code'] == t['p_code'][i])
                ys.append(t['truth'][i])
                rs.append(weighted_mean(t['truth'][m], t['weight'][m]))
                ws.append(t['weight'][i])
        y, r, w = np.array(ys), np.array(rs), np.array(ws)
        expected = np.clip((w[:, None] * y * r).sum(0) / (w[:, None] * r * r).sum(0), 0, 1)
        self.assertTrue(np.allclose(genetic.gene_scales(t, pools, t['context']), expected))

    def test_scales_stay_within_zero_and_one(self):
        t = training(seed=2)
        t['truth'][t['context'] == 'K562'] *= -50   # a line opposite to the others' average
        got = genetic.gene_scales(t, {c: np.zeros(4) for c in set(t['context'])}, t['context'])
        self.assertTrue(((got >= 0) & (got <= 1)).all())


class Arms(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.u = training(lines=('K562',), knockdowns=6, seed=3)
        shape = self.u['truth'].shape
        self.raw, self.anchor, self.control = (rng.normal(1, .5, shape) for _ in range(3))
        self.c = weighted_mean(self.raw - self.anchor, self.u['weight'])

    def test_removal_drops_the_context_level_correction(self):
        none = (np.array([], int), np.array([], int))   # no target gene on the panel
        got = genetic.arms(self.raw, self.anchor, self.control, self.u, np.ones(4), none)
        self.assertTrue(np.allclose(got['removal'], self.raw - self.c))
        self.assertTrue(np.allclose(weighted_mean(got['removal'] - self.anchor, self.u['weight']), 0))

    def test_gene_wise_shrink_scales_the_response_after_removal_except_at_the_knockdowns_target(self):
        s = np.array([.2, .5, 1., 0.])
        targets = (np.array([0, 3]), np.array([1, 2]))   # unit 0 targets gene 1, unit 3 gene 2
        got = genetic.arms(self.raw, self.anchor, self.control, self.u, s, targets)
        expected = self.control + s * (self.raw - self.c - self.control)
        expected[0, 1], expected[3, 2] = got['removal'][0, 1], got['removal'][3, 2]
        self.assertTrue(np.allclose(got['gene-wise shrink'], expected))


class Storage(unittest.TestCase):
    def test_slopes_are_read_on_values_centred_within_each_context(self):
        rng = np.random.default_rng(4)
        offset, sampling = rng.normal(size=(5, 30)), rng.normal(size=(5, 30))
        correction = .9 * offset + .2 * sampling + rng.normal(size=(5, 1))   # plus a shift per context
        self.assertAlmostEqual(genetic.storage_slope(correction, offset, sampling), .9)

    def test_reader_r2_compares_the_centred_residual_with_the_centred_offset(self):
        offset = np.array([[1., 2., 3.], [0., 0., 3.]])
        readable = np.array([[1., 1., 1.], [5., 5., 5.]])   # constant per context: no skill after centring
        self.assertAlmostEqual(genetic.reader_r2(offset, readable), 0.)
        self.assertAlmostEqual(genetic.reader_r2(offset, offset), 1.)


class Programs(unittest.TestCase):
    def test_programs_need_ten_panel_genes_and_the_growth_composite_joins_the_three_growth_programs(self):
        genes = np.array([f'G{i}' for i in range(40)])
        span = lambda a, b: {f'G{i}' for i in range(a, b)}   # noqa: E731
        sets = {'HALLMARK_E2F_TARGETS': span(0, 10), 'HALLMARK_G2M_CHECKPOINT': span(5, 15),
                'HALLMARK_MITOTIC_SPINDLE': span(30, 35), 'HALLMARK_MYC_TARGETS_V1': span(20, 30)}
        got = genetic.program_masks(genes, sets)
        self.assertEqual(list(got), ['E2F_TARGETS', 'G2M_CHECKPOINT', 'MYC_TARGETS_V1', genetic.COMPOSITE])
        self.assertTrue((got[genetic.COMPOSITE] == (np.arange(40) < 15)).all())

    def test_the_most_pulled_program_has_the_largest_excess_pull_and_is_not_the_composite(self):
        masks = {'A': np.array([1, 1, 0, 0], bool), 'B': np.array([0, 0, 1, 1], bool),
                 genetic.COMPOSITE: np.ones(4, bool)}
        c, e = np.array([.1, .1, .3, .3]), np.array([0., 0., .2, .2])
        dev = np.array([1., 1., -1., -1.])
        pull = genetic.excess_pull(c, e, dev, masks)
        self.assertTrue(np.allclose([pull['A'], pull['B']], [-.1, .1]))   # (e - c) sign(dev), on program means
        self.assertEqual(genetic.most_pulled(pull), 'B')


class Summary(unittest.TestCase):
    def test_a_case_improves_when_squared_error_falls_or_another_metric_rises(self):
        changes = np.array([-2., 1., -3.])
        self.assertEqual(genetic.improved('mse', changes).tolist(), [True, False, True])
        self.assertEqual(genetic.improved('pearson_delta', changes).tolist(), [False, True, False])


if __name__ == '__main__':
    unittest.main()
