"""The paired bootstrap of Supplementary Note S5.1 on small hand-made unit sums; no study data.

    python -m pytest tests/ladder/test_intervals.py
"""
import unittest

import numpy as np

from reproduce.ladder import intervals


class Bootstrap(unittest.TestCase):
    def test_each_draw_counts_how_often_each_held_out_line_is_drawn_with_replacement(self):
        K = intervals.counts(5, np.random.default_rng(0), 200)
        self.assertEqual(K.shape, (200, 5))
        self.assertTrue(np.all(K.sum(axis=1) == 5))
        self.assertTrue(np.array_equal(K, intervals.counts(5, np.random.default_rng(0), 200)))

    def test_change_from_mean_is_cells_minus_mean_per_component_in_percent_of_the_baseline_total(self):
        B = np.array([[2., 2.], [1., 1.], [1., 1.]])        # (total, common, contrast) x held-out lines
        mean = np.array([[2., 2.], [1., 1.], [1., 1.]])
        cells = np.array([[1., 2.], [0., 1.], [1., 1.]])
        got = intervals.change(mean, cells, B, np.ones((1, 2)))
        self.assertEqual(got[0].tolist(), [-25.])          # common: 100 (1 - 2) / 4
        self.assertEqual(got[1].tolist(), [0.])

    def test_a_change_proportional_to_the_baseline_total_in_every_line_has_a_degenerate_interval(self):
        rng = np.random.default_rng(1)
        B = rng.uniform(1, 2, (3, 14))
        mean = rng.uniform(1, 2, (3, 14))
        cells = mean - .02 * B[0]
        K = intervals.counts(14, rng, 500)
        got = intervals.change(mean, cells, B, K)
        self.assertTrue(np.allclose(got, -2.))


class LineSums(unittest.TestCase):
    def row(self, kind, seed, level, line, total, model='cpa'):
        return dict(fold='F1', model=model, seed=seed, level=level, kind=kind, unit=line, total=total, common=total / 2,
                    contrast=total / 2)

    def test_a_fit_reads_its_model_rows_only_averaged_over_seeds_and_lines_in_sorted_order(self):
        rows = [self.row('baseline', '', '', line, t) for line, t in (('b', 4.), ('a', 2.))]
        for level in ('L1', 'L4'):
            for seed, t in (('31002', 1.), ('31003', 3.)):
                rows += [self.row('model', seed, level, line, t) for line in ('b', 'a')]
                rows += [self.row('fit_anchor', seed, level, line, 99.) for line in ('b', 'a')]   # after the model rows
        mean, cells, baseline = intervals.fold_sums(rows)[('F1', 'cpa')]
        self.assertEqual(mean[0].tolist(), [2., 2.])          # (1 + 3) / 2 per line; the anchors do not enter
        self.assertEqual(cells.tolist(), mean.tolist())
        self.assertEqual(baseline[0].tolist(), [2., 4.])      # lines a, b


if __name__ == '__main__':
    unittest.main()
