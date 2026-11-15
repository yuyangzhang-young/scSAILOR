"""The definitions of the published models' retraining tables on small hand-made arrays; no study data: the layout of
the files of the fits and the gains of the two-phase run over the original on the nine metrics.

    python -m pytest tests/memorization/test_published_tables.py
"""
import unittest

import numpy as np

from reproduce.memorization import data, fits, metrics
from reproduce.memorization import published_tables as pt
from reproduce.origin_resolution import SEEDS
from reproduce.origin_resolution.probes import STATE_FIT

LINES = 5


def pair(values, weights):
    """A ``per_line`` pair: each line's weighted sum and weight."""
    return np.asarray(values, float) * np.asarray(weights, float), np.asarray(weights, float)


class FitFiles(unittest.TestCase):
    def test_same_machine_refits_are_the_original_design_of_the_retraining_runs(self):
        self.assertEqual(pt.fit_file('same-machine refits', 'state', 31003, 'L2', fold='F2'),
                         fits.tahoe('original', 'state', 31003, 'F2', level='L2'))

    def test_target_interventions_and_two_phase_runs_keep_their_design_name(self):
        self.assertEqual(pt.fit_file('two-phase', 'prepr', 31004, 'L1', task='c'),
                         fits.tahoe('two-phase', 'prepr', 31004, task='c'))

    def test_original_fits_are_the_ladder_at_mean_and_states_fold_one_fits_at_plus_var(self):
        self.assertEqual(pt.fit_file('original fits', 'cpa', 31002, 'L1'), data.ladder_file('F1', 'cpa', 31002))
        self.assertEqual(pt.fit_file('original fits', 'state', 31002, 'L2'),
                         STATE_FIT / 'st_anchored_train28_L2_31002/PREDICTIONS_step5000.npz')


class TwoPhaseGains(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(1, 2, LINES)
        self.lines = {s: {k: {m: pair(rng.uniform(0, 1, LINES), w) for m in metrics.METRICS}
                          for k in ('original', 'original + readers', 'two-phase + readers')} for s in SEEDS}
        self.rows = metrics.case_gains({'fold': 'F1'}, self.lines, LINES,
                                       {pt.TWO_PHASE: ('two-phase + readers', 'original + readers')})

    def row(self, metric, seed):
        (r,) = [r for r in self.rows if r['metric'] == metric and r['seed'] == seed]
        return r

    def test_mse_gain_is_the_fall_of_the_error_in_percent_of_the_original_total(self):
        s = SEEDS[0]
        x = self.lines[s]
        expected = 100 * (x['original + readers']['mse'][0].sum() / x['original + readers']['mse'][1].sum()
                          - x['two-phase + readers']['mse'][0].sum() / x['two-phase + readers']['mse'][1].sum()) \
            / (x['original']['mse'][0].sum() / x['original']['mse'][1].sum())
        self.assertAlmostEqual(self.row('mse', s)['gain'], expected)

    def test_other_metrics_gain_is_the_plain_difference_first_minus_second(self):
        s = SEEDS[1]
        x = self.lines[s]
        expected = (x['two-phase + readers']['cosine_pds'][0].sum() / x['two-phase + readers']['cosine_pds'][1].sum()
                    - x['original + readers']['cosine_pds'][0].sum() / x['original + readers']['cosine_pds'][1].sum())
        self.assertAlmostEqual(self.row('cosine_pds', s)['gain'], expected)

    def test_seed_mean_row_averages_the_seeds_and_intervals_are_ordered(self):
        for m in metrics.METRICS:
            mean = self.row(m, data.SEED_MEAN)['gain']
            self.assertAlmostEqual(mean, np.mean([self.row(m, s)['gain'] for s in SEEDS]))
        self.assertEqual(len(self.rows), len(metrics.METRICS) * (len(SEEDS) + 1))
        self.assertEqual({r['contrast'] for r in self.rows}, {pt.TWO_PHASE})
        for r in self.rows:
            self.assertLessEqual(r['ci90_low'], r['ci90_high'])


if __name__ == '__main__':
    unittest.main()
