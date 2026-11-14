"""Supplementary Fig. 1's adapted weighted R2: the score of a prediction and the plotted differences, on small
hand-made values; no study data.

    python -m pytest tests/ladder/test_weighted_r2.py
"""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from reproduce.ladder import weighted_r2


class Score(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.u = {'truth': rng.uniform(0, 2, (6, 8)), 'context': np.repeat(['a', 'b'], 3), 'weight': np.full(6, 1 / 6)}

    def test_the_truth_scores_one_in_both_versions(self):
        self.assertTrue(np.allclose(list(weighted_r2.adapted_r2(self.u['truth'], self.u).values()), 1.))

    def test_the_contrast_version_ignores_a_shift_shared_by_a_contexts_units(self):
        shifted = self.u['truth'] + np.repeat([[.3], [.7]], 3, axis=0)
        got = weighted_r2.adapted_r2(shifted, self.u)
        self.assertLess(got['weighted_r2'], 1.)
        self.assertAlmostEqual(got['weighted_r2_contrast'], 1.)

    def test_units_alone_in_their_context_are_left_out_and_the_weights_renormalised(self):
        u = {k: np.concatenate([v, v[:1]]) for k, v in self.u.items()}
        u['context'][-1] = 'c'                      # a singleton context: its score is undefined
        prediction = u['truth'] + .1
        self.assertEqual(weighted_r2.adapted_r2(prediction, u), weighted_r2.adapted_r2(prediction[:-1], self.u))


class Plotted(unittest.TestCase):
    def test_fold_points_are_the_kept_seeds_mean_minus_the_fold_baseline_and_ticks_average_folds(self):
        fits = pd.DataFrame([dict(screen='S', model='CPA', fold=f, level='Mean', seed=s, fit_status=st, weighted_r2=v,
                                  weighted_r2_contrast=v / 2)
                             for f, s, st, v in (('F1', '31002', 'kept', .2), ('F1', '31003', 'kept', .4),
                                                 ('F1', '31004', 'excluded: outlier', 9.),
                                                 ('F2', '31002', 'kept', .1))])
        refs = {('S', f, 'baseline'): {'weighted_r2': v, 'weighted_r2_contrast': 0., 'units_left_out': 0}
                for f, v in (('F1', .1), ('F2', -.1))}
        with mock.patch.object(weighted_r2, 'reference_scores', return_value=refs):
            got = weighted_r2.plotted_table(fits).set_index(['version', 'fold'])
        self.assertAlmostEqual(got.loc[('Total', 'F1'), 'difference'], .2)
        self.assertEqual(got.loc[('Total', 'F1'), 'seeds'], '31002;31003')
        self.assertAlmostEqual(got.loc[('Total', 'F2'), 'difference'], .2)
        self.assertAlmostEqual(got.loc[('Total', 'fold mean'), 'difference'], .2)
        self.assertAlmostEqual(got.loc[('Contrast', 'F1'), 'difference'], .15)


if __name__ == '__main__':
    unittest.main()
