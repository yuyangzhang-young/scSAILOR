"""The plate-aware baseline's reductions and Supplementary Table 7's columns on small hand-made values; no study data.

    python -m pytest tests/ladder/test_plate.py
"""
import unittest

import numpy as np
import pandas as pd

from reproduce.ladder import plate


class Readings(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.u = {'truth': rng.uniform(0, 2, (6, 8)), 'context': np.repeat(['a|plate6', 'b|plate14'], 3),
                  'weight': np.full(6, 1 / 6)}

    def test_reductions_are_baseline_minus_prediction_per_component_in_percent_of_the_baseline_total(self):
        baseline = self.u['truth'] + 1.                       # all of its error is common
        got = plate.reductions(self.u['truth'] + .5, baseline, self.u)
        self.assertAlmostEqual(got['relative_mse'], .25)
        self.assertAlmostEqual(got['common_reduction'], 75.)
        self.assertAlmostEqual(got['contrast_reduction'], 0.)

    def test_predictions_are_clipped_at_zero_before_scoring(self):
        negative = -np.ones_like(self.u['truth'])
        got = plate.reductions(negative, np.zeros_like(negative), self.u)
        self.assertEqual(got['relative_mse'], 1.)


class SupplementaryTable7(unittest.TestCase):
    def test_models_give_cells_and_their_lowest_level_per_fold_and_the_fold_mean_reductions_at_cells(self):
        rows = [dict(screen='Tahoe-100M', model='CPA', fold=f, level=lv, relative_mse=v,
                     delta_common_points=-c, delta_contrast_points=-k)
                for f, lv, v, c, k in (('F1', 'Mean', .95, 1, 1), ('F1', 'Cells', .90, 2, 3),
                                       ('F2', 'Mean', .91, 1, 1), ('F2', 'Cells', .93, 4, 5))]
        rows += [dict(screen='Tahoe-100M', model='CPA', fold='fold mean', level=lv, relative_mse=v,
                      delta_common_points=-c, delta_contrast_points=-k) for lv, v, c, k in (('Mean', .93, 1, 1),
                                                                                              ('Cells', .915, 3, 4))]
        reference = pd.DataFrame([dict(fold='F1', relative_mse=.88, common_reduction=6., contrast_reduction=5.),
                                  dict(fold='F2', relative_mse=.89, common_reduction=5., contrast_reduction=6.)])
        got = plate.supp_table_7(pd.DataFrame(rows), reference).set_index('model')
        self.assertEqual(got.loc['Plate-aware baseline', 'relative_mse_F2'], .89)
        self.assertEqual(got.loc['Plate-aware baseline', 'common_reduction'], 5.5)
        self.assertEqual(got.loc['CPA', 'relative_mse_F2'], .93)
        self.assertEqual(got.loc['CPA', 'lowest_F1'], .90)
        self.assertEqual((got.loc['CPA', 'lowest_F2'], got.loc['CPA', 'lowest_level_F2']), (.91, 'Mean'))
        self.assertEqual(got.loc['CPA', 'contrast_reduction'], 4.)


if __name__ == '__main__':
    unittest.main()
