"""The ladder's fold points, fold means and changes from Mean on a small hand-made per-fit table; no study data.

    python -m pytest tests/ladder/test_ladder.py
"""
import unittest

import pandas as pd

from reproduce.ladder import ladder


def fit(fold, level, seed, rel, common, contrast, status='kept'):
    return dict(screen='S', model='CPA', fold=fold, level=level, seed=seed, fit_status=status, relative_mse=rel,
                delta_common_points=common, delta_contrast_points=contrast)


class Summary(unittest.TestCase):
    def setUp(self):
        self.fits = pd.DataFrame([
            fit('F1', 'Mean', '31002', 1.0, 1., 2.), fit('F1', 'Mean', '31003', 2.0, 3., 4.),
            fit('F1', 'Cells', '31002', .5, -1., 0.), fit('F1', 'Cells', '31003', .7, -3., 2.),
            fit('F1', 'Cells', '31004', 9., 9., 9., 'excluded: outlier'),
            fit('F2', 'Mean', '31002', 3.0, 0., 0.), fit('F2', 'Cells', '31002', 2.0, -2., -4.)])

    def test_fold_points_average_the_kept_seeds_and_the_fold_mean_weights_folds_equally(self):
        got = ladder.summarize(self.fits).set_index(['fold', 'level'])
        self.assertEqual(got.loc[('F1', 'Mean'), 'relative_mse'], 1.5)
        self.assertEqual(got.loc[('F1', 'Cells'), 'relative_mse'], .6)
        self.assertEqual(got.loc[('F1', 'Cells'), 'seeds'], '31002;31003')
        self.assertEqual(got.loc[('fold mean', 'Mean'), 'relative_mse'], 2.25)
        self.assertEqual(got.loc[('fold mean', 'Cells'), 'relative_mse'], 1.3)

    def test_changes_from_mean_subtract_the_mean_levels_component_in_each_fold_then_average_folds(self):
        got = ladder.summarize(self.fits).set_index(['fold', 'level'])
        self.assertEqual(got.loc[('F1', 'Cells'), 'common_change_from_mean'], -4.)   # -2 - 2
        self.assertEqual(got.loc[('F2', 'Cells'), 'contrast_change_from_mean'], -4.)
        self.assertEqual(got.loc[('fold mean', 'Cells'), 'common_change_from_mean'], -3.)   # (-4 - 2) / 2
        self.assertEqual(got.loc[('F1', 'Mean'), 'common_change_from_mean'], 0.)


class Sensitivity(unittest.TestCase):
    def test_reinstating_an_outlier_returns_it_to_its_fold_point(self):
        rows = (('31002', 1., 'kept'), ('31003', 1., 'kept'), ('31004', 4., 'excluded: outlier (4.00 x ...)'))
        fits = pd.DataFrame([fit('F1', 'Mean', s, v, 0., 0., st) for s, v, st in rows])
        got = ladder.sensitivity_table(fits).iloc[0]
        self.assertEqual((got['relative_mse'], got['relative_mse_outliers_reinstated']), (1., 2.))


if __name__ == '__main__':
    unittest.main()
