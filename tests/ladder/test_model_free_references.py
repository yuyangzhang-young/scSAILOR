"""The model-free references of Supplementary Note S5.3 on small hand-made arrays; no study data.

    python -m pytest tests/ladder/test_model_free_references.py
"""
import unittest

import numpy as np
import pandas as pd

from reproduce.ladder import references


class NestedSplit(unittest.TestCase):
    def test_the_sorted_training_lines_are_permuted_with_the_fixed_seed_and_the_first_three_fitted(self):
        # Jiang et al. fold 1 holds out A549; its split as the paper's references used it
        fit, calibration = references.nested_split({'MCF7', 'HAP1', 'K562', 'HT29', 'BXPC3'}, 3)
        self.assertEqual((fit, calibration), (['K562', 'HT29', 'BXPC3'], ['HAP1', 'MCF7']))


class Predictions(unittest.TestCase):
    def setUp(self):
        # two training lines (x, y) in two strata (s, t); one perturbation; the pool mean is zero
        self.case = {
            'train': {'truth': np.array([[1.], [3.], [5.], [7.]]), 'context': np.array(['x|s', 'x|t', 'y|s', 'y|t']),
                      'p_code': np.zeros(4, int), 'weight': np.full(4, .25)},
            'bags': {'train': {c: (np.zeros(1), np.zeros(1)) for c in ('x|s', 'x|t', 'y|s', 'y|t')},
                     'heldout': {'z|s': (np.array([10.]), np.zeros(1)), 'z|t': (np.array([20.]), np.zeros(1))}},
            'line_of': lambda c: c.split('|')[0], 'stratum_of': lambda c: c.split('|')[1]}
        self.held = {'truth': np.zeros((2, 1)), 'context': np.array(['z|s', 'z|t']), 'p_code': np.zeros(2, int),
                     'weight': np.full(2, .5)}

    def test_no_change_baseline_and_stratum_mean_add_the_overall_and_the_stratum_training_response(self):
        b, a, A = references.predictions(self.case, self.held, 'heldout', ['x', 'y'])
        self.assertEqual(b.ravel().tolist(), [10., 20.])
        self.assertEqual(a.ravel().tolist(), [14., 24.])          # overall response (1 + 3 + 5 + 7) / 4
        self.assertEqual(A.ravel().tolist(), [13., 25.])          # stratum s: (1 + 5) / 2; t: (3 + 7) / 2

    def test_training_responses_come_from_the_given_lines_only(self):
        b, a, A = references.predictions(self.case, self.held, 'heldout', ['y'])
        self.assertEqual(a.ravel().tolist(), [16., 26.])
        self.assertEqual(A.ravel().tolist(), [15., 27.])


class Models(unittest.TestCase):
    def test_a_model_minus_the_strongest_reference_is_100_times_the_difference_of_relative_mses(self):
        refs = pd.DataFrame([dict(screen='Parse PBMC', fold=f, relative_mse=v, strongest=s)
                             for f, v, s in (('G1', .9, False), ('G1', .8, True), ('G2', .7, True))])
        points = pd.DataFrame([dict(screen='Parse PBMC', model='CPA', level='Cells', fold=f, relative_mse=v)
                               for f, v in (('G1', .85), ('G2', .75), ('fold mean', .8))])
        got = references.models_table(points, refs).set_index('fold')['minus_strongest']
        self.assertAlmostEqual(got['G1'], 5.)
        self.assertAlmostEqual(got['G2'], 5.)
        self.assertAlmostEqual(got['fold mean'], 5.)   # the mean of the folds' differences


if __name__ == '__main__':
    unittest.main()
