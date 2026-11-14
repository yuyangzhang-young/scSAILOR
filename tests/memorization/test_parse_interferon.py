"""The definitions of the interferon readings on Parse PBMC on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_parse_interferon.py
"""
import unittest

import numpy as np

from reproduce.memorization import parse_interferon as interferon


class Programs(unittest.TestCase):
    def test_programs_need_ten_panel_genes_and_average_them_equally(self):
        genes = np.array([f'G{i}' for i in range(30)])
        sets = {'BIG': {f'G{i}' for i in range(12)} | {'ABSENT'}, 'SMALL': {f'G{i}' for i in range(20, 29)}}
        got = interferon.program_weights(genes, sets)
        self.assertEqual(list(got), ['BIG'])
        self.assertTrue(np.allclose(got['BIG'], np.r_[np.full(12, 1 / 12), np.zeros(18)]))

    def test_control_level_is_relative_to_the_training_donors_of_the_same_cell_type(self):
        train = ['Donor1|A', 'Donor2|A', 'Donor1|B', 'Donor2|B']
        train_levels = np.array([1., 3., 10., 20.])
        got = interferon.relative_level(np.array([5., 5.]), ['Donor9|A', 'Donor9|B'], train_levels, train)
        self.assertTrue(np.allclose(got, [5 - 2, 5 - 15]))

    def test_between_donor_sd_is_the_sd_within_each_cell_type_averaged_over_cell_types(self):
        contexts = ['Donor1|A', 'Donor2|A', 'Donor3|A', 'Donor1|B', 'Donor2|B', 'Donor3|B']
        levels = np.array([[1.], [2.], [4.], [0.], [0.], [3.]])
        expected = (np.std([1, 2, 4], ddof=1) + np.std([0, 0, 3], ddof=1)) / 2
        self.assertTrue(np.allclose(interferon.between_donor_sd(levels, contexts), [expected]))


class ReferenceWells(unittest.TestCase):
    def test_a_cells_score_is_the_program_weighted_log1p_count_per_ten_thousand_on_panel_genes(self):
        # two cells: cell 0 has counts 2 (gene 0) and 3 (gene 5, off the panel); cell 1 has count 4 (gene 1)
        ptr, total = np.array([0, 2, 3]), np.array([10., 20.])
        gene, count = np.array([0, 5, 1]), np.array([2., 3., 4.])
        to_panel = np.array([1, 0, -1, -1, -1, -1])   # registry gene -> panel column, -1 off the panel
        weights = np.array([.5, .25])                  # per panel column
        got = interferon.cell_scores(ptr, total, gene, count, to_panel, weights, 0, 2)
        self.assertTrue(np.allclose(got, [np.log1p(2 * 1000) * .25, np.log1p(4 * 500) * .5]))

    def test_measured_change_is_the_weighted_cytokine_mean_minus_the_reference_cells_mean(self):
        z, w, scores = np.array([1., 3.]), np.array([1., 3.]), np.array([.5, 1.5, 1.])
        self.assertAlmostEqual(interferon.measured_change(z, w, scores), 2.5 - 1.)


if __name__ == '__main__':
    unittest.main()
