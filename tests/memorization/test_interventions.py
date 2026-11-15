"""The definitions of the target interventions' table and of the pull on the training lines, on small hand-made arrays;
no study data.

    python -m pytest tests/memorization/test_interventions.py
"""
import unittest

import numpy as np
import pandas as pd

from reproduce.memorization import data, fits, interventions

PLATES = ('plate6', 'plate14')


class Files(unittest.TestCase):
    def test_original_recipe_is_the_same_machine_run_when_there_is_one_else_the_ladder_fit(self):
        self.assertEqual(interventions.run_file('original', 'summary', 31002), fits.tahoe('original', 'summary', 31002))
        self.assertEqual(interventions.run_file('original', 'summary', 31003), data.ladder_file('F1', 'summary', 31003))
        self.assertEqual(interventions.run_file('original', 'attention_model', 31004),
                         data.ladder_file('F1', 'attention_model', 31004))
        self.assertEqual(interventions.run_file('original', 'deepsets', 31004),
                         fits.tahoe('original', 'deepsets', 31004))

    def test_an_interventions_run_is_its_designs_file(self):
        self.assertEqual(interventions.run_file('input-relative', 'summary', 31003),
                         fits.tahoe('input-relative', 'summary', 31003))
        self.assertEqual(interventions.run_file('readable-only', 'attention_model', 31004),
                         fits.tahoe('readable-only', 'attention_model', 31004))


class SeedMeans(unittest.TestCase):
    def test_seed_means_are_the_means_of_the_seeds_values_as_written(self):
        rows = pd.DataFrame({'design': ['original'] * 3 + ['input-relative'] * 3, 'model': 'DeepSets',
                             'seed': [31002, 31003, 31004] * 2,
                             'value': [1.00000041, 1.00000041, 1.00000041, 1., 2., 4.]})
        got = interventions.with_seed_means(rows, ['design', 'model'], ['value'])
        means = got[got.seed == data.SEED_MEAN].set_index('design').value
        # each value is written with six significant digits (1.00000041 as 1), so the mean is 1, not 1.00000041
        self.assertEqual(means['original'], 1.)
        self.assertAlmostEqual(means['input-relative'], 7 / 3)

    def test_each_seed_mean_row_follows_its_seeds(self):
        rows = pd.DataFrame({'design': ['original'] * 2 + ['readable-only'] * 2, 'model': 'Attention',
                             'seed': [31002, 31003] * 2, 'value': [1., 2., 3., 4.]})
        got = interventions.with_seed_means(rows, ['design', 'model'], ['value'])
        self.assertEqual(list(got.seed.astype(str)), ['31002', '31003', data.SEED_MEAN] * 2)
        self.assertEqual(list(got.design), ['original'] * 3 + ['readable-only'] * 3)


class TrainingPull(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.contexts = np.array([f'L{i}|{p}' for i in range(5) for p in PLATES])
        self.level = rng.normal(size=(10, 3))   # each training context's pool mean on three growth genes
        self.mix = np.array([.5, .25, .25])
        on = data.plates(self.contexts)
        own = self.level @ self.mix
        self.gap = np.array([own[on == p].mean() for p in on]) - own   # the plate's mean level minus the context's
        self.shift = np.where(on == 'plate6', 3., -1.)[:, None]   # a constant per plate does not enter

    def test_pull_is_the_slope_of_the_centred_program_level_correction_on_the_centred_gap(self):
        correction = .5 * (self.level.mean(axis=0) - self.level) + self.shift
        targets = -(self.level.mean(axis=0) - self.level) - 2 * self.shift
        got = interventions.training_pull(correction, targets, self.level, self.contexts, self.mix)
        self.assertAlmostEqual(got['pull_slope'], .5)
        self.assertAlmostEqual(got['pull_r'], 1.)
        self.assertAlmostEqual(got['targets_pull_slope'], -1.)
        self.assertAlmostEqual(got['targets_pull_r'], -1.)
        self.assertAlmostEqual(got['network_targets_r'], -1.)

    def test_pull_reads_the_correction_at_program_level(self):
        # a correction whose program-level value (mix-weighted) is twice the gap, spread unevenly over the genes
        correction = np.outer(2 * self.gap, [1., 2., 2.]) / (self.mix @ [1., 2., 2.])
        rng = np.random.default_rng(1)
        targets = rng.normal(size=(10, 3))
        got = interventions.training_pull(correction, targets, self.level, self.contexts, self.mix)
        self.assertAlmostEqual(got['pull_slope'], 2.)
        on = data.plates(self.contexts)
        a = targets @ self.mix
        a = a - np.array([a[on == p].mean() for p in on])
        c = self.gap - np.array([self.gap[on == p].mean() for p in on])
        self.assertAlmostEqual(got['targets_pull_slope'], np.sum(a * c) / np.sum(c * c))
        self.assertAlmostEqual(got['targets_pull_r'], np.corrcoef(a, c)[0, 1])


if __name__ == '__main__':
    unittest.main()
