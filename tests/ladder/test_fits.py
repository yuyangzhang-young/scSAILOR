"""The per-fit table's readers and fit rules on small hand-made score files; no study data.

    python -m pytest tests/ladder/test_fits.py
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from reproduce.ladder import fits


def write_csv(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


class Scores(unittest.TestCase):
    def test_relative_mse_divides_totals_and_component_points_are_differences_in_percent_of_baseline_total(self):
        got = fits.score_columns((4., 1., 3.), (5., 2., 3.))
        self.assertEqual(got['relative_mse'], .8)
        self.assertEqual(got['delta_common_points'], -20.)
        self.assertEqual(got['delta_contrast_points'], 0.)
        self.assertEqual((got['model_mse_total'], got['baseline_mse_common']), (4., 2.))


class Readers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_per_seed_values_are_ratios_to_the_baseline_total_at_the_fixed_length(self):
        table = write_csv(self.dir / 'cpa.csv', [
            dict(level='L1', step=140000, seed=31002, total_mse=.5, common_error_mse=.25, difference_error_mse=.25),
            dict(level='L1', step=20000, seed=31002, total_mse=9., common_error_mse=9., difference_error_mse=9.),
            dict(level='L1', step=140000, seed=31009, total_mse=9., common_error_mse=9., difference_error_mse=9.)])
        baseline = self.dir / 'baseline.json'
        baseline.write_text(json.dumps({'baseline': {'total': 4., 'common': .75, 'd': .25}}))
        got = list(fits.per_seed(table, baseline, 140000))
        self.assertEqual(got, [('31002', 'L1', (2., 1., 1.), (4., 3., 1.), None)])

    def test_per_seed_reads_either_name_of_the_baseline_record(self):
        baseline = self.dir / 'baseline.json'
        baseline.write_text(json.dumps({'baseline': {'base_total': 2., 'base_common': .5, 'base_d': .5}}))
        self.assertEqual(fits.baseline_of(baseline), (2., 1., 1.))

    def test_per_fit_rows_give_model_and_anchor_and_the_training_ratio_for_one_label(self):
        row = {'model': 'attention', 'seed': '31005', 'level': 'L2', 'train_model_total_mse': '1.98',
               'train_anchor_total_mse': '2'}
        for who in ('model', 'anchor'):
            row |= {f'heldout_{who}_total_mse': '4' if who == 'model' else '5',
                    f'heldout_{who}_common_error_mse': '1', f'heldout_{who}_difference_error_mse': '3'}
        table = write_csv(self.dir / 'built.csv', [row, dict(row, model='summary'), dict(row, level='L0')])
        got = list(fits.per_fit(table, 'heldout', 'attention_model'))   # the file labels it 'attention'
        self.assertEqual(got, [('31005', 'L2', (4., 1., 3.), (5., 1., 3.), .99)])

    def test_unit_sums_give_each_fit_its_sum_over_donors_divided_by_their_mass_and_can_leave_donors_out(self):
        rows = [dict(fold='G1', model='cpa', seed='', level='', kind='baseline', unit=u, mass=m, total=t, common=c,
                     contrast=t - c) for u, m, t, c in (('D1', .5, 2., 1.), ('D2', .5, 4., 1.))]
        rows += [dict(fold='G1', model='cpa', seed='31002', level='L4', kind='model', unit=u, mass=m, total=t,
                      common=c, contrast=t - c) for u, m, t, c in (('D1', .5, 1., .5), ('D2', .5, 3., 1.))]
        rows += [dict(fold='G1', model='summary', seed='31002', level='L4', kind=k, unit='D1', mass=.5, total=t,
                      common=t, contrast=0.) for k, t in (('model', 1.), ('fit_anchor', 3.))]
        got = fits.unit_fits(rows)
        self.assertEqual(got[('G1', 'cpa', '31002', 'L4')], ((4., 1.5, 2.5), (6., 2., 4.)))
        self.assertEqual(got[('G1', 'summary', '31002', 'L4')], ((2., 2., 0.), (6., 6., 0.)))
        ten = fits.unit_fits(rows, leave_out={'D2'})
        self.assertEqual(ten[('G1', 'cpa', '31002', 'L4')], ((2., 1., 1.), (4., 2., 2.)))


class FitRules(unittest.TestCase):
    def fit(self, seed, ratio, training=None, level='L1'):
        return dict(screen='S', fold='F1', model='cpa', level=level, seed=seed, relative_mse=ratio,
                    training_ratio=training)

    def test_a_fit_whose_training_error_reaches_0_99_of_the_anchors_is_set_aside_first(self):
        group = [self.fit('31002', 1., .99), self.fit('31003', 1.1, .98), self.fit('31004', 1.2)]
        got = fits.fit_rules(group)
        self.assertTrue(got[0].startswith('excluded: training failed'))
        self.assertEqual(got[1:], ['kept', 'kept'])

    def test_a_fit_above_1_4_times_the_median_of_the_fits_that_passed_training_itself_included_is_an_outlier(self):
        group = [self.fit('31002', 1.), self.fit('31003', 1.2), self.fit('31004', 1.69)]   # median 1.2
        got = fits.fit_rules(group + [self.fit('31002', 9., level='L2')])
        self.assertEqual(got[:2], ['kept', 'kept'])
        self.assertEqual(got[2], 'excluded: outlier (1.41 x sibling median; rule: above 1.4 x the median of the '
                                 'kept sibling seeds)')
        self.assertEqual(got[3], 'kept')   # alone at its level: its own median


if __name__ == '__main__':
    unittest.main()
