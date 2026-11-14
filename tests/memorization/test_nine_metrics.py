"""The nine metrics' and the arm summaries' definitions on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_nine_metrics.py
"""
import itertools
import unittest

import numpy as np
import pandas as pd

from scsailor.resolution import metrics as M
from reproduce.memorization import metrics
from reproduce.origin_resolution import SEEDS
from reproduce.origin_resolution.other_metrics import references

PLATES = ('plate6', 'plate14')
ARMS = ('original', 'removal', 'constant', 'readers', 'calibrated', 'calibrated + readers')


def units(lines=4, drugs=3, genes=60, seed=0):
    """Every drug on every context (a line on one of the two plates), truth well above zero, random weights."""
    rng = np.random.default_rng(seed)
    contexts = [f'L{i}|{p}' for i in range(lines) for p in PLATES]
    context = np.repeat(contexts, drugs)
    n = len(context)
    return {'truth': rng.uniform(2, 3, (n, genes)), 'context': context,
            'p_code': np.tile(np.arange(drugs), len(contexts)), 'weight': rng.uniform(.5, 1.5, n)}


def reference(u, seed=0):
    """The metrics' reference on ``u`` with a random control vector per context and training mean response."""
    rng = np.random.default_rng(seed)
    g = u['truth'].shape[1]
    bags = {c: (C, C) for c, C in ((c, rng.uniform(1, 2, g)) for c in np.unique(u['context']))}
    return references({'u': u}, {'bags': {'heldout': bags}}, 'heldout', rng.normal(0, .1, g))


def weighted_mse(P, u):
    return float(np.average(np.mean((np.maximum(P, 0) - u['truth']) ** 2, axis=1), weights=u['weight']))


def rows(table, **match):
    return [r for r in table if all(r[k] == v for k, v in match.items())]


class NineMetrics(unittest.TestCase):
    def read(self, preds, u):
        """The case's values and gains rows, with a design-aware baseline of 2.5 everywhere."""
        ref = reference(u)
        lines = metrics.case_lines(u, ref, np.full_like(u['truth'], 2.5), lambda s: preds[s])
        return metrics.case_values({'fold': 'F1'}, lines), metrics.case_gains({'fold': 'F1'}, lines, ref['n'])

    def noisy(self, seed=2):
        """Every arm of every seed: the truth plus noise, the readers' smaller than the original's and the calibrated
        prediction's."""
        u = units(seed=1)
        rng = np.random.default_rng(seed)
        sd = {'original': .4, 'removal': .35, 'constant': .3, 'readers': .2, 'calibrated': .3,
              'calibrated + readers': .15}
        return u, {s: {arm: u['truth'] + rng.normal(0, sd[arm], u['truth'].shape) for arm in ARMS} for s in SEEDS}

    def test_a_point_per_seed_then_the_mean_of_the_seeds(self):
        u, preds = self.noisy()
        values, _ = self.read(preds, u)
        self.assertEqual([r['seed'] for r in rows(values, prediction='original')], list(SEEDS) + [metrics.SEED_MEAN])
        for s in SEEDS:
            self.assertAlmostEqual(rows(values, prediction='original', seed=s)[0]['mse'],
                                   weighted_mse(preds[s]['original'], u))
        for m in metrics.METRICS:
            seeds = [r[m] for r in rows(values, prediction='readers') if r['seed'] != metrics.SEED_MEAN]
            self.assertAlmostEqual(rows(values, prediction='readers', seed=metrics.SEED_MEAN)[0][m], np.mean(seeds))
        baseline = rows(values, prediction='design-aware baseline')
        self.assertTrue(all(r['mse'] == baseline[0]['mse'] for r in baseline))

    def test_mse_gain_is_in_percent_of_the_originals_total_and_other_gains_are_first_minus_second(self):
        u, preds = self.noisy()
        values, gains = self.read(preds, u)
        self.assertEqual(list(dict.fromkeys(r['contrast'] for r in gains)), list(metrics.CONTRASTS))
        for s in SEEDS:
            point = {p: rows(values, prediction=p, seed=s)[0] for p in ('original', 'calibrated', 'readers')}
            for contrast, second in (('readers - original', 'original'), ('readers - calibrated', 'calibrated')):
                mse = rows(gains, contrast=contrast, metric='mse', seed=s)[0]['gain']
                expect = 100 * (point[second]['mse'] - point['readers']['mse']) / point['original']['mse']
                self.assertAlmostEqual(mse, expect)
                self.assertGreater(mse, 0)
                self.assertAlmostEqual(rows(gains, contrast=contrast, metric='pearson_delta', seed=s)[0]['gain'],
                                       point['readers']['pearson_delta'] - point[second]['pearson_delta'])

    def test_a_gain_equal_on_every_unit_has_a_zero_width_interval(self):
        # the readers lower every unit's squared error by the same amount: resampling lines pairs the two predictions,
        # so every draw gives the same gain
        u = units(seed=3)
        rng = np.random.default_rng(3)
        a = rng.uniform(1, 2, len(u['truth']))[:, None]
        sign = rng.choice([-1., 1.], u['truth'].shape)
        original = u['truth'] + 0.1 * np.sqrt(a) * sign
        better = u['truth'] + 0.1 * np.sqrt(a - .5) * sign
        preds = {s: {arm: better if arm == 'readers' else original for arm in ARMS} for s in SEEDS}
        _, gains = self.read(preds, u)
        for r in rows(gains, metric='mse', contrast='readers - original'):
            self.assertAlmostEqual(r['ci90_low'], r['gain'], places=9)
            self.assertAlmostEqual(r['ci90_high'], r['gain'], places=9)

    def test_intervals_resample_lines_with_the_same_draws_for_every_seed(self):
        u, preds = self.noisy()
        _, gains = self.read(preds, u)
        ref = reference(u)
        K = metrics.draws(ref['n'])
        self.assertEqual(K.shape, (metrics.RESAMPLES, ref['n']))
        self.assertTrue(np.array_equal(K.sum(axis=1), np.full(metrics.RESAMPLES, ref['n'])))
        boots = []
        for s in SEEDS:
            line = {p: M.per_line(M.unit_values(preds[s][p], u, ref), u, ref)['pearson_delta']
                    for p in ('readers', 'original')}
            resampled = {p: (K @ x[0]) / (K @ x[1]) for p, x in line.items()}
            boots.append(resampled['readers'] - resampled['original'])
            r = rows(gains, contrast='readers - original', metric='pearson_delta', seed=s)[0]
            self.assertTrue(np.allclose([r['ci90_low'], r['ci90_high']], np.quantile(boots[-1], [.05, .95])))
        r = rows(gains, contrast='readers - original', metric='pearson_delta', seed=metrics.SEED_MEAN)[0]
        self.assertTrue(np.allclose([r['ci90_low'], r['ci90_high']], np.quantile(np.mean(boots, axis=0), [.05, .95])))


class Summary(unittest.TestCase):
    def test_counts_the_cases_whose_seed_mean_gain_is_above_zero(self):
        gains = pd.DataFrame({'fold': ['F1'] * 4 + ['F2'] * 4, 'contrast': 'readers - original', 'metric': 'mse',
                              'seed': (list(SEEDS) + [metrics.SEED_MEAN]) * 2,
                              'gain': [1., -1., 2., -.5, 3., 2., 1., 2.]})
        out = metrics.summary(gains)
        self.assertEqual(out.to_dict('records'), [{'contrast': 'readers - original', 'metric': 'mse', 'n': 2,
                                                   'improved': 1}])


def four_arms_table(rng):
    """A +Var four-arms table on calibration-supported units: every model, fold, seed and arm the summaries read, plus a
    row on all held-out units that they leave out."""
    out = []
    for m, f, s, a in itertools.product(('Attention', 'DeepSets', 'Summary-statistics', 'State'), ('F1', 'F2', 'F3'),
                                        SEEDS, ARMS):
        common, contrast = rng.uniform(40, 60), rng.uniform(40, 60)
        out.append({'fold': f, 'model': m, 'level': '+Var', 'seed': s, 'units': 'calibration-supported', 'arm': a,
                    'total_points': common + contrast, 'common_points': common, 'contrast_points': contrast})
    out.append({**out[0], 'units': 'all held-out', 'total_points': 1e6})
    return pd.DataFrame(out)


def six(x):
    """A value as its table is written: six significant digits."""
    return float(f'{x:.6g}')


class ArmSummaries(unittest.TestCase):
    def test_absolute_values_are_the_mean_over_cases_of_seed_means_read_as_written(self):
        rng = np.random.default_rng(4)
        table = four_arms_table(rng)
        cases = [('F1', 'Attention'), ('F2', 'DeepSets')]
        values = pd.DataFrame([{'fold': f, 'model': m, 'level': '+Var', 'units': 'calibration-supported', 'seed': s,
                                'prediction': p, **{k: rng.uniform(0, 1) for k in metrics.METRICS}}
                               for f, m in cases for p in metrics.PREDICTIONS for s in [*SEEDS, metrics.SEED_MEAN]])
        table = table[[(f, m) in cases for f, m in zip(table.fold, table.model)]]
        out = metrics.absolute_values(table, values).set_index('prediction')

        def expect(rows, column):
            case = lambda f, m: rows[(rows.fold == f) & (rows.model == m)][column]   # noqa: E731
            return np.mean([np.mean([six(x) for x in case(f, m)]) for f, m in cases])

        self.assertEqual(list(out.index), list(metrics.PREDICTIONS))
        self.assertEqual(out.loc['design-aware baseline', 'mse_pct_of_design_aware'], 100.)
        seeds = table[table.units == 'calibration-supported']
        self.assertAlmostEqual(out.loc['readers', 'mse_pct_of_design_aware'],
                               expect(seeds[seeds.arm == 'readers'], 'total_points'), places=12)
        v = values[values.seed != metrics.SEED_MEAN]
        self.assertAlmostEqual(out.loc['calibrated + readers', 'pearson_delta'],
                               expect(v[v.prediction == 'calibrated + readers'], 'pearson_delta'), places=12)
        self.assertEqual((out.loc['original', 'cases'], out.loc['original', 'seeds']), (2, 3))

    def test_arm_changes_are_seed_means_of_each_parts_change_in_percent_of_the_original_total(self):
        table = four_arms_table(np.random.default_rng(5))
        out = metrics.arm_changes(table)
        self.assertEqual(len(out), 4 * 3 * len(metrics.CHANGED_ARMS))
        # the parts add up to the total, to the six significant digits the table is read at
        parts = out.common_pct_of_original + out.contrast_pct_of_original
        self.assertTrue(np.allclose(parts, out.total_pct_of_original, atol=1e-3))
        r = out[(out.model == 'DeepSets') & (out.fold == 'F2') & (out.arm == 'calibrated')].iloc[0]
        t = table[(table.model == 'DeepSets') & (table.fold == 'F2') & (table.units == 'calibration-supported')]
        t = t.set_index(['seed', 'arm'])
        written = lambda s, arm, column: six(t.loc[(s, arm), column])   # noqa: E731
        self.assertAlmostEqual(r.common_pct_of_original, np.mean(
            [100 * (written(s, 'calibrated', 'common_points') - written(s, 'original', 'common_points'))
             / written(s, 'original', 'total_points') for s in SEEDS]), places=12)


if __name__ == '__main__':
    unittest.main()
