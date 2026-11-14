"""The definitions of the line ranking by growth-program response on small hand-made arrays; no study data: the
selection of drug x program pairs on training lines, the line ranking and its plate reference, the pair-median growth
relation, the pull and the ranking without it, and the seed means read from the written tables.

    python -m pytest tests/memorization/test_ranking.py
"""
import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reproduce.memorization import ranking, selection

PLATES = ('plate6', 'plate14')


def design(lines, drugs):
    """Every drug on every line and plate: per unit its line, plate and drug code."""
    grid = [(f'L{i:02d}', p, q) for i in range(lines) for p in PLATES for q in range(drugs)]
    line, plate, code = (np.array(x) for x in zip(*grid))
    return line, plate, code


class Selection(unittest.TestCase):
    def test_weights_read_each_programs_mean_of_its_genes(self):
        masks = [np.array([1, 1, 0, 0, 1], bool), np.array([0, 1, 1, 0, 0], bool)]
        X = np.arange(10.).reshape(2, 5)
        got = X @ selection.weights(masks)
        self.assertTrue(np.allclose(got, np.stack([X[:, m].mean(axis=1) for m in masks], axis=1)))

    def test_spearman_by_column_with_ties_and_none_for_a_constant_column(self):
        rng = np.random.default_rng(0)
        A, B = rng.integers(0, 4, (9, 3)).astype(float), rng.normal(size=(9, 3))
        A[:, 2] = 1.
        got = selection.spearman_cols(A, B)
        for j in range(2):
            self.assertAlmostEqual(got[j], spearmanr(A[:, j], B[:, j])[0])
        self.assertTrue(np.isnan(got[2]))

    def test_a_drug_is_read_on_its_lines_with_both_plates_if_they_are_at_least_five(self):
        line, plate, code = design(lines=6, drugs=2)
        values = np.arange(len(line), dtype=float)[:, None]
        drop = (code == 1) & (plate == 'plate14') & np.isin(line, ['L00', 'L01'])   # drug 1: four lines on both plates
        got = selection.by_plate(values[~drop], line[~drop], plate[~drop], code[~drop])
        self.assertEqual(list(got), [0])
        lines, six, fourteen = got[0]
        self.assertEqual(lines, [f'L{i:02d}' for i in range(6)])
        rows = {(a, b): v for a, b, c, v in zip(line, plate, code, values[:, 0]) if c == 0}
        self.assertEqual(list(six[:, 0]), [rows[(x, 'plate6')] for x in lines])
        self.assertEqual(list(fourteen[:, 0]), [rows[(x, 'plate14')] for x in lines])

    def test_training_readings_reliability_between_plates_and_level_relation_within_plates(self):
        rng = np.random.default_rng(1)
        line, plate, code = design(lines=8, drugs=1)
        level = rng.normal(size=(len(line), 2))
        response = rng.normal(size=(len(line), 2))
        reliability, relation = selection.training_readings(response, level, line, plate, code)[0]
        six, fourteen = plate == 'plate6', plate == 'plate14'
        rho = lambda a, b: spearmanr(a, b)[0]   # noqa: E731  (units are in line order on each plate)
        for j in range(2):
            self.assertAlmostEqual(reliability[j], rho(response[six, j], response[fourteen, j]))
            self.assertAlmostEqual(relation[j], (rho(level[six, j], response[six, j])
                                                 + rho(level[fourteen, j], response[fourteen, j])) / 2)

    def test_pairs_selected_at_both_thresholds_inclusive_among_the_drugs_read(self):
        readings = {0: (np.array([.6, .6, .59]), np.array([.4, .39, .9])),
                    2: (np.array([.9, np.nan, .8]), np.array([.9, .9, np.nan])),
                    3: (np.array([.9, .9, .9]), np.array([.9, .9, .9]))}
        self.assertEqual(selection.selected(readings, drugs=[2, 0, 1]), [(0, 0), (2, 0)])


class LineRanking(unittest.TestCase):
    def test_line_response_is_the_mean_over_the_pairs_on_each_plate(self):
        line, plate, code = design(lines=3, drugs=3)
        R = np.random.default_rng(2).normal(size=(len(line), 2))
        pairs = [(0, 1), (2, 0)]
        mean = ranking.line_means(R, pairs, line, plate, code)
        self.assertEqual(list(mean.columns), list(PLATES))
        for x in mean.index:
            for p in PLATES:
                expected = np.mean([R[(line == x) & (plate == p) & (code == q), j][0] for q, j in pairs])
                self.assertAlmostEqual(mean.loc[x, p], expected)

    def test_ranking_is_one_plates_prediction_against_the_other_plates_measurement_both_ways(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(9, 2))
        Y = np.column_stack([-X[:, 1], X[:, 0]])   # plate 6 measured = -prediction from plate 14
        everything = np.arange(9)
        self.assertAlmostEqual(ranking.cross(X, Y, everything), 0.)
        self.assertAlmostEqual(ranking.cross(X, np.column_stack([X[:, 1], X[:, 0]]), everything), 1.)
        self.assertAlmostEqual(ranking.replicate(Y, everything), spearmanr(Y[:, 0], Y[:, 1])[0])

    def test_pair_median_growth_relation_crosses_plates_per_pair_both_ways_then_takes_the_median(self):
        rng = np.random.default_rng(5)
        lines = [f'L{i}' for i in range(6)]

        def read():   # drug code -> (lines, first plate's rows, second plate's rows), two programs
            return {q: (lines, rng.normal(size=(6, 2)), rng.normal(size=(6, 2))) for q in (3, 5)}

        predicted, measured = read(), read()
        pairs = [(3, 0), (3, 1), (5, 1)]
        rho = lambda a, b: spearmanr(a, b)[0]   # noqa: E731
        per_pair = [(rho(predicted[q][1][:, j], measured[q][2][:, j])
                     + rho(predicted[q][2][:, j], measured[q][1][:, j])) / 2 for q, j in pairs]
        self.assertAlmostEqual(ranking.pair_median(predicted, measured, pairs), float(np.median(per_pair)))
        self.assertNotAlmostEqual(np.median(per_pair), np.mean(per_pair))

    def test_interval_from_2000_line_resamples_of_the_fixed_generator(self):
        draws = np.random.default_rng(0).integers(0, 7, size=(2000, 7))
        values = np.arange(7.) ** 2
        low, high = ranking.interval(lambda i: values[i].mean(), 7)
        self.assertEqual((low, high), tuple(np.percentile(values[draws].mean(axis=1), [2.5, 97.5])))


class Pull(unittest.TestCase):
    def test_gap_is_the_same_plate_training_level_minus_the_contexts_own(self):
        train = ['A|plate6', 'A|plate14', 'B|plate6', 'B|plate14', 'C|plate6', 'C|plate14']
        level = np.arange(12.).reshape(6, 2)
        held = ['H|plate14', 'H|plate6']
        own = np.array([[1., 1.], [2., 2.]])
        got = ranking.level_gap(level, train, own, held)
        self.assertTrue(np.allclose(got, [level[1::2].mean(0) - 1, level[0::2].mean(0) - 2]))

    def test_slope_of_the_correction_on_the_gap_at_program_level_after_plate_centring(self):
        rng = np.random.default_rng(4)
        plates = np.array(PLATES * 5)
        gap = rng.normal(size=(10, 4))
        mix = np.array([.5, .5, 0, 0])
        shift = np.where(plates == 'plate6', 3., -1.)[:, None]
        correction = 2 * gap + shift + np.array([0, 0, 7, -7.])   # off-program genes do not enter
        self.assertAlmostEqual(ranking.pull_slope(correction, gap, mix, plates), 2.)

    def test_without_pull_takes_slope_times_the_contexts_gap_out_of_the_growth_genes(self):
        P = np.arange(12.).reshape(4, 3)
        gap = np.array([[1., 2.], [10., 20.]])   # two contexts x the two growth genes
        G = np.array([True, False, True])
        got = ranking.without_pull(P, .5, gap, G, np.array([0, 1, 1, 0]))
        self.assertTrue(np.array_equal(got, [[-.5, 1, 1], [-2, 4, -5], [1, 7, -2], [8.5, 10, 10]]))
        self.assertTrue(np.array_equal(P, np.arange(12.).reshape(4, 3)))   # the prediction itself is kept

    def test_program_mix_weights_the_growth_genes_as_the_pairs_mix_their_programs(self):
        W = np.array([[.5, 0], [.5, .5], [0, .5], [0, 0]])
        G = np.array([True, True, True, False])
        got = selection.program_mix(W, G, [(0, 0), (1, 0), (2, 1)])
        self.assertTrue(np.allclose(got, W[G] @ np.array([2 / 3, 1 / 3])))


class Tables(unittest.TestCase):
    def tables(self):
        """A growth ranking table (the plate reference and one model's three seeds of two predictors) and a pull
        table, on one fold."""
        reference = {'fold': 'F1', 'lines': 14, 'pairs': 2, 'model': ranking.PLATE_REFERENCE, 'level': '',
                     'predictor': '', 'seed': '', 'lines_spearman': .8, 'low': .5, 'high': .9}
        value = {'original': (.1234564, .2, .3), 'readers': (.4, .5, .6)}
        growth_ranking = pd.DataFrame([reference] + [
            {**reference, 'model': 'CPA', 'level': 'Mean', 'predictor': p, 'seed': s, 'lines_spearman': v,
             'low': np.nan, 'high': np.nan} for p, vs in value.items() for s, v in zip((31002, 31003, 31004), vs)])
        pull = pd.DataFrame({'fold': 'F1', 'model': 'CPA', 'level': 'Mean', 'seed': [31002, 31003, 31004],
                             'pull_slope': [.1, .2, .3000004], 'ranking_without_pull': [.7, .8, .9]})
        return growth_ranking, pull

    def test_seed_means_of_the_rankings_and_pull_as_written_beside_the_plate_reference(self):
        t = ranking.line_ranking_table(*self.tables())
        cpa = t[t.model == 'CPA'].iloc[0]
        self.assertAlmostEqual(cpa.original, (.123456 + .2 + .3) / 3, places=15)
        self.assertAlmostEqual(cpa.readers, .5, places=15)
        self.assertAlmostEqual(cpa.pull_slope, (.1 + .2 + .3) / 3, places=15)
        self.assertAlmostEqual(cpa.without_pull, .8, places=15)
        data = t[t.model == ranking.PLATE_REFERENCE].iloc[0]
        self.assertEqual((data.data, data.data_low, data.data_high), (.8, .5, .9))
        self.assertTrue(np.isnan(cpa.data) and np.isnan(data.original))

    def test_fold_means_of_the_per_fold_rankings(self):
        t = pd.DataFrame({'fold': ['F1', 'F1', 'F2', 'F2'], 'model': [ranking.PLATE_REFERENCE, 'CPA'] * 2,
                          'level': ['', 'Mean'] * 2, 'pull_slope': [np.nan, .1, np.nan, .2],
                          'original': [np.nan, .1, np.nan, .3], 'without_pull': [np.nan, .2, np.nan, .4],
                          'readers': [np.nan, .5, np.nan, .7], 'data': [.8, np.nan, .6, np.nan],
                          'data_low': [.5, np.nan, .4, np.nan], 'data_high': [.9, np.nan, .7, np.nan]})
        got = ranking.fold_means(t).set_index('model')
        self.assertAlmostEqual(got.loc[ranking.PLATE_REFERENCE, 'data'], .7)
        cpa = got.loc['CPA', ['original', 'without_pull', 'readers']].astype(float)
        self.assertTrue(np.allclose(cpa, [.2, .3, .6]))
        self.assertEqual(got.loc['CPA', 'folds'], 2)


if __name__ == '__main__':
    unittest.main()
