"""The readers' and the arms' definitions on small hand-made arrays; no study data.

    python -m pytest tests/memorization/test_readers.py
"""
import unittest

import numpy as np
import pandas as pd

from scsailor.resolution import net_cost
from reproduce.memorization import arms, data, programs, readers
from reproduce.memorization.data import Design

PLATES = ('plate6', 'plate14')


def contexts_of(lines):
    return [f'L{i}|{p}' for i in range(lines) for p in PLATES]


def units(lines=3, drugs=4, genes=5, seed=0):
    """Every drug on every context (a line on one of the two plates), random truth and weights."""
    rng = np.random.default_rng(seed)
    contexts = contexts_of(lines)
    context = np.repeat(contexts, drugs)
    n = len(context)
    return {'truth': rng.uniform(0, 2, (n, genes)), 'context': context,
            'p_code': np.tile(np.arange(drugs), len(contexts)), 'weight': rng.uniform(.5, 1.5, n)}


def per_unit(values, context, contexts):
    """Per-context rows spread to the units of each context."""
    return np.stack([values[contexts.index(c)] for c in context])


class Offsets(unittest.TestCase):
    def test_offset_is_the_context_mean_of_the_change_beyond_the_drug_average(self):
        u = units()
        change, context, code, w = u['truth'], u['context'], u['p_code'], u['weight']
        average = np.random.default_rng(9).normal(size=(4, change.shape[1]))   # per drug
        found, Y = readers.offsets(change, context, code, w, average)
        rest = change - np.stack([average[q] for q in code])
        self.assertEqual(list(found), sorted(set(context)))
        for c, y in zip(found, Y):
            m = context == c
            self.assertTrue(np.allclose(y, (w[m, None] * rest[m]).sum(0) / w[m].sum()))

    def test_constant_is_the_training_contexts_mean_offset_on_each_units_plate(self):
        contexts = contexts_of(3)
        Y = np.arange(12.).reshape(6, 2)
        got = readers.constant(Y, contexts, np.array(['plate14', 'plate6', 'plate14']))
        six, fourteen = Y[0::2].mean(0), Y[1::2].mean(0)
        self.assertTrue(np.allclose(got, [fourteen, six, fourteen]))


class Arms(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.u = units(seed=1)
        self.D = Design(self.u)
        n, g = self.u['truth'].shape
        self.P, self.A = rng.normal(1, .5, (n, g)), rng.normal(1, .5, (n, g))
        self.contexts = sorted(set(self.u['context']))
        self.off = {k: per_unit(rng.normal(0, .3, (len(self.contexts), g)), self.u['context'], self.contexts)
                    for k in ('constant', 'readers')}

    def test_removal_sets_the_context_level_correction_to_zero(self):
        Q = readers.arm_predictions(self.P, self.A, self.off, self.D)['removal']
        self.assertTrue(np.allclose(self.D.project_ctx(Q - self.A), 0))

    def test_every_arm_keeps_the_drug_differences_of_the_prediction(self):
        within = lambda X: X - self.D.project_ctx(X)   # noqa: E731
        for arm, Q in readers.arm_predictions(self.P, self.A, self.off, self.D).items():
            self.assertTrue(np.allclose(within(Q), within(self.P)), arm)

    def test_constant_and_readers_replace_the_correction_context_means_equal_anchor_plus_offset(self):
        got = readers.arm_predictions(self.P, self.A, self.off, self.D)
        self.assertEqual(list(got), list(readers.ARMS))
        for arm in ('constant', 'readers'):
            self.assertTrue(np.allclose(self.D.project_ctx(got[arm]), self.D.project_ctx(self.A) + self.off[arm]), arm)

    def test_calibrated_readers_apply_the_readers_replacement_to_the_calibrated_prediction(self):
        cal = self.P * .5 + self.A * .5
        got = readers.arm_predictions(self.P, self.A, self.off, self.D, calibrated=cal)
        self.assertEqual(list(got), list(readers.ARMS) + list(readers.CALIBRATED_ARMS))
        self.assertTrue(np.array_equal(got['calibrated'], cal))
        self.assertTrue(np.array_equal(got['calibrated + readers'],
                                       readers.arm_predictions(cal, self.A, self.off, self.D)['readers']))


def line_features(lines, width, rng, noise=.01):
    """Features shared by a line's two plates, with some plate noise, centred per plate and scaled."""
    F = np.repeat(rng.normal(size=(lines, width)), 2, axis=0) + rng.normal(0, noise, (2 * lines, width))
    return readers.centre_scale(F, np.tile(PLATES, lines), np.ones(2 * lines, bool))


def best_penalty(F, Y, group):
    """The penalty with the smallest error leaving out each group of contexts in turn (a direct solve, plate means
    recomputed without the group)."""
    plate = np.tile(PLATES, len(Y) // 2)
    errors = []
    for a in readers.PENALTIES:
        e = 0.
        for g in np.unique(group):
            fit, test = group != g, group == g
            mean = {p: Y[fit & (plate == p)].mean(0) for p in PLATES}
            Yc = Y[fit] - np.stack([mean[p] for p in plate[fit]])
            coef = np.linalg.solve(F[fit] @ F[fit].T + a * np.eye(fit.sum()), Yc)
            e += np.sum((np.stack([mean[p] for p in plate[test]]) + F[test] @ F[fit].T @ coef - Y[test]) ** 2)
        errors.append(e)
    return readers.PENALTIES[int(np.argmin(errors))]


class KernelReader(unittest.TestCase):
    def test_penalty_chosen_leaving_one_line_out_both_plates_together(self):
        # a readable part plus a memory of each line shared by its two plates: leaving one context out would read the
        # memory from the line's other plate and choose another penalty
        rng = np.random.default_rng(0)
        F = line_features(8, 6, rng, noise=.1)
        Y = (.3 * F @ rng.normal(size=(6, 3)) + np.repeat(rng.normal(size=(8, 3)), 2, axis=0)
             + rng.normal(0, .2, (16, 3)))
        by_line, by_context = best_penalty(F, Y, np.repeat(np.arange(8), 2)), best_penalty(F, Y, np.arange(16))
        self.assertNotEqual(by_line, by_context)
        self.assertEqual(readers.KernelReader(F @ F.T, contexts_of(8), np.arange(16), Y).penalty, by_line)

    def test_predicts_the_plate_mean_plus_the_fitted_deviation(self):
        rng = np.random.default_rng(4)
        F = line_features(7, 2, rng)
        w = np.array([[1.], [-2.]])
        Y = F[:12] @ w + np.where(np.arange(12) % 2, 5., -5.)[:, None]
        reader = readers.KernelReader(F @ F.T, contexts_of(6) + ['N|plate6', 'N|plate14'], np.arange(12), Y)
        self.assertEqual(reader.penalty, readers.PENALTIES[0])
        self.assertTrue(np.allclose(reader.predict([12, 13]), F[12:] @ w + np.array([[-5.], [5.]]), atol=.05))


class GeneReaders(unittest.TestCase):
    def test_penalty_per_gene_leaving_one_line_out(self):
        rng = np.random.default_rng(5)
        contexts = contexts_of(8)
        X = rng.normal(2, 1, (16, 2))
        plate = np.tile(PLATES, 8)
        readable = 3 * (X[:, 0] - np.where(plate == 'plate6', X[0::2, 0].mean(), X[1::2, 0].mean()))
        memory = np.repeat(rng.normal(size=8), 2)
        reader = readers.GeneReaders(X, contexts, np.column_stack([readable, memory]))
        self.assertEqual(reader.penalty[0], readers.PENALTIES[0])
        self.assertEqual(reader.penalty[1], readers.PENALTIES[-1])

    def test_reads_each_gene_from_its_own_input_set_mean(self):
        rng = np.random.default_rng(6)
        contexts = contexts_of(8)
        X = rng.normal(2, 1, (16, 2))
        Y = np.column_stack([2 * X[:, 0], -X[:, 1]])
        Y -= np.where(np.tile(PLATES, 8) == 'plate6', 1, 0)[:, None]
        reader = readers.GeneReaders(X, contexts, Y)
        new = np.array([[2.5, 1.0], [1.0, 3.0]])
        expect = np.column_stack([2 * new[:, 0], -new[:, 1]]) - np.array([[1], [0]])
        self.assertTrue(np.allclose(reader.predict(new, ['N|plate6', 'N|plate14']), expect, atol=.05))


class Centring(unittest.TestCase):
    def test_centre_removes_each_plates_mean(self):
        M = np.arange(12.).reshape(6, 2) ** 2
        plate = np.tile(PLATES, 3)
        got = readers.centre(M, plate)
        for p in PLATES:
            self.assertTrue(np.allclose(got[plate == p], M[plate == p] - M[plate == p].mean(0)))


class Features(unittest.TestCase):
    def test_centred_per_plate_on_training_contexts_and_scaled_by_their_sd(self):
        rng = np.random.default_rng(7)
        X = rng.normal(3, 2, (10, 2))
        plate = np.tile(PLATES, 5)
        train = np.arange(10) < 6
        Z = readers.centre_scale(X, plate, train)
        for p in PLATES:
            self.assertTrue(np.allclose(Z[train & (plate == p)].mean(0), 0))
        centred = X.copy()
        for p in PLATES:
            centred[plate == p] -= X[train & (plate == p)].mean(0)
        self.assertTrue(np.allclose(Z, centred / centred[train].std(0)))
        self.assertTrue(np.allclose(Z[train].std(0), 1))

    def test_gut_lineage_is_bowel_or_esophagus_stomach(self):
        meta = pd.DataFrame({'Cell_ID_Cellosaur': ['A', 'A', 'B', 'C', 'D', 'E'],
                             'Organ': ['Bowel', 'Bowel', 'Esophagus/Stomach', 'Pancreas', 'Liver', 'Lung']})
        self.assertEqual(readers.gut_lines(meta), frozenset({'A', 'B'}))


class Programs(unittest.TestCase):
    def test_growth_genes_are_the_three_programs_on_the_panel(self):
        sets = {'HALLMARK_E2F_TARGETS': {'a', 'b', 'x'}, 'HALLMARK_G2M_CHECKPOINT': {'b', 'c'},
                'HALLMARK_MITOTIC_SPINDLE': {'d'}, 'HALLMARK_APOPTOSIS': {'e'}}
        genes = np.array(['a', 'b', 'c', 'd', 'e', 'f'])
        masks, union = programs.masks(genes, sets)
        self.assertEqual([list(genes[m]) for m in masks], [['a', 'b'], ['b', 'c'], ['d']])
        self.assertEqual(list(genes[union]), ['a', 'b', 'c', 'd'])

    def test_reads_a_gmt_line_as_name_description_genes(self):
        self.assertEqual(programs.gene_sets('S1\thttp://x\ta\tb\nS2\t-\tc\n'), {'S1': {'a', 'b'}, 'S2': {'c'}})


class Table(unittest.TestCase):
    def test_total_error_and_its_common_and_contrast_components_in_points_of_the_baseline(self):
        rng = np.random.default_rng(8)
        u = units(seed=8)
        D = Design(u)
        preds = {'original': u['truth'] + rng.normal(0, .3, u['truth'].shape),
                 'removal': u['truth'] + rng.normal(0, .2, u['truth'].shape)}
        rows = arms.score({'fold': 'F1'}, preds, D, u['truth'], 2.)
        self.assertEqual([r['arm'] for r in rows], ['original', 'removal'])
        for r, Q in zip(rows, preds.values()):
            E = np.maximum(Q, 0) - u['truth']
            self.assertAlmostEqual(r['total_points'], 100 * D.energy(E) / 2.)
            self.assertAlmostEqual(r['common_points'], 100 * D.energy(D.project_ctx(E)) / 2.)
            self.assertAlmostEqual(r['total_points'], r['common_points'] + r['contrast_points'])

    def test_seed_means_of_the_total_error_as_written(self):
        info = {'fold': 'F1', 'model': 'CPA', 'level': 'Mean', 'units': 'all held-out', 'arm': 'original'}
        four = pd.DataFrame([{**info, 'seed': s, 'total_points': v, 'common_points': 50., 'contrast_points': v - 50.}
                             for s, v in zip((31002, 31003, 31004), (104.1451, 105.86802, 104.73141))])
        [row] = arms.seed_means_table(four).to_dict('records')
        self.assertEqual({k: row[k] for k in info}, info)
        self.assertAlmostEqual(row['total_points'], (104.145 + 105.868 + 104.731) / 3, places=12)

    def test_drug_label_reads_the_deposited_string_as_a_literal_and_runs_no_code(self):
        self.assertEqual(data.drug_label("[('Adagrasib', 0.05, 'uM')]"), 'Adagrasib 0.05 uM')
        with self.assertRaises(ValueError):
            data.drug_label("[(__import__('os').getcwd(), 0.05, 'uM')]")

    def test_input_set_and_pool_means_of_the_contexts_in_the_order_given(self):
        side = {'a': (np.array([1., 2.]), np.array([3., 4.])), 'b': (np.array([5., 6.]), np.array([7., 8.]))}
        self.assertTrue(np.array_equal(data.input_means(side, ['b', 'a']), [[5, 6], [1, 2]]))
        self.assertTrue(np.array_equal(data.pool_means(side, ['b', 'a']), [[7, 8], [3, 4]]))

    def test_all_contexts_put_the_training_contexts_first_each_side_sorted(self):
        bags = {'heldout': {'H|plate6': (np.array([1.]), None)},
                'train': {'B|plate6': (np.array([2.]), None), 'A|plate6': (np.array([3.]), None)}}
        contexts, train, means = data.all_contexts(bags)
        self.assertEqual(contexts, ['A|plate6', 'B|plate6', 'H|plate6'])
        self.assertEqual(list(train), [True, True, False])
        self.assertTrue(np.array_equal(means, [[3.], [2.], [1.]]))

    def test_calibration_supported_units_are_the_drugs_on_calibration_lines(self):
        keep = data.supported(np.array([0, 1, 2, 0, 3]), np.array(['a', 'a', 'b', 'c', 'c']), ['c', 'd'],
                               np.array([3, 1, 0, 2, 3]))
        self.assertEqual(list(keep), [True, False, True, False, True])


class DesignKappa(unittest.TestCase):
    def test_is_origin_resolutions_design_aware_amount_up_to_the_last_bits(self):
        rng = np.random.default_rng(10)
        b, x, w = rng.uniform(0, 2, (40, 7)), rng.normal(0, .5, (40, 7)), rng.uniform(.5, 1.5, 40)
        y = .7 * x + rng.normal(0, .3, (40, 7))
        reference = net_cost.calibrate_design({'u': {'truth': b + y, 'weight': w}, 'b': b, 'A': b + x})
        # the same estimator, its sums taken in another order: equal up to a few units in the last place
        self.assertLessEqual(abs(data.design_kappa(x, y, w) - reference), 4 * np.spacing(reference))

    def test_is_truncated_to_zero_and_one(self):
        rng = np.random.default_rng(11)
        x, w = rng.normal(size=(10, 3)), rng.uniform(.5, 1.5, 10)
        self.assertEqual(data.design_kappa(x, 2 * x, w), 1.)
        self.assertEqual(data.design_kappa(x, -x, w), 0.)


if __name__ == '__main__':
    unittest.main()
