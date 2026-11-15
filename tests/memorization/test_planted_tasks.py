"""The planted tasks' definitions on small hand-made arrays; no study data: the planted block and its clipping, the
change of the drug average, the line permutation of the null, the choice of the gated drugs, the carrier cue and the
known-answer slope.

    python -m pytest tests/memorization/test_planted_tasks.py
"""
import unittest

import numpy as np

from reproduce.memorization import planted

PLATES = ('plate6', 'plate14')


class Plant(unittest.TestCase):
    def test_changes_only_the_block_and_clips_it_at_zero_in_single_precision(self):
        values = np.array([[1., 2., 3.], [4., 5., 6.]], np.float32)
        out = planted.plant(values, np.array([1]), np.array([0, 2]), np.array([[-5., 1.5]]))
        self.assertTrue(np.array_equal(out, [[1., 2., 3.], [0., 5., 7.5]]))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(values[1, 0], 4.)


class DrugChanges(unittest.TestCase):
    def test_are_the_weighted_means_over_each_drugs_units(self):
        change = np.array([[1., 0.], [3., 2.], [5., 5.]])
        got = planted.drug_changes(change, np.array([0, 0, 1]), np.array([1., 3., 1.]), 2)
        self.assertTrue(np.allclose(got, [[2.5, 1.5], [5., 5.]]))


class LinePermutation(unittest.TestCase):
    def test_both_plates_of_a_line_take_one_other_lines_values_within_the_bound(self):
        contexts = [f'L{i}|{p}' for i in range(8) for p in PLATES]
        z = np.arange(16.)
        got = planted.line_permutation(contexts, z, np.random.default_rng(0))
        lookup = {c: v for c, v in zip(contexts, z)}
        source = {}
        for c, v in zip(contexts, got):
            line, plate = c.split('|')
            match = [k for k in contexts if k.endswith('|' + plate) and lookup[k] == v]
            source.setdefault(line, set()).add(match[0].split('|')[0])
        self.assertTrue(all(len(s) == 1 for s in source.values()))
        self.assertEqual(sorted(next(iter(s)) for s in source.values()), sorted(source))
        self.assertLessEqual(abs(np.corrcoef(got, z)[0, 1]), planted.PERMUTATION_MAX_CORR)


class GatedDrugs(unittest.TestCase):
    def test_are_the_drugs_measured_everywhere_with_the_flattest_receptor_slope_without_steroids(self):
        contexts, receptor = 4, np.array([-1.5, -.5, .5, 1.5])
        slopes = [2., .1, -.3, .0, .2, .05]
        names = ['a', 'b', 'Triamcinolone', 'c', 'd', 'e']
        parent = np.tile(np.arange(contexts), len(names))
        p_code = np.repeat(np.arange(len(names)), contexts)
        response = np.array([slopes[p] * receptor[c] for c, p in zip(parent, p_code)])
        keep = ~((p_code == 5) & (parent == 0))   # drug 'e' is not measured in every context
        got = planted.gated_drugs(response[keep], parent[keep], p_code[keep], receptor, names, contexts)
        self.assertEqual(got, [3, 1, 4])


class CarrierCue(unittest.TestCase):
    def test_is_centred_within_cell_type_and_scaled_by_the_sd(self):
        contexts = ['D1|T', 'D2|T', 'D1|B', 'D2|B']
        score = np.array([1., 3., 10., 14.])
        centred = np.array([-1., 1., -2., 2.])
        self.assertTrue(np.allclose(planted.carrier_cue(score, contexts), centred / centred.std()))


class KnownSlope(unittest.TestCase):
    def setUp(self):
        self.known = known = planted.Known.__new__(planted.Known)
        known.units = {'weight': np.array([1., 2., 1., 1.])}
        known.rows, known.genes, known.beta = np.array([True, True, True, False]), np.array([1]), 2.
        known.cue = np.array([-1., 0., 2., 5.])
        known.dt = np.zeros((4, 2))
        known.dt[:, 1] = known.beta * known.cue

    def test_the_planted_truth_reads_one(self):
        self.assertAlmostEqual(self.known.read(np.zeros((4, 2)))['truth_slope_over_beta'], 1.)

    def test_is_the_weighted_slope_on_the_planted_block_over_beta(self):
        net = np.array([[100., 3.], [100., 1.], [100., 0.], [100., -50.]])   # gene 0 and unit 3 are off the block
        x0 = np.array([-1., 0., 2.]) - .25   # the cue minus its weighted mean over the planted units
        w = np.array([1., 2., 1.])
        expected = np.sum(w * x0 * [3., 1., 0.]) / np.sum(w * x0 ** 2) / 2.
        self.assertAlmostEqual(self.known.read(net)['slope_over_beta'], expected)
