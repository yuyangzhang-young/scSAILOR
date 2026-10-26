"""The dataset references' arithmetic on small hand-made arrays (no study data).

    python -m pytest tests/resolution/test_references.py
"""
import unittest

import numpy as np

from scsailor.resolution import references as R


class CellSampling(unittest.TestCase):
    def test_gene_averaged_variance_over_the_cell_count(self):
        x = np.array([[0., 1.], [2., 3.], [4., 8.]])
        v, variance = R.cell_sampling(x)
        self.assertTrue(np.allclose(variance, [4., 13.]))
        self.assertAlmostEqual(v, 8.5 / 3)

    def test_moments_from_streamed_sums(self):
        x = np.random.default_rng(0).normal(size=(7, 3))
        mean, variance = R.moments(x.sum(0)[None], (x * x).sum(0)[None], np.array([7]))
        self.assertTrue(np.allclose(mean[0], x.mean(0)) and np.allclose(variance[0], x.var(0, ddof=1)))


class Floors(unittest.TestCase):
    units = {"context": np.array(["a", "a", "b"]), "weight": np.array([1., 1., 2.])}

    def test_context_shares(self):
        w, q = R.context_shares(self.units["context"], self.units["weight"])
        self.assertTrue(np.allclose(w, [.25, .25, .5]) and np.allclose(q, [.5, .5, 1.]))

    def test_drug_difference_floor_per_unit(self):
        self.assertTrue(np.allclose(R.d1_difference(self.units, np.array([4., 8., 2.])), [.5, 1., 0.]))

    def test_floor_of_one_whole_line_context(self):
        self.assertEqual(R.line_floor(np.array([2., 4.]), np.array([1., 1.])), {"total": 3., "common": 1.5, "difference": 1.5})

    def test_pool_sampling_of_twin_contexts(self):
        term = R.pool_sampling_term(self.units["context"], self.units["weight"], {"a": 1., "b": 3.}, {"a": "b", "b": "a"})
        self.assertAlmostEqual(term, .5 * 2 + .5 * 2)


class Twins(unittest.TestCase):
    def test_treated_and_response_deviations(self):
        y, pool = np.array([[1.], [3.]]), np.array([[1.], [0.]])
        twin = np.array([1, 0])
        self.assertTrue(np.allclose(R.twin_deviation(y, twin), np.array([[-2.], [2.]]) / np.sqrt(2)))
        self.assertTrue(np.allclose(R.twin_deviation(y, twin, pool), np.array([[-3.], [3.]]) / np.sqrt(2)))

    def test_well_split_adds_up_to_the_drug_difference_energy(self):
        rng = np.random.default_rng(1)
        drug, plate = np.array([0, 0, 0, 1, 1, 1, 1, 2]), np.array([0, 0, 1, 0, 0, 1, 1, 0])
        delta, weight, d1 = rng.normal(size=(8, 5)), rng.uniform(1, 2, 8), rng.uniform(0, 1, 8)
        drugs, single = R.well_split(delta, weight, drug, plate, d1)
        energy = float(np.sum(weight / weight.sum() * np.mean(delta ** 2, axis=1)))
        split = sum(r["shared"] + r["line_specific"] for r in drugs.values()) + single["energy"]
        self.assertAlmostEqual(split, energy)
        self.assertEqual(single["wells"], 2)
        self.assertAlmostEqual(sum(r["d1"] for r in drugs.values()) + single["d1"], d1.sum())

    def test_pair_rows_carry_the_partners_sampling(self):
        per = R.pair_sampling_content(np.array([0]), np.array([1]), np.array([1.]), np.array(["c"]), np.array([2., 4.]))
        self.assertEqual(per, {"c": (1., 3., 3.)})


class Pools(unittest.TestCase):
    def test_identical_pools_have_no_level_beyond_sampling(self):
        pools = [{"C": np.ones(3), "pool_sampling": 0., "plate": p, "campaign": c, "pool_cells": 100}
                 for p, c in (("x1", "X"), ("x1", "X"), ("x2", "X"), ("y1", "Y"), ("y1", "Y"), ("y2", "Y"))]
        out = R.pool_decomposition(pools, ("X", "Y"))
        for key in ("pool_within_plate", "plate_within_campaign", "campaign_shift"):
            self.assertEqual(out[key], 0.)
        self.assertEqual((out["X_pools"], out["X_plates"]), (3, 2))


class GEM(unittest.TestCase):
    def test_group_variance_net_of_sampling(self):
        tau2 = R.gem_variance(np.array([[0.], [2.]]), np.array([[3.], [3.]]), np.array([3, 3]))
        self.assertTrue(np.allclose(tau2, [2. - 1.]))

    def test_term_against_a_matching_reference_is_zero(self):
        share = np.array([[.25, .75], [.5, .5]])
        self.assertTrue(np.allclose(R.gem_term(2., share), [1.25, 1.]))
        self.assertTrue(np.allclose(R.gem_term(2., share, np.array([.5, .5])), [.25, 0.]))


class Knockdown(unittest.TestCase):
    def test_a_quantity_rising_with_the_knockdown_difference(self):
        rng = np.random.default_rng(2)
        dkd = rng.normal(size=60)
        q = np.stack([dkd + .1 * rng.normal(size=60), -dkd])
        out = R.knockdown_readout(q, np.array([False, True]), dkd, rng.uniform(size=60), rng.uniform(size=60),
                                  rng.uniform(size=60), np.random.default_rng(3), permutations=200)
        self.assertTrue(out["slope"][0] > 0 > out["slope"][1])
        self.assertTrue(np.all(out["p"] < .05))
        self.assertEqual(out["quintile_sizes"], [12] * 5)


class Reliability(unittest.TestCase):
    def test_identical_plates_are_fully_reliable(self):
        rng = np.random.default_rng(4)
        line = np.repeat(["l1", "l2", "l3"], 4)
        half = np.tile([0, 0, 1, 1], 3)
        drug = np.tile([0, 1, 0, 1], 3)
        context = np.array([f"{l}|{h}" for l, h in zip(line, half)])
        shift = {l: rng.normal(size=4) for l in ("l1", "l2", "l3")}
        response = {d: rng.normal(size=4) for d in (0, 1)}
        correction = np.stack([shift[l] + response[d] + .3 * response[d] * (l == "l2") for l, d in zip(line, drug)])
        units = {"context": context, "weight": np.ones(12)}
        twin = np.array([2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9])
        out = R.reliability(correction, units, line, half, twin)
        self.assertAlmostEqual(out["common"]["split_half_R_single_plate"], 1.)
        self.assertAlmostEqual(out["differential"]["split_half_R_unit_twins"], 1.)
        self.assertEqual(out["differential"]["pairs"], 6)
        self.assertAlmostEqual(R.split_half(np.ones((2, 3)), -np.ones((2, 3)), np.array([.5, .5])), -1.)


if __name__ == "__main__":
    unittest.main()
