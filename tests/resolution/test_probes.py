"""The fixed-weight probes' defining properties on small hand-made inputs; no study data.

    python -m pytest tests/resolution/test_probes.py
"""
import unittest

import numpy as np

from scsailor.resolution import probes
from scsailor.resolution.scoring import project, risk


def units(lines=3, plates=("plate6", "plate14"), drugs=4, genes=6, seed=0):
    rng = np.random.default_rng(seed)
    contexts = np.repeat([f"L{i}|{p}" for i in range(lines) for p in plates], drugs)
    return {"truth": rng.uniform(0, 2, (len(contexts), genes)), "context": contexts,
            "p_code": np.tile(np.arange(drugs), lines * len(plates)), "weight": rng.uniform(.5, 1.5, len(contexts))}


class Carrier(unittest.TestCase):
    def test_the_rebuilt_carrier_has_the_given_moments(self):
        rng = np.random.default_rng(5)
        s, h = rng.uniform(.5, 3, 6), rng.uniform(.1, .6, 6)
        atoms, capped = probes.l2_atoms(s, h, 256, 9.2, rng)
        self.assertEqual(capped, 0.)
        self.assertTrue(np.allclose(atoms.mean(axis=0), s))
        self.assertTrue(np.allclose(atoms.var(axis=0, ddof=1), h))

    def test_a_variance_the_mean_cannot_hold_is_capped(self):
        rng = np.random.default_rng(6)
        atoms, capped = probes.l2_atoms(np.array([.01, 1.]), np.array([100., .1]), 256, 9.2, rng)
        self.assertEqual(capped, .5)
        self.assertTrue(atoms.min() >= 0 and atoms.max() <= 9.2)


class Conditions(unittest.TestCase):
    def setUp(self):
        self.u = units()
        rng = np.random.default_rng(1)
        self.true = self.u["truth"] + rng.normal(0, .1, self.u["truth"].shape)
        self.swapped = self.u["truth"] + rng.normal(0, .4, self.u["truth"].shape)
        self.baseline = np.maximum(self.u["truth"] + rng.normal(0, .3, self.u["truth"].shape), 0)
        self.rows = probes.condition_scores({"true": self.true, "swapped": self.swapped}, self.u, self.baseline)

    def test_the_own_input_has_no_dependence_row(self):
        self.assertNotIn("dependence_total", self.rows["true"])

    def test_benefit_is_the_error_difference_between_conditions(self):
        for part in probes.PARTS:
            self.assertAlmostEqual(self.rows["swapped"][f"H_{part}"], self.rows["true"][f"L_{part}"] - self.rows["swapped"][f"L_{part}"], places=10)

    def test_dependence_is_the_risk_of_the_prediction_change(self):
        change = risk(np.maximum(self.true, 0) - np.maximum(self.swapped, 0), self.u["context"], self.u["weight"])
        base = risk(self.baseline - self.u["truth"], self.u["context"], self.u["weight"])["total_mse"]
        self.assertAlmostEqual(self.rows["swapped"]["dependence_total"], 100 * change["total_mse"] / base, places=10)

    def test_parts_add_up(self):
        row = self.rows["swapped"]
        self.assertAlmostEqual(row["L_total"], row["L_common"] + row["L_contrast"], places=10)


class Excess(unittest.TestCase):
    def test_excess_terms_add_to_the_cost_under_the_own_input(self):
        changed = {f"L_{p}": v for p, v in zip(probes.PARTS, (1., .6, .4))}
        original = {f"L_{p}": v for p, v in zip(probes.PARTS, (.5, .2, .3))}
        changed_true = {f"L_{p}": v for p, v in zip(probes.PARTS, (3., 2., 1.))}
        original_true = {f"L_{p}": v for p, v in zip(probes.PARTS, (.5, .3, .2))}
        row = probes.excess_columns(changed, original, changed_true, original_true)
        self.assertEqual(row["E_total"], .5)
        self.assertEqual(row["Delta_E_total"], 2.)
        self.assertEqual(row["parent_change_total"], 0.)


class MatchedShrink(unittest.TestCase):
    def setUp(self):
        self.u = units()
        rng = np.random.default_rng(2)
        self.anchor = rng.uniform(0, 1, self.u["truth"].shape)
        self.true = self.anchor + rng.normal(0, .5, self.anchor.shape)
        self.swapped = self.anchor + .3 * (self.true - self.anchor)

    def test_the_context_level_correction_gets_the_swapped_size(self):
        shrunk, alpha = probes.matched_shrink(self.true, self.swapped, self.anchor, self.u["context"], self.u["weight"])
        size = lambda x: np.linalg.norm(project(x - self.anchor, self.u["context"], self.u["weight"]), axis=1)
        self.assertTrue(np.allclose(size(shrunk), size(self.swapped)))
        self.assertTrue(np.all(alpha <= 1))

    def test_the_perturbation_level_correction_stays(self):
        shrunk, _ = probes.matched_shrink(self.true, self.swapped, self.anchor, self.u["context"], self.u["weight"])
        rest = lambda x: x - project(x, self.u["context"], self.u["weight"])
        self.assertTrue(np.allclose(rest(shrunk - self.anchor), rest(self.true - self.anchor)))


class Geometry(unittest.TestCase):
    def test_a_scaled_increment_reads_its_scale(self):
        rng = np.random.default_rng(3)
        need, w = rng.normal(size=(8, 5)), rng.uniform(.5, 1.5, 8)
        m = probes.moments(2 * need, need, w)
        d = probes.derived(np.r_[m["total"], m["total"]])
        self.assertAlmostEqual(float(d["alpha_star"]), .5, places=10)
        self.assertAlmostEqual(float(d["cosine"]), 1., places=10)

    def test_an_empty_increment_has_no_scale(self):
        self.assertTrue(np.isnan(probes.ratio(1., 0.)))


class OwnValue(unittest.TestCase):
    def test_the_value_of_own_controls_is_the_error_they_remove(self):
        u = units()
        rng = np.random.default_rng(4)
        own = u["truth"] + rng.normal(0, .1, u["truth"].shape)
        other = u["truth"] + rng.normal(0, .4, u["truth"].shape)
        baseline = np.maximum(u["truth"] + rng.normal(0, .3, u["truth"].shape), 0)
        draws = rng.integers(0, 3, size=(50, 3))
        rows = probes.own_value({"own": own, "other": other}, u, baseline, draws, ("other",))
        self.assertAlmostEqual(rows["other"]["value_total"], rows["other"]["total"] - rows["own"]["total"], places=10)
        self.assertGreater(rows["other"]["value_total"], 0)
        self.assertLessEqual(rows["other"]["value_total_low"], rows["other"]["value_total_high"])


if __name__ == "__main__":
    unittest.main()
