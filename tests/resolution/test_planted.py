"""Planted changes' defining properties on small hand-made inputs; no study data.

    python -m pytest tests/resolution/test_planted.py
"""
import unittest

import numpy as np

from scsailor.resolution import menu, net_cost, planted
from scsailor.resolution.scoring import project
from scsailor.resolution.slopes import inner

from test_net_cost import side


class Plant(unittest.TestCase):
    def setUp(self):
        self.sd = side()
        self.amounts = net_cost.calibrate_original(self.sd)
        self.u, self.p = self.sd["u"], self.sd["p"][1]

    def residual(self, part):
        """The target minus the original model calibrated toward the design-aware baseline, in one part."""
        T = net_cost.design_baseline(self.sd, self.amounts["kappa"])
        return planted.split(self.u["truth"] - menu.two_part(T, self.p, *self.amounts["seeds"]["1"]["original_design"], self.u), self.u, part)

    def change(self, part, s, energy):
        return planted.plant(self.sd, 1, self.amounts, part, s, energy, np.random.default_rng(0)) - self.p

    def test_the_change_has_the_planted_energy_and_slope_on_the_residual(self):
        w = self.u["weight"]
        for part in planted.PARTS:
            for s, energy in ((.2, 5.), (.5, 2.), (0., 3.)):
                with self.subTest(part=part, slope=s):
                    x, e = self.change(part, s, energy), self.residual(part)
                    self.assertAlmostEqual(inner(x, x, w) / net_cost.total(self.p, self.u), energy / 100, places=12)
                    self.assertAlmostEqual(inner(x, e, w) / inner(x, x, w), s, places=10)

    def test_the_change_stays_in_its_part(self):
        common, contrast = self.change("common", .2, 5.), self.change("contrast", .2, 5.)
        context, w = self.u["context"], self.u["weight"]
        self.assertTrue(np.allclose(common, project(common, context, w), atol=1e-12))
        self.assertTrue(np.allclose(project(contrast, context, w), 0, atol=1e-12))

    def test_a_slope_that_needs_more_than_the_residual_cannot_be_planted(self):
        w = self.u["weight"]
        e = self.residual("common")
        energy = 2 * inner(e, e, w) / .75 ** 2 / net_cost.total(self.p, self.u) * 100   # slope^2 x energy above the residual's
        self.assertIsNone(planted.plant(self.sd, 1, self.amounts, "common", .75, energy, np.random.default_rng(0)))
        self.assertIsNone(planted.read_back(self.sd, self.sd, self.amounts, 0, "common", .75, energy))

    def test_the_grid(self):
        grid = list(planted.grid())
        self.assertEqual(len(grid), 2 * (len(planted.SLOPES) * len(planted.RETURNS) + len(planted.NULL_ENERGY)))
        for part, s, g, energy in grid:
            if s > 0:
                self.assertAlmostEqual(s * s * energy, g, places=12)
            else:
                self.assertEqual(g, 0.)


class ReadBack(unittest.TestCase):
    def test_the_cost_sits_in_the_planted_part(self):
        sd = side()
        amounts = net_cost.calibrate_original(sd)
        for part, other in (("common", "cal_cost_contrast"), ("contrast", "cal_cost_common")):
            with self.subTest(part=part):
                readings, gaps = planted.read_back(sd, sd, amounts, 0, part, .5, 3.)
                self.assertAlmostEqual(readings[other], 0., places=8)
                self.assertAlmostEqual(readings[f"cal_cost_{part}"], readings["cal_cost"], places=8)
                self.assertEqual(set(gaps), {1, 2, 3, "variant"})


if __name__ == "__main__":
    unittest.main()
