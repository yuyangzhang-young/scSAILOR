"""The matched comparisons' defining properties on small hand-made inputs; no study data.

    python -m pytest tests/resolution/test_net_cost.py
"""
import unittest

import numpy as np

from scsailor.resolution import net_cost

SEEDS = (1, 2, 3)


def side(lines=3, strata=("s", "t"), drugs=4, genes=6, seed=0):
    """A side with an original model and a change per seed, on contexts line|stratum, every perturbation in each."""
    rng = np.random.default_rng(seed)
    contexts = np.repeat([f"L{i}|{s}" for i in range(lines) for s in strata], drugs)
    n = len(contexts)
    truth = rng.uniform(1, 3, (n, genes))
    u = {"truth": truth, "context": contexts, "p_code": np.tile(np.arange(drugs), lines * len(strata)), "weight": rng.uniform(.5, 1.5, n)}
    p = {s: truth + rng.normal(0, .2, truth.shape) for s in SEEDS}
    return {"u": u, "p": p, "R": {s: p[s] + rng.normal(0, .2, truth.shape) for s in SEEDS}, "a": truth + rng.normal(0, .3, truth.shape),
            "b": truth + rng.normal(0, .6, truth.shape), "A": truth + rng.normal(0, .25, truth.shape)}


class Slopes(unittest.TestCase):
    def test_two_models_slopes_toward_each_other_add_to_one(self):
        rows = net_cost.extra_seed_slopes(side())
        found = {(r["seed"], r["second"]): r["total"] for r in rows}
        self.assertEqual(len(found), 6)
        for i, j in found:
            self.assertAlmostEqual(found[(i, j)] + found[(j, i)], 1., places=10)

    def test_a_change_along_the_need_has_slope_one(self):
        sd = side()
        sd["R"] = {s: sd["u"]["truth"] for s in SEEDS}
        mean, per_seed = net_cost.change_slopes(sd)
        self.assertAlmostEqual(mean["total"], 1., places=10)
        self.assertEqual(len(per_seed), 3)

    def test_the_price_of_a_point(self):
        self.assertEqual(net_cost.price(.5), 0.)
        self.assertAlmostEqual(net_cost.price(2 ** .5 - 1), 1., places=10)
        self.assertIsNone(net_cost.price(-.1))


class Design(unittest.TestCase):
    def test_kappa_is_the_truncated_slope_on_the_stratum_mean(self):
        sd = side()
        sd["A"] = sd["b"] + 2 * (sd["u"]["truth"] - sd["b"])
        self.assertAlmostEqual(net_cost.calibrate_design(sd), .5, places=10)
        sd["A"] = sd["b"] + .1 * (sd["u"]["truth"] - sd["b"])
        self.assertEqual(net_cost.calibrate_design(sd), 1.)

    def test_the_stratum_shift_is_the_stratum_mean_minus_the_mean_over_strata(self):
        contexts = np.asarray(["L1|s", "L1|t", "L2|s", "L2|t"])
        train = {"truth": np.asarray([[1.], [3.], [1.], [3.]]), "context": contexts, "p_code": np.zeros(4, dtype=int), "weight": np.ones(4)}
        bags = {c: (np.zeros(1), np.zeros(1)) for c in contexts}
        strata = np.asarray([c.split("|")[1] for c in contexts])
        u = {"p_code": np.asarray([0, 0, 1])}
        shift = net_cost.stratum_shift(u, np.asarray(["s", "t", "s"]), train, bags, np.ones(4, dtype=bool), strata)
        self.assertEqual(shift[:, 0].tolist(), [-1., 1., 0.])   # a perturbation absent from training keeps zero


class Gate(unittest.TestCase):
    def setUp(self):
        self.sd = side()
        self.amounts = net_cost.calibrate_original(self.sd)
        self.kept = net_cost.calibrate_change(self.sd, self.amounts)
        self.gaps = net_cost.cumulative_gaps(self.sd, self.amounts, self.kept)

    def test_the_sets_accumulate(self):
        names = [set(self.gaps[stage][1]) for stage in (1, 2, 3)]
        self.assertEqual(len(names[0]), 6)
        self.assertTrue(names[0] < names[1] < names[2])

    def test_the_gate_needs_every_seed_above_the_margin_and_some_of_the_change_kept(self):
        gaps = {1: {"x": 1., "y": .3}, 2: {"x": .4, "y": .9}}
        self.assertTrue(net_cost.recommended(gaps, [.2, .1], .2))
        self.assertFalse(net_cost.recommended(gaps, [.2, .1], .5))
        self.assertFalse(net_cost.recommended(gaps, [.2, 0.], .2))

    def test_both_orders_of_the_accounting_add_to_the_same_distance(self):
        row = net_cost.ledger(self.sd, self.amounts, self.kept)
        self.assertAlmostEqual(row["design"] + row["correction"] + row["change"], row["gap"], places=10)
        self.assertAlmostEqual(row["reverse_change"] + row["reverse_correction"] + row["reverse_design"], row["gap"], places=10)


class Recovery(unittest.TestCase):
    def test_recovery_at_the_kept_size(self):
        self.assertEqual(net_cost.recovery_at(.8, .9, 0.), 0.)
        self.assertAlmostEqual(net_cost.recovery_at(.8, .9, 1.), .8, places=12)

    def test_a_planted_reading(self):
        support = {"bits": {"1": {"Q": .7, "predicted_halfcontrast_energy": 2., "truth_halfcontrast_energy": 4.}}}
        self.assertEqual(net_cost.planted_reading(support, 1), (.7, .5))


if __name__ == "__main__":
    unittest.main()
