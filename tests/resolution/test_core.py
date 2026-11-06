"""The core's defining properties on small hand-made inputs; no study data.

    python -m pytest tests/resolution/test_core.py
"""
import unittest

import numpy as np

from scsailor.resolution import attribution, card, comparison, menu, origins, policies, scoring


def units(lines=3, plates=("plate6", "plate14"), drugs=4, genes=5, seed=0):
    """Contexts line|plate, every drug in every context, random targets and weights."""
    rng = np.random.default_rng(seed)
    contexts = np.repeat([f"L{i}|{p}" for i in range(lines) for p in plates], drugs)
    return {"truth": rng.uniform(0, 2, (len(contexts), genes)), "context": contexts,
            "p_code": np.tile(np.arange(drugs), lines * len(plates)), "weight": rng.uniform(.5, 1.5, len(contexts))}


class Scoring(unittest.TestCase):
    def test_parts_add_up(self):
        u = units()
        pred = u["truth"] + np.random.default_rng(1).normal(0, .5, u["truth"].shape)
        s = scoring.scores(pred, u)
        self.assertAlmostEqual(s["total_mse"], s["common_error_mse"] + s["difference_error_mse"], places=12)
        parts = scoring.unit_parts(pred, u)
        self.assertAlmostEqual(parts["total"].sum(), s["total_mse"], places=12)
        self.assertAlmostEqual(parts["common"].sum(), s["common_error_mse"], places=12)
        lines = scoring.line_parts(pred, u)
        self.assertAlmostEqual(sum(lines["total"].values()), s["total_mse"], places=12)

    def test_retained_weights_follow_the_datasets_rule(self):
        u = units()
        keep = np.ones(len(u["truth"]), bool)
        keep[[0, 1, 9]] = False   # two drugs of one context and one of another
        r = scoring.retained(u, keep)
        mass = {c: r["weight"][r["context"] == c].sum() for c in np.unique(r["context"])}
        self.assertTrue(np.allclose(list(mass.values()), 1 / len(mass)))   # contexts equal
        first = r["context"] == r["context"][0]
        self.assertTrue(np.allclose(r["weight"][first], r["weight"][first][0]))   # drugs equal within a context


class Comparison(unittest.TestCase):
    def runs(self, shift, lines):
        rng = np.random.default_rng(2)
        base = {f"L{i}": 1. for i in range(lines)}
        return ([({"total": {k: v + shift + rng.normal(0, .001) for k, v in base.items()}}, 100.) for _ in range(3)],
                [({"total": base}, 100.)] * 3)

    def test_a_clear_gain_is_detected_on_enough_units(self):
        x, y = self.runs(-.1, 14)
        d = comparison.detect(x, y, with_bootstrap=False)
        self.assertEqual((d["level"], d["detected"], d["seed_rule"]), ("unit-level", True, "holds"))

    def test_too_few_units_give_a_case_level_reading(self):
        x, y = self.runs(-.1, 4)
        d = comparison.detect(x, y, with_bootstrap=False)
        self.assertEqual((d["level"], d["detected"]), ("case-level", None))


class Calibration(unittest.TestCase):
    def test_ties_keep_less_change(self):
        u = units()
        p = u["truth"] + .3
        self.assertEqual(card.calibrate(lambda x: p, u)[0], 1.)      # one amount: no change (1) first
        self.assertEqual(menu.pick_two(p, p, u)[0], [1., 1.])        # two amounts: the larger
        self.assertEqual(policies.choose("cheapest_fix", p, p, p, u, np.ones(len(p), bool))[0], 1.)

    def test_cell_line_shrink_removes_only_the_context_level_correction(self):
        u = units()
        rng = np.random.default_rng(3)
        a, p = u["truth"] + rng.normal(0, .2, u["truth"].shape), u["truth"] + rng.normal(0, .4, u["truth"].shape)
        self.assertTrue(np.array_equal(policies.predict("cheapest_fix", 1., p, None, a, u), p))
        zero = policies.predict("cheapest_fix", 0., p, None, a, u)
        self.assertTrue(np.allclose(scoring.project(zero - a, u["context"], u["weight"]), 0))
        self.assertTrue(np.allclose(menu.two_part(a, p, 1., 1., u), p))


class Attribution(unittest.TestCase):
    def test_shapley_shares_add_up_to_the_joint_removal(self):
        value = {frozenset(): 0., frozenset({"M5"}): -3., frozenset({"M7"}): -1., frozenset({"M5", "M7"}): -3.5}
        shares = card.shapley(value.__getitem__, ["M5", "M7"])
        self.assertAlmostEqual(sum(shares.values()), -3.5)

    def test_noise_floor_over_whole_contexts(self):
        u = units()
        noise = np.random.default_rng(4).uniform(.1, .2, len(u["truth"]))
        f = attribution.floor_parts(u, noise)
        w = u["weight"] / u["weight"].sum()
        mass = {c: w[u["context"] == c].sum() for c in np.unique(u["context"])}
        q = w / np.asarray([mass[c] for c in u["context"]])
        self.assertAlmostEqual(f["total"].sum(), float(np.sum(w * noise)))
        self.assertAlmostEqual(f["common"].sum(), float(np.sum(w * q * noise)))
        self.assertAlmostEqual(f["differential"].sum(), float(np.sum(w * (1 - q) * noise)))


class Origins(unittest.TestCase):
    def test_one_row_per_origin_and_fold_one_marks(self):
        d = {"level": "unit-level", "detected": True, "total_pts": -4., "seed_totals_pts": [-4., -4., -4.], "p": .001,
             "floor": 1e-4, "seed_rule": "holds"}
        ev = {"detections": {"M5": d, "M7": d}, "kind": {"M5": "", "M7": ""}, "shares": {"total": {}},
              "references": {"T2": {"common": .2}, "anchor": {"total": 1., "differential": 1.},
                             "nochange": {"total": 1., "differential": 1.}},
              "d2": {"status": "reference"}, "d7": {"status": "descriptive"}, "units": 14, "seed_spread_pct": 1., "fold": "F2"}
        rows = origins.rows(ev, {"M4": ("supported", "crossed tests", "none")}, fold1_only=("M4",))
        self.assertEqual([r["origin"] for r in rows], list(origins.ORIGINS))
        self.assertEqual(next(r for r in rows if r["origin"] == "M4")["evidence"], "fold 1: crossed tests")


if __name__ == "__main__":
    unittest.main()
