"""The slope readings' defining properties on small hand-made inputs; no study data.

    python -m pytest tests/resolution/test_slopes.py
"""
import unittest

import numpy as np

from scsailor.resolution import slopes

SEEDS = (1, 2, 3)


def side(lines=3, strata=("s", "t"), drugs=4, genes=6, seed=0):
    """A side of one model: contexts line|stratum, every perturbation in every context, three seeds' predictions."""
    rng = np.random.default_rng(seed)
    contexts = np.repeat([f"L{i}|{s}" for i in range(lines) for s in strata], drugs)
    n = len(contexts)
    truth = rng.uniform(0, 2, (n, genes))
    u = {"truth": truth, "context": contexts, "p_code": np.tile(np.arange(drugs), lines * len(strata)), "weight": rng.uniform(.5, 1.5, n)}
    return {"u": u, "seeds": {s: truth + rng.normal(0, .3, truth.shape) for s in SEEDS}, "T": truth + rng.normal(0, .5, truth.shape),
            "b": np.repeat(rng.uniform(0, 2, (n // drugs, genes)), drugs, axis=0), "outside": np.repeat(rng.uniform(0, 2, (n // drugs, genes)), drugs, axis=0),
            "lines": np.asarray([c.split("|")[0] for c in contexts]),
            "strata": np.asarray([c.split("|")[1] for c in contexts])}


class Slope(unittest.TestCase):
    def test_slope_recovers_a_known_scale(self):
        rng = np.random.default_rng(1)
        x, w = rng.normal(size=(20, 5)), rng.uniform(.5, 1.5, 20)
        self.assertAlmostEqual(slopes.slope(x, .3 * x, w), .3, places=12)

    def test_the_reference_moves_the_shared_response_and_line_needs_only(self):
        sd = side()
        outside, bag = slopes.layers(sd, "outside", 1), slopes.layers(sd, "b", 1)
        self.assertTrue(np.allclose(outside["perturbation"][1], bag["perturbation"][1]))
        self.assertFalse(np.allclose(outside["line"][1], bag["line"][1]))
        self.assertFalse(np.allclose(outside["response"][1], bag["response"][1]))

    def test_the_line_layer_is_constant_within_a_context(self):
        sd = side()
        x, _ = slopes.layers(sd, "b", 1)["line"]
        first = sd["u"]["context"] == sd["u"]["context"][0]
        self.assertTrue(np.allclose(x[first], x[first][0]))


class Judgment(unittest.TestCase):
    def test_the_slope_decides_the_judgment_and_the_fraction(self):
        for s, judgment, amount in ((-.1, "refuse", 0.), (0., "refuse", 0.), (.03, "no usable signal", 0.), (.2, "shrink", .2), (.5, "keep", 1.), (1.4, "keep", 1.)):
            self.assertEqual(slopes.judge(s), judgment)
            self.assertEqual(slopes.assigned(s), amount)

    def test_the_fraction_stays_between_zero_and_one(self):
        self.assertEqual([slopes.fraction(s) for s in (-1., .4, 3.)], [0., .4, 1.])

    def test_specificity_classes(self):
        self.assertEqual(slopes.context_class(.01, .5), "empty")
        self.assertEqual(slopes.context_class(.3, None), "not accepted")
        self.assertEqual(slopes.context_class(.3, .2), "specific")
        self.assertEqual(slopes.context_class(.3, .1), "generic")


class Specificity(unittest.TestCase):
    def test_a_need_shared_between_lines_of_a_stratum_reads_as_generic(self):
        sd = side(lines=2, strata=("s",), genes=8)
        shared = np.random.default_rng(3).normal(size=(4, 8))
        sd["u"]["truth"] = sd["T"] + np.tile(shared, (2, 1))
        sd["seeds"] = {s: sd["T"] + np.tile(shared, (2, 1)) for s in SEEDS}
        found = slopes.specificity(sd, 1.)
        self.assertAlmostEqual(found["m"], 1., places=10)
        self.assertAlmostEqual(found["S"], 0., places=10)

    def test_no_partner_gives_none(self):
        sd = side(lines=1, strata=("s",))
        self.assertEqual(slopes.specificity(sd, .3), {"slope_matched": None, "m": None, "S": None, "S_matched": None})


class LeaveOut(unittest.TestCase):
    def test_line_terms_add_to_the_seed_slope(self):
        sd = side()
        terms = slopes.line_terms(sd, "b")["perturbation"]
        ratio = sum(t[0] for t in terms.values()) / sum(t[1] for t in terms.values())
        self.assertAlmostEqual(float(np.mean(ratio)), slopes.seed_slopes(sd, "b")["perturbation"], places=10)

    def test_every_split_of_the_lines_is_read(self):
        terms = slopes.line_terms(side(lines=4), "b")["line"]
        self.assertEqual(len(list(slopes.mock_readings(terms))), 14)


class Gain(unittest.TestCase):
    def test_zero_amount_gains_nothing(self):
        sd = side()
        self.assertEqual(slopes.gain(sd, "line", 0.), 0.)

    def test_a_layer_that_closes_the_error_lowers_it(self):
        sd = side()
        sd["seeds"] = {s: sd["u"]["truth"].copy() for s in SEEDS}
        self.assertGreater(slopes.gain(sd, "perturbation", 1.), 0)


if __name__ == "__main__":
    unittest.main()
