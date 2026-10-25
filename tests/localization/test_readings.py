"""The reading rules on small hand-written tables: no study data, no torch.

    python -m pytest tests/localization/test_readings.py
"""
import unittest

from scsailor.localization import readings


def runs(*pairs):
    """Map runs of one arm as ``(rate, outcome)`` pairs."""
    return [{"rate": float(rate), "outcome": outcome} for rate, outcome in pairs]


NOT = "not recovered"
GRID_NOT = runs((1, NOT), (3, NOT), (10, NOT), (30, NOT))
# A planted fault that kills the summary-statistics interaction branch: only the back reset may recover the target.
DEAD_BACK = ("projections", {"back"}, {"drug side", "front", "mean projection", "variance projection"}, set())
SETTINGS = readings.Settings(map_seeds=(31002, 31003), junction={"summary": "projections", "deepsets": "phi"},
                             planted={"dead-interaction": DEAD_BACK})


def evaluation(error, **q):
    """One support of a saved evaluation: raw target-endpoint error and Q per bit."""
    return {"mse": {"raw_O50": error}, "bits": {bit: {"Q": value} for bit, value in q.items()}}


def cell(seed, cut, arm, rate=30.0, validation_q=0.0, validation_margins=True, model="summary", half="synthetic",
         case="parent", outcome=NOT, start=NOT):
    """One map cell, as ``readings.cell`` reads it from a finished run."""
    return {"model": model, "half": half, "seed": seed, "case": case, "cut": cut, "arm": arm, "rate": rate,
            "run": f"{model}-{half}-{seed}-{cut}-{arm}-x{rate:g}", "outcome": outcome, "parent_start": start,
            "target_Q": 0.5, "target_low": 0.4, "target_high": 0.6, "validation_target_Q": validation_q,
            "validation_margins_met": validation_margins}


class Recovery(unittest.TestCase):
    def test_recovered_when_the_lower_bound_reaches_the_level(self):
        self.assertEqual(readings.RECOVERY_LEVEL, 0.25)
        self.assertEqual(readings.interval_reading(0.25, 0.9), "recovered")
        self.assertEqual(readings.interval_reading(0.2499, 0.9), "undetermined")

    def test_not_recovered_only_when_the_upper_bound_is_below_the_level(self):
        self.assertEqual(readings.interval_reading(-0.1, 0.2499), "not recovered")
        self.assertEqual(readings.interval_reading(-0.1, 0.25), "undetermined")


class Margins(unittest.TestCase):
    parent = evaluation(1.0, **{"0": 0.9, "1": 0.0, "5": 0.4})

    def test_raw_error_may_end_at_twice_the_parents(self):
        self.assertTrue(readings.margins(self.parent, evaluation(2.0, **{"0": 0.9, "5": 0.4}), "synthetic", "1"))
        self.assertFalse(readings.margins(self.parent, evaluation(2.01, **{"0": 0.9, "5": 0.4}), "synthetic", "1"))

    def test_the_common_mean_may_end_at_most_a_tenth_below_the_parent(self):
        self.assertTrue(readings.margins(self.parent, evaluation(1.0, **{"0": 0.85, "5": 0.4}), "synthetic", "1"))
        self.assertFalse(readings.margins(self.parent, evaluation(1.0, **{"0": 0.75, "5": 0.4}), "synthetic", "1"))

    def test_a_bit_the_parent_has_not_learned_carries_no_margin(self):
        self.assertTrue(readings.margins(self.parent, evaluation(1.0, **{"0": 0.9, "5": 0.0}), "synthetic", "1"))
        learned = evaluation(1.0, **{"0": 0.9, "1": 0.0, "5": 0.6})
        self.assertFalse(readings.margins(learned, evaluation(1.0, **{"0": 0.9, "5": 0.4}), "synthetic", "1"))

    def test_the_natural_half_keeps_only_the_error_margin(self):
        self.assertTrue(readings.margins(self.parent, evaluation(2.0, **{"0": 0.0, "5": 0.0}), "natural", "1"))


class Contrasts(unittest.TestCase):
    def test_kept_arm_recovers_at_any_rate(self):
        kept = runs((1, NOT), (30, "recovered with collateral damage"))
        self.assertEqual(readings.contrast(kept, GRID_NOT)[0], "kept arm recovers")

    def test_reset_benefit_needs_the_kept_arms_whole_grid_and_a_recovery_at_a_matched_rate(self):
        reset = runs((1, NOT), (30, "recovered"))
        self.assertEqual(readings.contrast(GRID_NOT, reset), ("reset benefit", {1.0, 30.0}))
        self.assertEqual(readings.contrast(runs((1, NOT), (30, NOT)), reset)[0], "undetermined")

    def test_neither_recovers_needs_both_grids(self):
        self.assertEqual(readings.contrast(GRID_NOT, GRID_NOT)[0], "neither recovers")
        self.assertEqual(readings.contrast(GRID_NOT, runs((1, NOT), (30, NOT)))[0],
                         "neither recovers at the rates run")

    def test_a_reset_recovery_only_at_an_unmatched_rate_is_undetermined(self):
        kept = runs((1, NOT), (3, "diverged"), (10, NOT), (30, NOT))
        reset = runs((1, NOT), (3, "recovered"))
        self.assertEqual(readings.contrast(kept, reset), ("undetermined", {1.0}))

    def test_rate_gated_recovery_is_read_against_an_evaluable_x1(self):
        self.assertEqual(readings.rate_gated(runs((1, NOT), (30, "recovered"))), "yes")
        self.assertEqual(readings.rate_gated(runs((1, "recovered"), (30, "recovered"))), "no")
        self.assertEqual(readings.rate_gated(runs((1, "diverged"), (30, "recovered"))), "no evaluable x1")
        self.assertEqual(readings.rate_gated(runs((1, "undetermined"), (30, "recovered"))), "x1 undetermined")
        self.assertEqual(readings.rate_gated(GRID_NOT), "")


class SolvedAtTheStart(unittest.TestCase):
    def test_a_parent_recovered_at_the_start_is_solved_and_its_contrasts_are_intervention_effects(self):
        cells = [cell(31002, "projections", arm, rate, outcome=NOT, start="recovered")
                 for arm in ("a", "e") for rate in readings.GRID]
        self.assertEqual(readings.cases(cells)[0]["parent_derived_arms"], "solved at the start")
        (row,) = readings.contrasts(cells)
        self.assertEqual((row["contrast"], row["outcome"]), ("front", "neither recovers"))
        self.assertEqual(row["reading"], "intervention effect on a parent solved at the start")


class Carried(unittest.TestCase):
    def test_the_best_mean_validation_recovery_among_cells_qualified_in_both_seeds_is_carried(self):
        cells = [cell(seed, "projections", "e", validation_q=q) for seed, q in ((31002, 0.8), (31003, 0.7))]
        cells += [cell(seed, "projections", "e_mean", validation_q=q) for seed, q in ((31002, 0.9), (31003, 0.85))]
        cells += [cell(31002, "projections", "f", validation_q=0.99), cell(31003, "projections", "f", validation_q=0.1)]
        cells += [cell(seed, "projections", "d", validation_q=0.99, validation_margins=seed == 31002)
                  for seed in (31002, 31003)]
        # Never handed on: the cut input, the reference arm and the fresh network.
        cells += [cell(seed, "input", arm, validation_q=0.99) for seed in (31002, 31003) for arm in ("a", "fresh")]
        cells += [cell(seed, "projections", "interaction_e_mean", validation_q=0.99) for seed in (31002, 31003)]
        rows, _ = readings.carried(cells, SETTINGS)
        (row,) = rows
        self.assertEqual((row["carried"], row["kind"]), ("projections/e_mean/x30", "qualified repair"))
        self.assertAlmostEqual(row["mean_validation_target_Q"], 0.875)

    def test_the_fixed_intervention_is_handed_on_when_no_cell_qualifies(self):
        cells = [cell(seed, "phi", arm, rate, validation_q=0.1, model="deepsets")
                 for seed in (31002, 31003) for arm in ("a", "e") for rate in (1.0, 30.0)]
        rows, _ = readings.carried(cells, SETTINGS)
        (row,) = rows
        self.assertEqual((row["carried"], row["kind"]),
                         ("phi/a/x1", "diagnostic intervention (qualified repair: none)"))


class PlantedFaults(unittest.TestCase):
    clean = {"drug side": "neither recovers", "front": "neither recovers", "mean projection": "neither recovers",
             "variance projection": "neither recovers"}

    def test_pass_when_the_required_contrast_reads_reset_benefit_and_the_excluded_ones_neither_recovers(self):
        found = {**self.clean, "back": "reset benefit"}
        self.assertEqual(readings.planted_verdict(found, DEAD_BACK, "recovers", False), "pass")

    def test_miss_when_the_required_contrast_concludes_without_a_reset_benefit(self):
        found = {**self.clean, "back": "neither recovers"}
        self.assertEqual(readings.planted_verdict(found, DEAD_BACK, "recovers", False), "miss")


if __name__ == "__main__":
    unittest.main()
