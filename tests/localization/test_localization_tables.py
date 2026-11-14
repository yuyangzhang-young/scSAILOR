"""The definitions behind information localization's onset, planted-fault and light-version tables
(``reproduce.information_localization.tables``), on small hand-made inputs; no study data.

    python -m pytest tests/localization/test_localization_tables.py
"""
import unittest

from reproduce.information_localization import tables


def profile(energies):
    """A cue profile with one support: bit -> cut -> energy coefficient."""
    return {"supports": {"heldout": {"bits": {bit: {"cuts": {cut: {"energy_coefficient": {"value": v}}
                                                            for cut, v in cuts.items()}}
                                               for bit, cuts in energies.items()}}}}


def map_row(cut, arm, rate, seed, q, outcome="recovered"):
    return {"cut": cut, "arm": arm, "rate": str(rate), "seed": str(seed), "target_Q": str(q), "outcome": outcome,
            "run": f"map/{cut}-{arm}-x{rate}-{seed}"}


class Onset(unittest.TestCase):
    def test_the_selective_cue_against_the_shared_and_the_null_cue_at_each_key_cut(self):
        energies = {"1": {"input_cells:mean": .02, "backbone.sproj": .3}, "0": {"input_cells:mean": .04, "backbone.sproj": 0.},
                    "4": {"input_cells:mean": .01, "backbone.sproj": .1}}
        rows = tables.onset_rows("summary", 31002, 40000, profile(energies), {("summary", 31002, 40000, "heldout"): .5})
        self.assertEqual([r["cut"] for r in rows], ["input_cells:mean", "backbone.sproj"])
        first, second = rows
        self.assertEqual((first["Model"], first["selective_Q1"]), ("Summary-statistics", .5))
        self.assertAlmostEqual(first["selective_to_shared"], .5)
        self.assertAlmostEqual(first["selective_to_null"], 2.)
        self.assertEqual(second["selective_to_shared"], "")   # no shared cue at the cut
        self.assertAlmostEqual(second["selective_to_null"], 3.)


class PlantedFaultCell(unittest.TestCase):
    rows = [map_row("projections", "f", 1, 31002, .1), map_row("projections", "f", 30, 31002, .7),
            map_row("projections", "e", 1, 31002, .3), map_row("projections", "e", 1, 31003, .1),
            map_row("projections", "e", 30, 31002, .9, "diverged"), map_row("input", "a", 3, 31002, .2),
            map_row("projections", "a", 1, 31002, .05), map_row("projections", "a", 3, 31002, .25)]

    def test_a_starting_point_reads_its_best_seed_mean_over_cuts_and_rates(self):
        back = tables.planted_fault_cell(self.rows, "back")
        self.assertEqual(back["recovery"], .7)
        self.assertEqual(back["lowest_rate_reaching_0.25"], 30.)
        self.assertTrue(back["asterisk"])             # only x1 and x30 were run
        self.assertEqual(back["rates_run"], "1 30")
        front = tables.planted_fault_cell(self.rows, "front")   # a diverged run is left out; seeds are averaged
        self.assertAlmostEqual(front["recovery"], .2)
        self.assertEqual(front["lowest_rate_reaching_0.25"], "")
        whole = tables.planted_fault_cell(self.rows, "whole")
        self.assertEqual((whole["recovery"], whole["asterisk"]), (.2, False))

    def test_a_rate_whose_best_is_exactly_the_recovery_level_reaches_it(self):
        fine_tuning = tables.planted_fault_cell(self.rows, "ft")
        self.assertEqual((fine_tuning["recovery"], fine_tuning["lowest_rate_reaching_0.25"]), (.25, 3.))

    def test_the_original_model_averages_each_seeds_original_model(self):
        parent = {"map/projections-f-x1-31002": .2, "map/projections-e-x1-31003": .4}.get
        cell = tables.planted_fault_cell(self.rows, "orig", parent)
        self.assertAlmostEqual(cell["recovery"], .3)
        self.assertEqual((cell["asterisk"], cell["rates_run"]), (False, ""))


class LightHours(unittest.TestCase):
    @staticmethod
    def map_run(seed, cut, arm, rate, seconds, group=(), half="natural", case="parent"):
        return {"model": "state", "fold": "F1", "half": half, "seed": seed, "case": case, "cut": cut, "arm": arm,
                "rate": float(rate), "seconds": seconds, "device": "cuda", "group": list(group)}

    def test_full_and_light_hours_where_every_run_had_its_own_gpu(self):
        runs = [self.map_run(31002, "head", "a", 1, 3600), self.map_run(31002, "head", "a", 3, 7200),
                self.map_run(31003, "head", "a", 1, 1800), self.map_run(31004, "head", "a", 1, 9000)]
        light = {("natural", 31002, "head", "a", 1.), ("synthetic", 31002, "head", "a", 1.)}
        row = tables.gpu_hours(runs, light, [("natural", 31003, "head", "a", 1.)])
        self.assertEqual(row, {"full_runs": 3, "full_hours": 3.5, "light_runs": 3, "light_runs_in_map": 2, "light_hours": 1.5})

    def test_no_recorded_hours_where_runs_shared_a_gpu(self):
        runs = [self.map_run(31002, "head", "a", 1, 3600, group=("x", "y"))]
        row = tables.gpu_hours(runs, {("natural", 31002, "head", "a", 1.)}, [])
        self.assertEqual((row["full_hours"], row["light_hours"]), ("", ""))

    def test_the_estimate_is_the_median_hours_per_run_times_the_light_runs(self):
        runs = [self.map_run(31002, "a", "a", 1, 3600, group=("x", "y")), self.map_run(31002, "b", "a", 1, 7200),
                self.map_run(31003, "c", "a", 1, 3600 * 3, group=("x", "y", "z"))]
        self.assertEqual(tables.estimated_hours(runs, 10), (3, 1., 10.))

    def test_every_map_has_its_gpu(self):
        self.assertEqual({key for key in tables.GPU}, {(m, f) for m in ("summary", "deepsets", "state", "official_state")
                                                         for f in ("F1", "F2", "F3")} | {("cpa", "F1"), ("prepr", "F1")})


if __name__ == "__main__":
    unittest.main()
