"""Error handling of the assay; no study data, no model forwards.

* ``replay.write_predictions`` rejects a scope other than "full" and "base" before writing anything;
* ``capability.pool_seeds`` reports a bit with no true contrast (E = 0 in every seed, as for the mean
  bits when ``T_mean`` is False) as "NA" without Q statistics, and ``capability.null_leakage_table``
  leaves the leakage ratio undefined when the statistic's main bits have no true contrast;
* ``probe.run_probe`` rejects a split that does not cover the instance's training backgrounds.

    python -m pytest tests/assay/test_fixes.py
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scsailor.assay import capability, probe, replay, score
from scsailor.assay.constants import N_BITS

Q_STATISTICS = ("Q_mean", "Q_min", "Q_max", "direction_consistent", "deviation")
SEEDS = (31002, 31003, 31004)


class ReplayScopes(unittest.TestCase):
    def test_unknown_scope_is_rejected_before_any_file_is_written(self):
        instance = SimpleNamespace(name="none", record={"parameters": {}}, evaluation={"F1": [], "seen": []})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "predictions.h5"
            with self.assertRaises(ValueError) as raised:
                replay.write_predictions(None, instance, "L2", 40000, {"F1": "full", "seen": "flips"}, 16, output,
                                         "cpu")
            self.assertFalse(output.exists())
        for word in ("flips", "full", "base"):
            self.assertIn(word, str(raised.exception))


def fit_scores(zero_energy_bits):
    """One fit's SCORES.json content: bits in ``zero_energy_bits`` and the null bits have E = 0."""
    modules = []
    for a in range(N_BITS):
        E = 0. if a in zero_energy_bits or a % 5 == 4 else 1.
        A, K = .25, (0. if E == 0 else 1.)
        M = E + A - K
        modules.append({"module": a, "E": E, "M": M, "A": A, "K": K, **score._ratios(E, M, A, K)})
    return {"F1": {"modules": modules}, "seen": {"modules": modules}}


class NoTrueContrast(unittest.TestCase):
    """T_mean = False: the mean bits (0-4) move no answer, so every seed row has E = 0 and Q None."""

    @classmethod
    def setUpClass(cls):
        scores = {("deepsets", "L2", seed): (fit_scores(range(5)), "none") for seed in SEEDS}
        cls.rows = capability.seed_rows(scores, ["deepsets"], SEEDS, levels=("L2",))

    def test_pooled_mean_bits_are_na_without_q_statistics(self):
        pooled = {(row["support"], row["module_id"]): row for row in capability.pool_seeds(self.rows)}
        for support in ("heldout", "seen"):
            for a in range(5):
                row = pooled[(support, a)]
                self.assertEqual(row["status"], "NA", (support, a))
                self.assertEqual([row[k] for k in Q_STATISTICS], [None] * 5, (support, a))
                self.assertEqual([row[k] for k in ("null_ratio_mean", "null_ratio_min", "null_ratio_max")],
                                 [None] * 3, (support, a))
            self.assertEqual(pooled[(support, 5)]["status"], "ok")
            self.assertEqual(pooled[(support, 5)]["Q_mean"], .75)

    def test_null_leakage_is_undefined_without_main_contrast(self):
        rows = {(row["seed"], row["support"], row["module_id"]): row
                for row in capability.null_leakage_table(self.rows)}
        self.assertIsNone(rows[(SEEDS[0], "heldout", 4)]["A_over_E_main"])
        self.assertEqual(rows[(SEEDS[0], "heldout", 9)]["A_over_E_main"], .25)


class ProbeSplit(unittest.TestCase):
    def test_split_must_cover_the_training_backgrounds(self):
        instance = SimpleNamespace(train_contexts=[f"c{q}" for q in range(56)])
        design = probe.ProbeDesign(root=31001, draw_key=20260926, bags_per_context=8, split=(44, 4, 4))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "probe"
            with self.assertRaises(ValueError) as raised:
                probe.run_probe(None, instance, {"model_name": "deepsets", "level": "L2"}, design, output, "cpu")
            self.assertFalse(output.exists())
        self.assertIn("56", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
