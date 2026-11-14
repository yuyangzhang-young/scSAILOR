"""The paired-input isolation check (``reproduce/assay/checks.py``) on the tiny random instance; no study data.

    python -m pytest tests/assay/test_checks.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from test_tiny_instance import write_instance

from reproduce.assay import checks
from scsailor.assay import Instance, carrier


class CarrierCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        write_instance(Path(cls.temporary.name), np.random.default_rng(0))
        cls.table = checks.carrier_check(Instance(cls.temporary.name))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_one_row_per_input_set_of_both_sides(self):
        self.assertEqual(list(zip(self.table.side, self.table.context)), [("held-out", "h0"), ("training", "t1")])

    def test_a_flip_changes_nothing_but_its_cue_and_nothing_at_all_below_the_cues_first_level(self):
        for column in ("largest_output_difference", "largest_depth_difference", "largest_difference_below_first_level"):
            self.assertEqual(set(self.table[column]), {0.})
        self.assertEqual(set(self.table.pairs), {25 * 6})
        self.assertEqual(set(self.table.pairs_below_first_level), {5 * (1 + 2 + 3 + 4 + 5)})
        self.assertEqual(set(self.table.decoded_wrong), {0})

    def test_a_change_to_an_output_gene_would_be_seen(self):
        x, depth = np.random.default_rng(2).gamma(2., .5, (256, 2000)), np.arange(256) + 1000
        leaky = carrier.carrier

        def carrier_leaking(x, depth, state, level, rng, params):
            rows, row_depth = leaky(x, depth, state, level, rng, params)
            rows[:, 60] += state[0]                                  # the shift state leaks into gene 60
            return rows, row_depth

        with mock.patch.object(checks, "carrier", carrier_leaking):
            row = checks.paired_inputs(x, depth, Instance(self.temporary.name).params)
        self.assertAlmostEqual(row["largest_output_difference"], 2., places=5)   # float32 inputs


if __name__ == "__main__":
    unittest.main()
