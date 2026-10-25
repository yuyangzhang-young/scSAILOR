"""Recovery on a small hand-made panel: no study data, no torch.

    python -m pytest tests/localization/test_recovery.py
"""
import unittest

import numpy as np

from scsailor.assay.constants import FIRST_SCORED_GENE
from scsailor.localization.recovery import pair_row, recovery, support_summary

GENES = FIRST_SCORED_GENE + 4


def endpoint(values):
    """An output vector whose scored genes hold ``values`` (the unscored genes are ignored by recovery)."""
    out = np.full(GENES, 9.0)
    out[FIRST_SCORED_GENE:] = values
    return out


TRUTH = (endpoint([1.0, 1.0, 1.0, 1.0]), endpoint([3.0, 1.0, 5.0, 1.0]))   # half-contrast (1, 0, 2, 0)


class Recovery(unittest.TestCase):
    def test_an_exact_contrast_recovers_one(self):
        row = pair_row("q1", 0, TRUTH, TRUTH)
        self.assertEqual(recovery([row])["Q"], 1.0)

    def test_no_predicted_contrast_recovers_zero(self):
        flat = (endpoint([2.0, 2.0, 2.0, 2.0]),) * 2
        self.assertEqual(recovery([pair_row("q1", 0, flat, TRUTH)])["Q"], 0.0)

    def test_the_contrast_is_read_after_clipping_at_zero_and_q_raw_before(self):
        # Unclipped contrast (1, 0, 2, 0) on the minus side shifted below zero: clipping halves gene 1's contrast.
        predictions = (endpoint([-1.0, 1.0, 1.0, 1.0]), endpoint([1.0, 1.0, 5.0, 1.0]))
        row = pair_row("q1", 0, predictions, TRUTH)
        result = recovery([row])
        self.assertAlmostEqual(result["Q"], 1 - (0.5 ** 2 / 4) / (5 / 4))
        self.assertEqual(result["Q_raw"], 1.0)

    def test_a_null_truth_has_no_recovery(self):
        null = (TRUTH[0], TRUTH[0])
        result = recovery([pair_row("q1", 0, TRUTH, null)])
        self.assertIsNone(result["Q"])
        self.assertEqual(result["status"], "null_target_report_absolute_leakage")

    def test_recovery_pools_errors_over_pairs_before_dividing(self):
        good = pair_row("q1", 0, TRUTH, TRUTH)
        flat = pair_row("q2", 1, (endpoint([2.0] * 4),) * 2, TRUTH)
        self.assertEqual(recovery([good, flat])["Q"], 0.5)

    def test_summary_splits_by_context_and_reads_raw_error_on_the_target_bit(self):
        good = pair_row("q1", 0, TRUTH, TRUTH)
        flat = pair_row("q2", 1, (endpoint([2.0] * 4),) * 2, TRUTH)
        summary = support_summary({1: [good, flat], 0: [flat, good]}, 1, ["q1", "q2"])
        per_context = summary["bits"]["1"]["per_context"]
        self.assertEqual((per_context["q1"]["Q"], per_context["q2"]["Q"]), (1.0, 0.0))
        self.assertEqual(summary["mse"]["n_endpoints"], 4)
        # The flat pair's endpoints miss the truth by (1, 1, 1, 1) and (1, 1, 3, 1): MSE 1 and 3.
        self.assertEqual(summary["mse"]["raw_O50"], np.mean([0.0, 0.0, 1.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
