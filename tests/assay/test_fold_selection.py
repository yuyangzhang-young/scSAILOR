"""The genetic-screen fold selection (``reproduce/assay/fold_selection.py``) on small hand-made tables; no study data.

    python -m pytest tests/assay/test_fold_selection.py
"""
import tempfile
import unittest
from pathlib import Path

from reproduce.assay import fold_selection


def write(root, screen, fold, no_change, anchor, noise):
    folder = Path(root) / fold_selection.REFERENCES_FOLDER / screen / fold
    folder.mkdir(parents=True)
    (folder / "REFERENCE_SCORES.csv").write_text(
        "heldout_anchor_raw_total_mse,heldout_anchor_raw_difference_error_mse,"
        f"heldout_input_only_raw_difference_error_mse\n9,{anchor},{no_change}\n")
    (folder / "NOISE_FLOOR.csv").write_text(f"total_mse,common_error_mse,difference_error_mse\n9,9,{noise}\n")


class Share(unittest.TestCase):
    def test_share_of_the_above_noise_difference_signal_explained_by_the_anchor_and_the_extremes(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "jiang", "F1", no_change=5., anchor=3., noise=1.)       # (5 - 3) / (5 - 1) = +50%
            write(tmp, "jiang", "F2", no_change=5., anchor=6., noise=1.)       # -25%
            write(tmp, "jiang", "F3", no_change=5., anchor=5., noise=1.)       # 0%
            table = fold_selection.fold_selection(tmp)
        got = table.set_index("fold")
        self.assertEqual(list(got.held_out_line), ["A549", "BXPC3", "HAP1"])
        self.assertEqual(list(got.anchor_share_percent), [50., -25., 0.])
        self.assertEqual(list(got.above_noise_signal), [4., 4., 4.])
        self.assertEqual(list(got.selected), ["largest share", "smallest share", ""])


if __name__ == "__main__":
    unittest.main()
