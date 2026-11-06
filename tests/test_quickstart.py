"""The quickstart in ``examples/`` runs end to end, at a few updates per step (about 10 seconds on a CPU).

It runs in its own process, because it registers its model's cuts and sets torch's threads.

    python -m pytest tests/test_quickstart.py
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class Quickstart(unittest.TestCase):
    def test_every_step_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            subprocess.run([sys.executable, str(QUICKSTART), "--out", str(out), "--updates", "2", "--start-updates", "2",
                            "--pairs", "2"], check=True)
            self.assertTrue((out / "assay" / "scores" / "SCORES.json").is_file())
            self.assertTrue(all((out / "localization" / arm / "TRAIN_CURVE.jsonl").is_file() for arm in ("a", "e")))


if __name__ == "__main__":
    unittest.main()
