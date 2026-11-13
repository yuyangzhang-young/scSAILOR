"""The paper's assay settings (``reproduce/assay/study.py``): importing them loads no torch, and a checkpoint saved
under the study's name of the attention model, ``state``, loads as ``attention_model`` with its weights.

    python -m pytest tests/assay/test_study.py
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_tiny_instance import write_instance

from reproduce.assay import study

ROOT = Path(__file__).resolve().parents[2]
CHECK = """
import json, sys
import reproduce.assay.study
print(json.dumps({"torch": "torch" in sys.modules}))
"""


class Import(unittest.TestCase):
    def test_import_loads_no_torch(self):
        done = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, check=True, cwd=ROOT)
        self.assertEqual(json.loads(done.stdout), {"torch": False})


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class SavedName(unittest.TestCase):
    def test_attention_model_saved_as_state_loads_under_its_public_name(self):
        import torch

        from scsailor.assay import train

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            directory = root / study.DATASETS["tahoe"] / "F1" / "synthetic"
            directory.mkdir(parents=True)
            write_instance(directory, np.random.default_rng(0))
            instance = study.load_instance(root, "tahoe", "F1")
            model = study.make_model(instance.generator, "attention_model", 31002, "cpu")
            saved_fit = train.Fit("state", 31002, "L2", study.PEAK_LR["attention_model"], steps=0, total=1)
            train.save_checkpoint(Path(tmp) / "step0.pt", model, saved_fit, 0, instance)
            loaded, record = study.load_checkpoint(instance, Path(tmp) / "step0.pt", "cpu")
        self.assertEqual(record["model_name"], "attention_model")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
