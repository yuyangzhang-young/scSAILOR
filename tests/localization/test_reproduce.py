"""The paper's localization settings (``reproduce/information_localization``): importing them loads no torch, the
parents' learning rate is the assay's recipe and the natural task's schedule, the panels have the paper's sizes,
score and profile file names are read under both names of the attention model, and a parent saved under its study
name, ``state``, loads as ``attention_model``.

    python -m pytest tests/localization/test_reproduce.py
"""
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from reproduce.assay.study import PEAK_LR
from reproduce.information_localization import tables, tahoe
from scsailor.natural.data import lr_at

ROOT = Path(__file__).resolve().parents[2]
CHECK = """
import json, sys
import reproduce.information_localization.tahoe, reproduce.information_localization.planted_faults
import reproduce.information_localization.time_and_presence, reproduce.information_localization.tables
print(json.dumps({"torch": "torch" in sys.modules}))
"""


class Settings(unittest.TestCase):
    def test_import_loads_no_torch(self):
        done = subprocess.run([sys.executable, "-c", CHECK], capture_output=True, text=True, check=True, cwd=ROOT)
        self.assertEqual(json.loads(done.stdout), {"torch": False})

    def test_parent_rate_is_the_assay_recipe(self):
        # past the warm-up: a cosine from the peak at update 1,000 to a tenth of it at 160,000
        for name, peak in (("summary", 1e-4), ("attention_model", 3e-5)):
            self.assertAlmostEqual(tahoe.parent_schedule(name, 1000), peak, delta=1e-20)
            self.assertAlmostEqual(tahoe.parent_schedule(name, 160000), peak / 10, delta=1e-20)
            self.assertAlmostEqual(tahoe.parent_schedule(name, 40000),
                                   peak * (.1 + .45 * (1 + math.cos(math.pi * 39000 / 159000))), delta=1e-20)

    def test_natural_schedule_is_the_parents(self):
        # the map multiplies the rate at 40,000 updates (8.7288e-5 and 2.6186e-5); a cosine ending at 40,000 gives 1e-5
        for name in PEAK_LR:
            for step in (1, 1000, 1001, 40000, 160000):
                self.assertEqual(lr_at(step, PEAK_LR[name]), tahoe.parent_schedule(name, step))
        self.assertAlmostEqual(tahoe.parent_schedule("summary", 40000), 8.7288e-5, delta=1e-9)

    def test_panels(self):
        # Tahoe-100M: 56 training contexts and 28 held-out contexts per fold
        task = SimpleNamespace(train_contexts=[f"t{i}" for i in range(56)], heldout_contexts=[f"h{i}" for i in range(28)])
        panels = tahoe.panels(task, "map")
        self.assertEqual({k: (v.pairs, len(v.contexts)) for k, v in panels.items()},
                         {"seen": (32, 8), "heldout": (56, 28), "validation": (112, 56)})
        self.assertEqual(list(tahoe.panels(task, "pilot")), ["seen"])
        self.assertEqual(list(tahoe.evaluation_panels(task)), ["seen", "heldout"])

    def test_file_names(self):
        for model in ("state", "attention_model"):
            self.assertEqual(tables.score_name(f"EVAL_{model}_31004_80000"), ("state", "31004", "80000"))
            self.assertEqual(tables.profile_case(f"PRESENCE_synthetic_{model}_31002", 31002), "parent")
        self.assertEqual(tables.score_name("EVAL_official-state-authors_31002_5000"),
                         ("official-state-authors", "31002", "5000"))
        self.assertEqual(tables.profile_case("PRESENCE_synthetic_summary_31002_blocked-mean-cue", 31002),
                         "blocked-mean-cue")


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class SavedName(unittest.TestCase):
    def test_parent_saved_as_state_loads_as_attention_model(self):
        import numpy as np
        import torch

        from scsailor.models import RawModel

        rng = np.random.default_rng(0)
        model = RawModel("attention_model", rng.normal(size=2000).astype(np.float32), 4,
                         rng.normal(size=(4, 2000)).astype(np.float32), rng.normal(size=(3, 2000)).astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parent.pt"
            torch.save({"model": model.state_dict(), "model_name": "state", "seed": 31002, "level": "L2",
                        "step": 40000}, path)
            loaded, record = tahoe.load_parent(path, "attention_model", 31002)
            with self.assertRaises(ValueError):
                tahoe.load_parent(path, "attention_model", 31003)
        self.assertEqual(record["model_name"], "state")
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
