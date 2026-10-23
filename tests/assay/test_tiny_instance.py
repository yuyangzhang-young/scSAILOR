"""The assay on a tiny instance built from random arrays: no study data, no network.

Four random backgrounds (three for training, one held out) are written to a temporary directory with
the package's own arithmetic (``calibrate``, ``bag_indices``, ``truth_table``) and plain h5py writes.
The instance must then give a training batch, and predictions equal to the truth must score R = 0 and
Q = 1 (None for the null bits). It takes a few seconds:

    python -m pytest tests/assay/test_tiny_instance.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scsailor.assay import Instance, generator, score
from scsailor.assay.constants import N_BITS, N_CELLS, N_GENES, N_ROWS
from scsailor.assay.instance import HELDOUT, SEEN
from scsailor.assay.stream import TrainingStream

CELLS = 300                  # control cells per background; a bag takes 256 of them
PERTURBATIONS = 6            # five centred response directions
TRAIN, HELD_OUT = ("t0", "t1", "t2"), ("h0",)
EVALUATION = {HELDOUT: [{"context": "h0", "q": 0, "root": 41001, "pool_path": "F1/h0", "bag_key": [41001, 0, 999]}],
              SEEN: [{"context": "t1", "q": 1, "root": 51001, "pool_path": "train/t1", "bag_key": [51001, 1, 999]}]}


def write_instance(directory, rng):
    """Pools, a generator calibrated on random treated profiles, the record, evaluation bags and truth."""
    pools = {}
    with h5py.File(directory / "input_pools.h5", "w") as bank:
        for group, contexts in (("train", TRAIN), (HELDOUT, HELD_OUT)):
            for context in contexts:
                x = rng.gamma(2., .5, size=(CELLS, N_GENES)).astype(np.float32)
                depth = rng.integers(1000, 20000, size=CELLS)
                bank[f"{group}/{context}/x"], bank[f"{group}/{context}/depth"] = x, depth
                pools[f"{group}/{context}"] = x, depth
    means = np.stack([pools[f"train/{context}"][0].mean(axis=0, dtype=np.float64) for context in TRAIN])
    parent = np.repeat(np.arange(len(TRAIN)), PERTURBATIONS)
    pcode = np.tile(np.arange(PERTURBATIONS), len(TRAIN))
    target = (means[parent] + rng.normal(0, .3, size=(PERTURBATIONS, N_GENES))[pcode]
              + rng.normal(0, .05, size=(len(parent), N_GENES)))
    weight = np.full(len(parent), 1 / len(parent))
    pi = generator.perturbation_weights(pcode, weight, PERTURBATIONS)
    r = generator.response_field(target, parent, pcode, weight, means, PERTURBATIONS)
    fields, _ = generator.calibrate(r, pi, means)
    np.savez(directory / "generator.npz", **fields, intercept=np.zeros(N_GENES),
             p_ids=np.array([f"p{i}" for i in range(PERTURBATIONS)]))
    (directory / "INSTANCE.json").write_text(json.dumps(
        {"instance": "portable", "parameters": generator.Parameters().record(), "train_contexts": list(TRAIN),
         "F1_contexts": list(HELD_OUT), "evaluation": EVALUATION}))
    calibrated = generator.Generator(directory / "generator.npz")
    with h5py.File(directory / "evaluation_bags.h5", "w") as bags, h5py.File(directory / "truth.h5", "w") as truth:
        for side, entries in EVALUATION.items():
            for entry in entries:
                x, depth = pools[entry["pool_path"]]
                indices = generator.bag_indices(len(depth), entry["bag_key"])
                path = f"{side}/{entry['context']}"
                bags[f"{path}/x"], bags[f"{path}/depth"] = x[indices], depth[indices]
                states, values = generator.truth_table(generator.Background(x), calibrated, entry["root"], entry["q"])
                truth[f"{path}/states"], truth[f"{path}/truth"] = states, values


class TinyInstance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        write_instance(cls.directory, np.random.default_rng(0))
        cls.instance = Instance(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_training_stream_draws_an_l2_batch(self):
        batch = TrainingStream(self.instance).batch(1, N_CELLS, "L2", 31002)
        self.assertEqual([v.shape for v in batch], [(8, N_ROWS, N_GENES), (8, N_ROWS), (8,), (8, N_GENES)])
        self.assertEqual([v.dtype for v in batch], [np.float32, np.int64, np.int64, np.float32])

    def test_predictions_equal_to_the_truth_score_perfectly(self):
        path = self.directory / "predictions.h5"
        with h5py.File(path, "w") as out:
            out.attrs["instance"] = self.instance.name
            out.attrs["parameters"] = json.dumps(self.instance.record["parameters"], sort_keys=True)
            for side, entries in self.instance.evaluation.items():
                for entry in entries:
                    _, truth = self.instance.truth(side, entry["context"])
                    group = out.create_group(f"{side}/{entry['context']}")
                    group["raw"], group["anchor_raw"] = truth, np.zeros_like(truth)
                    group["done"] = np.ones(truth.shape[:2], dtype=bool)
        scores, _ = score.score_predictions(self.instance, path)
        for side in (HELDOUT, SEEN):
            self.assertEqual(scores[side]["R"], 0)
            self.assertEqual([m["Q"] for m in scores[side]["modules"]],
                             [None if a % 5 == 4 else 1 for a in range(N_BITS)])


if __name__ == "__main__":
    unittest.main()
