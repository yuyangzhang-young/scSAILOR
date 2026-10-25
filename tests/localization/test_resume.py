"""A run paused at an update and resumed equals the same run made straight through.

A small CPU model, trained by the localization loop (``starts.train``) with a separately saved embedding (as a
drug-embedding copy is saved) and an evaluation every two updates; no study data. The paused run stops after
its checkpoint at update 3 (exit status ``starts.PAUSED``), a fresh process state resumes it, and weights,
copy, evaluations and curve must equal the straight run's; no evaluation is repeated.

    python -m pytest tests/localization/test_resume.py
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

UPDATES, EVERY = 8, 2


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
class PauseAndResume(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        from scsailor.localization import starts

        self.np, self.torch, self.starts = np, torch, starts
        torch.set_num_threads(1)

    def make(self):
        """The model, its separately saved embedding and the optimizer, from one fixed initialization."""
        torch = self.torch
        torch.manual_seed(12)
        copy = torch.nn.Embedding(4, 5)

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 5)
                self.copies = [copy]   # not a submodule: saved apart from the model's weights

            def forward(self, x, depth, p, level):
                return self.layer(x.mean(dim=1)) + self.copies[0](p)

        model = Toy()
        parameters = list(model.parameters()) + list(copy.parameters())
        return model, copy, parameters, torch.optim.AdamW(parameters, lr=.01)

    def batch(self, update):
        rng = self.np.random.default_rng(update)
        return (rng.normal(size=(2, 6, 3)).astype("float32"), self.np.zeros((2, 6), "int64"),
                rng.integers(4, size=2), rng.normal(size=(2, 5)).astype("float32"))

    def segment(self, out, checkpoint_class):
        model, copy, parameters, optimizer = self.make()
        checkpoint = checkpoint_class(out, {"name": "toy", "stage": "map"}, UPDATES)
        score = lambda current: {"weight_sum": float(current.layer.weight.detach().sum())}  # noqa: E731
        self.starts.train(model, optimizer, parameters, self.batch, lambda u: .01 / u, UPDATES, 1, out, checkpoint,
                          "cpu", copy=copy, every=EVERY, score=score)
        checkpoint.close()
        return model, copy

    def outputs(self, out, model, copy):
        return {"weights": {k: v.clone() for k, v in model.state_dict().items()}, "copy": copy.weight.detach().clone(),
                "trace": (out / "EVAL_TRACE.jsonl").read_text().splitlines(),
                "curve": [{k: v for k, v in json.loads(line).items() if k != "elapsed_s"}
                          for line in (out / "TRAIN_CURVE.jsonl").read_text().splitlines()]}

    def test_paused_and_resumed_equals_straight(self):
        starts, torch = self.starts, self.torch

        class StopAtThree(starts.RunCheckpoint):
            def due(self, update, updates):
                if update == 3:
                    self.stop_requested = True
                return super().due(update, updates)

        with tempfile.TemporaryDirectory() as temp:
            straight = Path(temp) / "straight"
            expected = self.outputs(straight, *self.segment(straight, starts.RunCheckpoint))
            paused = Path(temp) / "paused"
            with self.assertRaises(SystemExit) as stopped:
                self.segment(paused, StopAtThree)
            self.assertEqual(stopped.exception.code, starts.PAUSED)
            self.assertEqual(json.loads((paused / "CHECKPOINT.json").read_text())["update"], 3)
            found = self.outputs(paused, *self.segment(paused, starts.RunCheckpoint))
        self.assertEqual([json.loads(line)["update"] for line in found["trace"]], [2, 4, 6])
        self.assertEqual(found["trace"], expected["trace"])
        self.assertEqual(found["curve"], expected["curve"])
        self.assertTrue(torch.equal(found["copy"], expected["copy"]))
        for key, value in expected["weights"].items():
            self.assertTrue(torch.equal(found["weights"][key], value), key)


if __name__ == "__main__":
    unittest.main()
