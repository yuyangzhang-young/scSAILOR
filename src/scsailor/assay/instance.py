"""One synthetic-answer instance on disk.

An instance directory holds ``INSTANCE.json`` (backgrounds, generator
parameters, evaluation support), ``generator.npz``, ``input_pools.h5``
(control cells per background, log1p CP10k on the 2000-gene panel),
``evaluation_bags.h5`` (the fixed 256-cell bags) and ``truth.h5`` (base
states and known answers on the evaluation bags).

The evaluation support has two sides: ``HELDOUT`` (the held-out backgrounds; saved as "F1" in every
fold of every dataset) and ``SEEN`` (training backgrounds). In ``INSTANCE.json`` an evaluation entry's
``q`` is the background's index in its side's list (``context_index`` in code) and ``root`` starts the
keys of its base states and replay draws.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .generator import Generator, Parameters

HELDOUT, SEEN = "F1", "seen"

__all__ = ["Instance", "EvaluationBag", "HELDOUT", "SEEN"]


@dataclass(frozen=True)
class EvaluationBag:
    """One evaluation background's fixed bag and its base states."""
    context: str
    context_index: int    # the background's index in its side's list (saved key "q")
    root: int             # first entry of the keys of its base states and replay draws
    x: np.ndarray         # 256 x genes
    depth: np.ndarray     # 256
    states: np.ndarray    # bases x 26 x 25: each base state and its 25 single-bit flips


class Instance:
    def __init__(self, inputs):
        self.inputs = Path(inputs)
        self.record = json.loads((self.inputs / "INSTANCE.json").read_text())
        self.name = self.record["instance"]
        self.params = Parameters(**self.record["parameters"])
        self.generator = Generator(self.inputs / "generator.npz", self.params)

    @property
    def train_contexts(self):
        return self.record["train_contexts"]

    @property
    def heldout_contexts(self):
        return self.record["F1_contexts"]   # saved key: the held-out side is "F1" in every fold

    @property
    def evaluation(self):
        """Evaluation backgrounds per side (``HELDOUT``, ``SEEN``): context, q, root, pool path, bag key."""
        return self.record["evaluation"]

    def evaluation_bags(self, side):
        """The evaluation bags of one side, in the instance's order, each read when reached."""
        for entry in self.evaluation[side]:
            x, depth = self.bag(side, entry["context"])
            yield EvaluationBag(entry["context"], entry["q"], entry["root"], x, depth,
                                self.states(side, entry["context"]))

    def pool(self, group, context):
        """All control cells of one background as ``(x, depth)``; ``group`` is "train" or ``HELDOUT``."""
        with h5py.File(self.inputs / "input_pools.h5", "r") as bank:
            return bank[f"{group}/{context}/x"][:], bank[f"{group}/{context}/depth"][:]

    def bag(self, side, context):
        """The fixed evaluation bag of one background as ``(x, depth)``."""
        with h5py.File(self.inputs / "evaluation_bags.h5", "r") as bags:
            return bags[f"{side}/{context}/x"][:], bags[f"{side}/{context}/depth"][:]

    def states(self, side, context):
        """Base states (4 x 26 x 25) of one evaluation bag: each base state and its 25 single-bit flips."""
        with h5py.File(self.inputs / "truth.h5", "r") as truth:
            return truth[f"{side}/{context}/states"][:]

    def truth(self, side, context):
        """Base states (4 x 26 x 25) and known answers (4 x 26 x perturbations x genes) on one evaluation bag."""
        with h5py.File(self.inputs / "truth.h5", "r") as truth:
            return truth[f"{side}/{context}/states"][:], truth[f"{side}/{context}/truth"][:]
