"""The synthetic task as localization draws from it: one synthetic-answer instance (``scsailor.assay.Instance``).

``draws``, ``recovery``, ``presence`` and ``starts`` take a task with:

- ``generator``, ``n_perturbations``, ``train_contexts``, ``heldout_contexts``: the instance's;
- ``pool(group, context)``: the control cells of one background (``group`` "train" or the held-out "F1") as
  ``(x, depth, background)``, with ``background`` its ``scsailor.assay.generator.Background``; read once and kept;
- ``carrier(x, depth, state, level, rng)``: one level's model input from a bag of cells under a planted state,
  with the instance's parameters (``scsailor.assay.carrier.carrier``);
- ``stream()``: the instance's training stream (``scsailor.assay.stream.TrainingStream``); one ``batch`` call gives
  eight bags.
"""
from __future__ import annotations

from ..assay.carrier import carrier
from ..assay.generator import Background
from ..assay.instance import Instance
from ..assay.stream import TrainingStream

__all__ = ["Task"]


class Task:
    def __init__(self, inputs):
        self.instance = Instance(inputs)
        self.record, self.params, self.generator = self.instance.record, self.instance.params, self.instance.generator
        self._pools = {}
        self._stream = None

    @property
    def train_contexts(self):
        return self.instance.train_contexts

    @property
    def heldout_contexts(self):
        return self.instance.heldout_contexts

    @property
    def n_perturbations(self):
        return len(self.generator.pi)

    def pool(self, group, context):
        key = (group, context)
        if key not in self._pools:
            x, depth = self.instance.pool(group, context)
            self._pools[key] = (x, depth, Background(x))
        return self._pools[key]

    def carrier(self, x, depth, state, level, rng):
        return carrier(x, depth, state, level, rng, self.params)

    def stream(self):
        if self._stream is None:
            self._stream = TrainingStream(self.instance)
        return self._stream
