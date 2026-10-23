"""The training stream: eight bags with their synthetic targets per update, drawn from the training backgrounds only.

Update ``step`` of a stream with seed ``seed`` draws, from generators seeded by
``(seed, step, ...)``, eight backgrounds (with replacement, uniformly), eight
perturbations (with the training weights pi) and, for each bag, 256 cells, a
random 25-bit state and the carrier's technical randomness (keys in ``constants``).
The label is the known answer of that background, perturbation and state. A batch
therefore depends only on its step and the stream seed, whatever order batches are
drawn in.
"""
from __future__ import annotations

import numpy as np

from .carrier import carrier
from .constants import CARRIER_KEY, CELLS_KEY, N_BITS, N_GENES, N_ROWS, STATE_KEY
from .generator import Background, known_answer

BAGS = 8   # bags per update

__all__ = ["TrainingStream"]


class TrainingStream:
    def __init__(self, instance):
        contexts = instance.train_contexts
        if set(contexts) & set(instance.heldout_contexts):
            raise ValueError("Training and held-out backgrounds overlap")
        self.params = instance.params
        self.generator = instance.generator
        self.pools = [instance.pool("train", context) for context in contexts]
        self.backgrounds = [Background(x) for x, _ in self.pools]

    def batch(self, step, m, level, seed):
        """Cells (8 x 512 x genes, float32), depths (8 x 512), perturbations (8) and labels (8 x genes, float32)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 0]))
        contexts = rng.choice(len(self.pools), size=BAGS, replace=True)
        p = rng.choice(len(self.generator.p_ids), size=BAGS, p=self.generator.pi)
        x = np.empty((BAGS, N_ROWS, N_GENES), dtype=np.float32)
        depth = np.empty((BAGS, N_ROWS), dtype=np.int64)
        labels = np.empty((BAGS, N_GENES), dtype=np.float32)
        for slot, context_index in enumerate(contexts):
            pool, d = self.pools[context_index]
            cell_indices = np.random.default_rng(np.random.SeedSequence([seed, step, slot, CELLS_KEY])).permutation(
                len(pool))[:m]
            state = np.random.default_rng(np.random.SeedSequence([seed, step, slot, STATE_KEY])).choice([-1, 1],
                                                                                                        size=N_BITS)
            technical = np.random.default_rng(np.random.SeedSequence([seed, step, slot, CARRIER_KEY]))
            x[slot], depth[slot] = carrier(pool[cell_indices], d[cell_indices], state, level, technical, self.params)
            labels[slot] = known_answer(self.generator, self.backgrounds[context_index], [p[slot]], state[None])[0]
        return x, depth, p.astype(np.int64), labels
