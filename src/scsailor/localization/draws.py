"""Draws of the synthetic task: flip pairs, profile bags and training batches.

A draw takes ``N_CELLS`` control cells of one background, assigns a planted state (25 signs) and a
perturbation, builds the model input with the task's carrier, and computes the synthetic target. Each draw's
random generator is seeded from a root and the draw's position, so identical arguments give identical
draws. The draw keys are fixed: changing one changes every panel and profile.

* ``flip_pairs``: pair ``i`` of a recovery panel, keyed ``(root, bit, i)``, cycling through the contexts;
* ``profile_bags``: ``per_source[j]`` bags of source ``j``, keyed ``(root, j, i, 0)`` for a single bag and
  ``(root, j, i, bit + 1)`` for a flip pair of ``bit``;
* ``training_batch``: consecutive native batches of a training stream (``native_indices``).

Inside a draw the generator gives, in this order: the state, the perturbation, the cells, and the
carrier's seed (an integer below ``2**63 - 1``). A flip pair shares all four.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..assay.constants import N_BITS, N_CELLS
from ..assay.generator import known_answer
from .constants import BAGS_PER_NATIVE_BATCH, LEVEL

__all__ = ["Endpoint", "Pair", "seeded", "make_endpoint", "draw", "flip_pairs", "profile_bags", "native_indices",
           "training_batch"]

CARRIER_SEED_BOUND = 2**63 - 1


@dataclass
class Endpoint:
    """One model input with its synthetic target."""
    x: torch.Tensor        # carrier rows, shape (1, 512, genes)
    depth: torch.Tensor    # carrier depths, shape (1, 512)
    p: int                 # perturbation index
    truth: np.ndarray      # synthetic target, shape (genes,)
    state: np.ndarray      # planted state, 25 signs
    context: str           # background
    code: float            # planted drug code of p


@dataclass
class Pair:
    """Two endpoints that differ only in the sign of one planted state bit."""
    minus: Endpoint
    plus: Endpoint
    bit: int


def seeded(*parts):
    """A PCG64 generator seeded from integer parts."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(parts))))


def make_endpoint(task, context, pool, state, p, cell_indices, carrier_seed, level=LEVEL):
    """The input and synthetic target of one bag of ``pool`` (``task.pool``: cells, depths and their
    ``scsailor.assay.generator.Background``) under ``state`` and perturbation ``p``."""
    import torch

    x, depth, background = pool
    state = np.asarray(state, dtype=np.int64)
    rows, depths = task.carrier(x[cell_indices], depth[cell_indices], state, level, np.random.default_rng(carrier_seed))
    truth = known_answer(task.generator, background, [p], state[None].astype(np.float64))[0]
    return Endpoint(torch.from_numpy(rows)[None], torch.from_numpy(depths)[None], p, truth, state,
                    context, float(task.generator.code[p]))


def draw(task, context, pool, rng, bit):
    """Draw state, perturbation, cells and carrier seed; return one endpoint, or a flip pair of ``bit``."""
    state = rng.choice([-1, 1], size=N_BITS)
    p = int(rng.integers(task.n_perturbations))
    cell_indices = rng.permutation(len(pool[0]))[:N_CELLS]
    carrier_seed = int(rng.integers(CARRIER_SEED_BOUND))
    if bit is None:
        return p, state, [make_endpoint(task, context, pool, state, p, cell_indices, carrier_seed)]
    ends = []
    for sign in (-1, 1):
        flipped = state.copy()
        flipped[bit] = sign
        ends.append(make_endpoint(task, context, pool, flipped, p, cell_indices, carrier_seed))
    return p, state, ends


def flip_pairs(task, root, bit, n_pairs, contexts, group="train"):
    """Recovery panel: pair ``i`` uses context ``contexts[i % len(contexts)]`` of pool group ``group``."""
    pairs = []
    for index in range(n_pairs):
        context = contexts[index % len(contexts)]
        _, _, (minus, plus) = draw(task, context, task.pool(group, context), seeded(root, bit, index), bit)
        pairs.append(Pair(minus, plus, bit))
    return pairs


def profile_bags(task, contexts, group, root, per_source, bit=None):
    """Profile bags ``(p, state, endpoints)``; one endpoint, or a flip pair when ``bit`` is given."""
    bags = []
    for j, context in enumerate(contexts):
        pool = task.pool(group, context)
        for i in range(per_source[j]):
            rng = seeded(root, j, i, 0 if bit is None else bit + 1)
            bags.append(draw(task, context, pool, rng, bit))
    return bags


def native_indices(update, bags):
    """The native batches that make update ``update`` (1-based) of ``bags`` bags per update.

    With k = bags / 8 native batches per update, update u reads native indices k(u - 1) + 1 to ku.
    """
    if update < 1 or bags % BAGS_PER_NATIVE_BATCH:
        raise ValueError("Updates start at one and use whole native batches of eight bags")
    count = bags // BAGS_PER_NATIVE_BATCH
    return tuple(range(count * (update - 1) + 1, count * update + 1))


def training_batch(stream, update, stream_root, bags, level=LEVEL):
    """Training bags for one update: its consecutive native batches of the stream, concatenated."""
    pieces = [stream.batch(index, N_CELLS, level, stream_root) for index in native_indices(update, bags)]
    return pieces[0] if len(pieces) == 1 else tuple(np.concatenate(fields, axis=0) for fields in zip(*pieces))
