"""Retrieval on Tahoe-100M fold 1: does a model pass the values it stored for its training lines to new lines whose
control cells resemble them?

The planted null (task c, ``planted``) adds ``beta`` times a cue to the planted growth genes of every training unit;
each training line carries another line's cue (both plates together), a value its own control cells do not predict.
The inputs are unchanged, so the difference between a model's held-out prediction after training on task c and after
the original recipe's natural training (same model and seed, ``interventions.original``) is the effect of those
training values on new lines alone (``planted_change``): per held-out context, the weighted mean over its units and
the planted genes of clip(P_c) - clip(P_natural) - r_change, divided by ``beta``, where ``r_change`` is the task's
change of the drug average.

A model that stores each training context's offset and retrieves it for inputs that resemble it passes on its
neighbours' cues: the neighbours' cue of a held-out context is the mean cue of its nearest training contexts on the same
plate (``nearest``), by the Euclidean distance of the input-set means over the panel genes (``'untreated profile, 1
nearest'``) or of the growth reader's three program means (``readers.features``; ``'growth level (three program
means), 3 nearest'``).

**Pooled reading** (``pooled``): over a model's three seeds, the mean of the plate-centred correlation across held-out
contexts of the change with the neighbours' cue (``centred_r``); its permutation probability compares it with the same
statistic after reassigning the training lines' cues among the lines (both plates of a line together), ``PERMUTATIONS``
times, the same draws for every seed: (1 + the number at least as large) / (1 + ``PERMUTATIONS``).

**Permutations** (``generator``): one generator seeded ``SEED`` serves every reading in a fixed order: ``READINGS``
readings of single fits (each ``PERMUTATIONS`` line permutations, then ``RESAMPLES`` resamples of the held-out lines),
then pooled readings of ``PERMUTATIONS`` line permutations each, of which the table prints those at the positions
``POOLED_BEFORE``. ``generator`` draws and discards what precedes a printed reading, so that each printed probability
comes from its own position in that order.
"""
import numpy as np
import pandas as pd

from scsailor.resolution import grids

from ..origin_resolution import SEEDS
from ..origin_resolution import tahoe
from . import data, fits, interventions, planted, readers, storage

__all__ = ["FOLD", "SEED", "PERMUTATIONS", "RESAMPLES", "PROFILE", "GROWTH", "MODELS", "READINGS", "POOLED_BEFORE",
           "planted_change", "nearest", "centred_r", "Cues", "pooled", "generator", "similarities", "retrieval_table"]

FOLD = 'F1'
SEED, PERMUTATIONS, RESAMPLES = 20261112, 2000, 2000
PROFILE = 'untreated profile, 1 nearest'
GROWTH = 'growth level (three program means), 3 nearest'
NEAREST = {PROFILE: 1, GROWTH: 3}
MODELS = {m: SEEDS for m in storage.BUILT}   # the table's models and each one's seeds
READINGS = 95
POOLED_BEFORE = {('summary', PROFILE): 6, ('summary', GROWTH): 10, ('deepsets', PROFILE): 12,
                 ('deepsets', GROWTH): 16, ('attention_model', PROFILE): 18, ('attention_model', GROWTH): 22}


def planted_change(task, natural, units, r_change, genes, beta):
    """The held-out contexts (sorted) and each one's change: the weighted mean over its ``units`` of the mean over the
    planted ``genes`` of clip(task) - clip(natural) - r_change (``r_change``: per drug), divided by ``beta``."""
    change = np.maximum(task, 0) - np.maximum(natural, 0) - r_change[units['p_code']]
    contexts, M = data.context_means(change[:, genes].mean(axis=1, keepdims=True) / beta, units['context'],
                                     units['weight'])
    return contexts, M[:, 0]


def nearest(held, train, X_held, X_train, k):
    """Weights (held x training contexts): 1/k on the k training contexts of the same plate nearest to each held-out
    context (Euclidean distance between the rows of ``X_held`` and ``X_train``)."""
    W = np.zeros((len(held), len(train)))
    distance = np.linalg.norm(X_held[:, None, :] - X_train[None, :, :], axis=2)
    same = data.plates(held)[:, None] == data.plates(train)[None]
    for i in range(len(held)):
        candidates = np.flatnonzero(same[i])
        W[i, candidates[np.argsort(distance[i, candidates])[:k]]] = 1 / k
    return W


def centred_r(delta, m, plates):
    """The Pearson correlation of ``delta`` and ``m`` after removing each plate's mean from both."""
    d, n = (readers.centre(np.asarray(v, float), plates) for v in (delta, m))
    return float(np.corrcoef(d, n)[0, 1])


class Cues:
    """The training contexts' cues (``cue``, one per context of ``contexts``), held per training line and plate."""

    def __init__(self, contexts, cue):
        self.lines, self.line = np.unique(data.lines(contexts), return_inverse=True)
        self.plate = np.array([tahoe.PLATES.index(p) for p in data.plates(contexts)])
        self.table = np.full((len(self.lines), len(tahoe.PLATES)), np.nan)
        self.table[self.line, self.plate] = cue

    def of(self, order):
        """Each training context's cue when line i carries the cues of line ``order[i]`` (both plates)."""
        return self.table[order][self.line, self.plate]


def pooled(pairs, cues, plates, rng, permutations=PERMUTATIONS):
    """The pooled reading of several fits (``pairs``: each fit's change per held-out context and its neighbour weights,
    held-out x training contexts): the mean over fits of ``centred_r`` of the change and the neighbours' cue, and its
    permutation probability over ``permutations`` reassignments of the training lines' ``cues`` drawn from ``rng``.
    Returns (r_centred, p_permutation)."""
    def statistic(cue):
        return float(np.mean([centred_r(delta, W @ cue, plates) for delta, W in pairs]))

    observed = statistic(cues.of(np.arange(len(cues.lines))))
    null = np.array([statistic(cues.of(rng.permutation(len(cues.lines)))) for _ in range(permutations)])
    return observed, float((1 + np.sum(null >= observed)) / (1 + permutations))


def generator(readings, pooled_before, lines, held_lines, permutations=PERMUTATIONS, resamples=RESAMPLES):
    """The generator seeded ``SEED`` after the draws that precede a pooled reading: ``readings`` single-fit readings
    (each ``permutations`` permutations of the ``lines`` training lines, then ``resamples`` resamples of the
    ``held_lines`` held-out lines) and ``pooled_before`` pooled readings (``permutations`` line permutations each)."""
    rng = np.random.default_rng(SEED)
    for _ in range(readings):
        for _ in range(permutations):
            rng.permutation(lines)
        for _ in range(resamples):
            rng.choice(held_lines, held_lines)
    for _ in range(pooled_before * permutations):
        rng.permutation(lines)
    return rng


def similarities(held, train):
    """The neighbour weights (held-out x training contexts) of each printed similarity: by input-set mean over the
    panel genes (``PROFILE``) and by the growth reader's three program means (``GROWTH``)."""
    bags = data.fold_data(FOLD)['bags']
    kernel_contexts, X, _ = readers.features(FOLD)
    row = {c: i for i, c in enumerate(kernel_contexts)}
    features = {PROFILE: (data.input_means(bags['heldout'], held), data.input_means(bags['train'], train)),
                GROWTH: (X[[row[c] for c in held]], X[[row[c] for c in train]])}
    return {name: nearest(held, train, *features[name], NEAREST[name]) for name in NEAREST}


def retrieval_table(models=MODELS, natural=interventions.original, readings=READINGS, pooled_before=POOLED_BEFORE):
    """One row per model of ``models`` (model -> seeds) and printed similarity: the pooled reading of the model's seeds
    ('r_centred', 'p_permutation') and the number of fits. Each fit trained on the planted null
    (``fits.tahoe('original', model, seed, task='c')``) is read against the natural fit in ``natural(model, seed)``;
    the generator of each reading skips ``readings`` single-fit readings and ``pooled_before[(model, similarity)]``
    pooled readings (``generator``). Written with ``to_csv(**data.WRITE)``."""
    task = planted.task('c')
    order = np.argsort(task['train_contexts'].astype(str))
    train = task['train_contexts'].astype(str)[order]
    cues = Cues(train, np.asarray(task['cue_train'], float)[order])
    u = data.fold_data(FOLD)['u']

    def change(m, s):
        P = (grids.load(path, 'outer', u) for path in (fits.tahoe('original', m, s, task='c'), natural(m, s)))
        return planted_change(*P, u, task['r_change'], task['genes'], float(task['beta']))

    rows = []
    for m, seeds in models.items():
        changes = [change(m, s) for s in seeds]
        held = changes[0][0]
        plates = data.plates(held)
        for name, W in similarities(held, train).items():
            rng = generator(readings, pooled_before[(m, name)], len(cues.lines), len(np.unique(data.lines(held))))
            r, p = pooled([(delta, W) for _, delta in changes], cues, plates, rng)
            rows.append({'model': data.MODELS[m], 'similarity': name, 'fits': len(changes), 'r_centred': r,
                         'p_permutation': p})
    return pd.DataFrame(rows)
