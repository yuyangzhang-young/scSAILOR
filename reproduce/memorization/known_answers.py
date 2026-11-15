"""Planted-null and planted-response checks of the two-phase retraining on Tahoe-100M fold 1, seed 31002.

Three planted tasks (``planted``): the planted null (``c``), a random context value that no control statistic predicts,
read on the training lines; the readable shift (``f``), a drug-specific shift that held-out lines can read from their
controls, read on the held-out lines; and the drug x line null (``fnull``), the corresponding drug-by-line value without
a transferable control relation, read on the training lines.

A fit trained on a planted task is read against the fit of the same model, recipe, input level and seed trained on the
measured targets (the natural fit), both raw networks without readers, each clipped at zero
(``retained_fraction``): the change of its prediction over the natural fit's, minus the task's change of the anchor, is
read by the task (``planted.Known.read``), and the retained fraction is its slope on the cue over the planted truth's.

The method is the two-phase recipe (``fits.tahoe('two-phase', ...)``): the summary-statistics, DeepSets and attention
models at Mean, State at +Var. The original recipe (``fits.tahoe('original', ...)``): the built models at Mean, paired
with the original recipe's natural fit (``interventions.original``); State at +Var for the readable shift and the drug
x line null, and for the planted null, for which no +Var fit exists, the mean of the three seeds' Mean fits. 'kept' is
the method's retained fraction over the original's.
"""
import numpy as np
import pandas as pd

from scsailor.resolution import grids

from ..origin_resolution import SEEDS
from . import data, fits, interventions, planted

__all__ = ["TASKS", "SIDES", "SEED", "retained_fraction", "read", "row", "original_fits",
           "known_answers_table"]

TASKS = {'c': ('planted null', 'train'), 'f': ('readable shift', 'heldout'), 'fnull': ('drug x line null', 'train')}
SIDES = {'train': ('train', 'training lines'), 'heldout': ('outer', 'held-out lines')}   # grids side, the table's name
SEED = 31002


def retained_fraction(known, task_prediction, natural_prediction):
    """The retained fraction of a planted task (``known``: ``planted.Known``) in a fit's prediction over the natural
    fit's, both clipped at zero, beyond the task's anchor change."""
    change = np.maximum(task_prediction, 0) - np.maximum(natural_prediction, 0) - known.change
    out = known.read(change)
    return out['slope_over_beta'] / out['truth_slope_over_beta']


def read(known, side, task_file, natural_file):
    """``retained_fraction`` of the fits saved in ``task_file`` and ``natural_file`` on the side's units."""
    load = lambda path: grids.load(path, SIDES[side][0], known.units)   # noqa: E731
    return retained_fraction(known, load(task_file), load(natural_file))


def row(info, method, originals):
    """A row of the table: the method's retained fraction, the original's (the mean over its fits ``originals``) and
    'kept', the first over the second."""
    original = float(np.mean(originals))
    return {**info, 'method': method, 'original': original, 'kept': method / original}


def original_fits(model, task):
    """The original recipe's (planted fit, natural fit) pairs for a model and task, and how the table describes them."""
    if model != 'state':
        pair = (fits.tahoe('original', model, SEED, task=task), interventions.original(model, SEED))
        return [pair], f'Mean, seed {SEED}'
    if task == 'c':
        return ([(fits.tahoe('original', 'state', s, task=task), fits.tahoe('original', 'state', s)) for s in SEEDS],
                f'Mean, mean of seeds {SEEDS[0]}-{SEEDS[-1]}')
    return ([(fits.tahoe('original', 'state', SEED, task=task, level='L2'),
              fits.tahoe('original', 'state', SEED, level='L2'))], f'+Var, seed {SEED}')


def known_answers_table():
    """One row per model (the built models at Mean, State at +Var) and planted task: the side read, the two-phase
    fit's retained fraction ('method'), the original recipe's ('original'), the original fits read ('original_fits')
    and 'kept'. Written with ``to_csv(**data.WRITE)``."""
    rows = []
    for model in ('summary', 'deepsets', 'attention_model', 'state'):
        level = 'L2' if model == 'state' else 'L1'
        for task, (name, side) in TASKS.items():
            known = planted.Known(task, side)
            method = read(known, side, fits.tahoe('two-phase', model, SEED, task=task, level=level),
                          fits.tahoe('two-phase', model, SEED, level=level))
            pairs, described = original_fits(model, task)
            info = {'model': data.MODELS[model], 'level': '+Var' if model == 'state' else 'Mean', 'task': name,
                    'side': SIDES[side][1], 'original_fits': described}
            rows.append(row(info, method, [read(known, side, *pair) for pair in pairs]))
    return pd.DataFrame(rows)[['model', 'level', 'task', 'side', 'method', 'original', 'original_fits', 'kept']]
