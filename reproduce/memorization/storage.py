"""Storage of the training lines' responses and control levels: how far a model's context-level correction reproduces
each context's offset, on the lines it was trained on and on held-out lines, and how far it restores the context's
control-pool level, with training length.

Terms as in ``data`` and ``readers``. A context's **offset** (the paper's source context response) is the weighted mean
over its units of the truth minus the context's control-pool mean ``C`` minus the training lines' drug average ``r``;
its **sampling difference** ``s`` is the weighted mean of ``C - b``, the pool mean minus the input-set mean. The
readers without the gut-lineage feature (gene-wise readers on every gene, the growth reader on the three growth
programs for the growth genes) split the offset into its **readable part** ``R``, the readers' offset read from the
context's input set, and the **unreadable residual** ``E`` = offset - ``R`` (the paper's readable and unreadable
response). On the training lines, each line's ``R`` comes from readers fitted on the other lines; on held-out lines,
from readers fitted on all training lines.

A model's **correction** of a context is the weighted mean over its units of the prediction minus the anchor ``b + r``.
The correction, ``R``, ``E`` and ``s`` are centred per plate over the side's contexts (``centre_contexts``), and the
correction is regressed on the other three without intercept (``slopes``). The slope on ``E`` measures storage: a model
that has memorized the training lines' offsets reproduces their unreadable part there (a slope near one), a part it
cannot know on held-out lines. The slope on ``s`` measures control-level memory: one restores the context's whole
control pool from the cells it is given, zero keeps the sampled level.

* ``storage_table``: the slope on ``E``, pooled over contexts and genes, for the growth genes (``programs``) and the
  other genes separately; the summary-statistics, DeepSets and attention models' Mean-level fits on the ladder, folds
  1-3.
* ``training_length_table``: the slope on ``s`` on fold 1's training lines, at program level (the growth genes mapped
  to one value per context with the growth pairs' mix of programs, ``selection.mix``), of the same runs read
  after 40,000, 80,000 and 120,000 updates (``checkpoint``; ``LENGTH_RUNS``), each seed and their mean.
"""
from functools import cache

import numpy as np
import pandas as pd

from scsailor.resolution import grids

from ..origin_resolution import SEEDS
from . import data, programs, readers, selection

__all__ = ["BUILT", "STEPS", "LENGTH_RUNS", "centre_contexts", "correction", "slopes", "program_slopes",
           "readable_training", "context_levels", "sides", "checkpoint", "storage_table", "training_length_table"]

BUILT = ('summary', 'deepsets', 'attention_model')
STEPS = (40000, 80000, 120000)   # updates; the ladder's fits are the runs at 40,000
# the fold-1 runs whose predictions were also saved after 80,000 and 120,000 updates
LENGTH_RUNS = {'summary': (31002, 31003, 31004), 'deepsets': (31002, 31004), 'attention_model': (31003, 31004)}


def centre_contexts(M, contexts):
    """The rows of ``M`` (one per context key, 'line|plate') minus their plate's mean (``readers.centre``)."""
    return readers.centre(M, data.plates(contexts))


def correction(prediction, anchor, units):
    """A model's context-level correction: per context (sorted), the weighted mean of prediction minus anchor over its
    units, centred per plate."""
    contexts, M = data.context_means(prediction - anchor, units['context'], units['weight'])
    return centre_contexts(M, contexts)


def slopes(c, R, E, s, mask):
    """The slopes on ``R``, ``E`` and ``s`` of the regression of the correction ``c`` on them (contexts x genes,
    centred), without intercept, pooled over contexts and the genes of ``mask``."""
    X = np.stack([R[:, mask].ravel(), E[:, mask].ravel(), s[:, mask].ravel()], axis=1)
    return np.linalg.lstsq(X, c[:, mask].ravel(), rcond=None)[0]


def program_slopes(c, R, E, s, G, weights):
    """The same regression across contexts at program level: each term's growth genes ``G`` mapped to one value per
    context by ``weights``. Returns the slopes on R, E and s."""
    X = np.stack([M[:, G] @ weights for M in (R, E, s)], axis=1)
    return np.linalg.lstsq(X, c[:, G] @ weights, rcond=None)[0]


def readable_training(Y, contexts, means, K, kernel_contexts, G):
    """The readable part of each training context's offset: the readers' offset of the context, the readers fitted on
    the other lines' contexts (``readers.readers_offsets``; ``K`` the growth reader's kernel, whose rows are
    ``kernel_contexts``)."""
    line = data.lines(contexts)
    R = np.zeros_like(Y)
    for held in np.unique(line):
        out = line == held
        R[out] = readers.readers_offsets(Y=Y[~out], contexts=contexts[~out], means=means[~out], K=K,
                                         kernel_contexts=kernel_contexts, G=G, test=contexts[out],
                                         test_means=means[out])
    return R


def context_levels(units, C, b, r):
    """The contexts (sorted) of a side's units and their offsets (truth - C - r) and sampling differences (C - b).
    ``sides`` passes ``data.fold_data``'s ``r``, the drug average as the models received it: the same average as the
    readers' ``r_double``, stored in single precision."""
    contexts, offset = data.context_means(units['truth'] - C - r[units['p_code']], units['context'], units['weight'])
    _, sampling = data.context_means(C - b, units['context'], units['weight'])
    return contexts, offset, sampling


@cache
def sides(fold):
    """Per side of the fold ('training lines', 'held-out lines'): its units 'u', their anchors 'A0', the side of the
    saved predictions 'grid', and per context (sorted) the readable part 'R', the unreadable residual 'E' and the
    sampling difference 's', each centred per plate."""
    d = data.fold_data(fold)
    tr, bags = d['train'], d['bags']
    _, G = programs.growth(fold)
    kernel_contexts, X, _ = readers.features(fold)
    K = X @ X.T
    trained, offset_trained, sampling_trained = context_levels(tr['u'], tr['C'], tr['b'], d['r'])
    held, offset_held, sampling_held = context_levels(d['u'], d['C'], d['b'], d['r'])
    means = data.input_means(bags['train'], trained)
    readable_trained = readable_training(offset_trained, trained, means, K, kernel_contexts, G)
    readable_held = readers.readers_offsets(Y=offset_trained, contexts=trained, means=means, K=K,
                                            kernel_contexts=kernel_contexts, G=G, test=held,
                                            test_means=data.input_means(bags['heldout'], held))

    def side(contexts, offset, readable, sampling, units, A0, grid):
        """One side's entry of ``sides``."""
        return {'R': centre_contexts(readable, contexts), 'E': centre_contexts(offset - readable, contexts),
                's': centre_contexts(sampling, contexts), 'u': units, 'A0': A0, 'grid': grid}

    return {'training lines': side(contexts=trained, offset=offset_trained, readable=readable_trained,
                                   sampling=sampling_trained, units=tr['u'], A0=tr['A0'], grid='train'),
            'held-out lines': side(contexts=held, offset=offset_held, readable=readable_held, sampling=sampling_held,
                                   units=d['u'], A0=d['A0'], grid='outer')}


def checkpoint(model, seed, step):
    """The file of a fold-1 Mean-level run's predictions (training and held-out lines) after ``step`` updates:
    the ladder's own file at 40,000 (``data.ladder_file``)."""
    return data.ladder_file('F1', model, seed, None if step == STEPS[0] else step)


def side_prediction(path, side):
    """A saved prediction on the units of one side (``sides``)."""
    return grids.load(path, side['grid'], side['u'])


def storage_table():
    """One row per fold, model, side and gene set: the storage slope ('slope_unreadable'). Written with
    ``to_csv(**data.WRITE)``."""
    rows = []
    for fold in data.FOLDS:
        _, G = programs.growth(fold)
        for side, x in sides(fold).items():
            for m in BUILT:
                # the regression is linear in the prediction: on the seeds' mean prediction, its slopes are the mean of
                # the seeds' slopes
                P = np.mean([side_prediction(data.ladder_file(fold, m, s), x) for s in SEEDS], axis=0)
                c = correction(P, x['A0'], x['u'])
                for genes, mask in (('growth genes', G), ('other genes', ~G)):
                    rows.append({'fold': fold, 'model': data.MODELS[m], 'side': side, 'genes': genes,
                                 'slope_unreadable': slopes(c, x['R'], x['E'], x['s'], mask)[1]})
    return pd.DataFrame(rows)


def training_length_table():
    """One row per model, length and seed (and the seed mean): the slope on the sampling difference of the program-level
    regression on fold 1's training lines (``program_slopes``), with the model's number of seeds. Written with
    ``to_csv(**data.WRITE)``."""
    x = sides('F1')['training lines']
    G, mix = selection.mix('F1')
    rows = []
    for m, seeds in LENGTH_RUNS.items():
        for step in STEPS:
            info = {'fold': 'F1', 'level': 'Mean', 'model': data.MODELS[m], 'step': step, 'seeds': len(seeds)}
            slope = [program_slopes(correction(side_prediction(checkpoint(m, s, step), x), x['A0'], x['u']),
                                    x['R'], x['E'], x['s'], G, mix)[2] for s in seeds]
            rows += [{**info, 'seed': s, 'train_program_slope_sampling': v} for s, v in zip(seeds, slope)]
            rows.append({**info, 'seed': data.SEED_MEAN, 'train_program_slope_sampling': float(np.mean(slope))})
    return pd.DataFrame(rows)
