"""The paper's training-selected drug-program pairs: the drug x growth-program pairs, chosen on a Tahoe-100M fold's 28
training lines, on which its growth-program responses are read.

A program's response in a unit is the mean change of its genes (``weights``; the growth programs of ``programs``):
measured, the truth minus the context's control-pool mean; predicted, the clipped prediction minus the input-set mean.
A context's level of a program is its control-pool mean over the program's genes.

A drug is read on the lines that received it on both plates, when they are at least ``MIN_LINES`` (``by_plate``). Per
drug and growth program, on the training lines (``training_readings``):

* reliability (the paper's between-plate response correlation): the Spearman correlation across lines of the response
  on one plate with the response on the other;
* level relation (the mean within-plate control-level-response correlation): per plate, the Spearman correlation
  across lines of the program's level with its response, the two plates averaged.

A pair is selected when its reliability is at least ``RELIABILITY`` and its level relation at least ``RELATION``, among
the drugs also read on the held-out lines; it is read when calibration covers its drug (the drug occurs on the
calibration lines), so that every predictor, calibrated or not, is read on the same pairs.
"""
from functools import cache

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scsailor.natural.score import row_correlation

from ..origin_resolution import tahoe
from . import data, programs

__all__ = ["MIN_LINES", "RELIABILITY", "RELATION", "weights", "spearman_cols", "by_plate", "training_readings",
           "selected", "fold_selection", "pairs", "program_mix", "mix", "selection_table"]

MIN_LINES = 5
RELIABILITY = .6
RELATION = .4


def weights(masks):
    """Genes x programs: each program's column is 1/n on its n genes, so that X @ weights is each program's mean of
    X over its genes."""
    M = np.stack(masks, axis=1).astype(np.float64)
    return M / M.sum(axis=0)


def spearman_cols(A, B):
    """Column-wise Spearman correlation of two (lines x columns) arrays without missing values; NaN for a constant
    column."""
    return row_correlation(rankdata(A, axis=0).T, rankdata(B, axis=0).T)


def by_plate(values, line, plate, code):
    """Drug code -> (its lines with both plates, sorted; their rows of ``values`` on the first plate; on the second),
    for the drugs with at least ``MIN_LINES`` such lines."""
    out = {}
    index = pd.DataFrame({'line': line, 'plate': plate, 'code': code, 'row': np.arange(len(line))})
    for q, g in index.groupby('code'):
        first = g[g.plate == tahoe.PLATES[0]].set_index('line').row
        second = g[g.plate == tahoe.PLATES[1]].set_index('line').row
        lines = sorted(set(first.index) & set(second.index))
        if len(lines) >= MIN_LINES:
            out[int(q)] = (lines, values[first.loc[lines].values], values[second.loc[lines].values])
    return out


def training_readings(response, level, line, plate, code):
    """Drug code -> (reliability, level relation), each per program (columns of ``response`` and ``level``, units x
    programs)."""
    r = by_plate(response, line, plate, code)
    lv = by_plate(level, line, plate, code)
    out = {}
    for q, (_, a1, a2) in r.items():
        _, l1, l2 = lv[q]
        out[q] = (spearman_cols(a1, a2), (spearman_cols(l1, a1) + spearman_cols(l2, a2)) / 2)
    return out


def selected(readings, drugs):
    """The selected pairs (drug code, program index) among ``drugs``, drugs ascending and programs in order."""
    out = []
    for q in sorted(drugs):
        if q in readings:
            reliability, relation = readings[q]
            out += [(q, j) for j in range(len(reliability))
                    if reliability[j] >= RELIABILITY and relation[j] >= RELATION]
    return out


@cache
def fold_selection(fold):
    """The fold's pairs read (selected and covered by calibration) and its selected pairs as table rows."""
    d = data.fold_data(fold)
    masks, _ = programs.growth(fold)
    W = weights(masks)
    tr = d['train']
    readings = training_readings((tr['u']['truth'] - tr['C']) @ W, tr['C'] @ W, tr['line'], tr['plate'],
                                 tr['u']['p_code'])
    # the drugs read on the held-out lines (by_plate's keys; its values are not needed here)
    held_out_drugs = by_plate(d['code'][:, None], d['line'], d['plate'], d['code']).keys()
    chosen = selected(readings, held_out_drugs)
    keep, _, _ = data.calibration_units(fold)
    covered = set(np.unique(d['code'][keep]).tolist())
    rows = [{'fold': fold, 'p_code': q, 'drug': d['names'][q], 'program': programs.GROWTH[j],
             'covered_by_calibration': q in covered} for q, j in chosen]
    return [(q, j) for q, j in chosen if q in covered], rows


def pairs(fold):
    """The fold's pairs read: (drug code, index of the growth program in ``programs.GROWTH``)."""
    return fold_selection(fold)[0]


def program_mix(W, G, pairs):
    """Per growth gene, its weight at program level: the programs' weights (``W``, genes x programs) on the growth genes
    ``G``, mixed in the proportions in which the pairs read the programs."""
    mix = np.bincount([j for _, j in pairs], minlength=W.shape[1]).astype(float)
    return W[G] @ (mix / mix.sum())


@cache
def mix(fold):
    """The fold's growth genes and their weights at program level with its pairs' mix (``program_mix``)."""
    masks, G = programs.growth(fold)
    return G, program_mix(weights(masks), G, pairs(fold))


def selection_table():
    """One row per fold and selected pair: the drug, the program and whether calibration covers the drug. Written with
    ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for fold in data.FOLDS for r in fold_selection(fold)[1]])
