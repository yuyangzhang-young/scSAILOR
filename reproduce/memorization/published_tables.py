"""The tables of Results 7 read from the published models' retraining runs on Tahoe-100M (``published_retraining``
trains them): the target interventions, the two-phase retraining of State and the retrieval of planted training
responses. Terms as in ``data``, ``readers``, ``storage`` and ``ranking``.

**Fits** (``fit_file``; model codes ``state``, ``cpa``, ``prepr``; levels ``L1`` Mean, ``L2`` +Var): 'original
fits', the models' fits on the ladder (``data.ladder_file``; State's at +Var, its fold-1 fits of the probes,
``probes.STATE_FIT``); 'same-machine refits', the original recipe trained again (``fits.tahoe`` design 'original');
'input-relative', 'readable-only' and 'two-phase', the retraining runs of that design (``published_retraining``).
Tasks: ``natural`` and the planted null ``c`` (``planted.task``).

**Target interventions** (``interventions_table``; fold 1): State at +Var (original fits, input-relative,
readable-only) and at Mean (original fits, same-machine refits); CPA and PrePR-CT at Mean (original fits, same-machine
refits, input-relative, readable-only). Per run: on the training lines, the storage slope (``storage.slopes``, growth
genes) and the control-level memory (``storage.program_slopes``, the slope on the sampling difference at program
level); on the held-out lines, the pull (``ranking.Pull``), the line ranking (``ranking.Growth``) and the total error in
% of the design-aware baseline's. Each seed is read alone, then the seeds are averaged.

**Two-phase retraining of State** (``two_phase_table``; +Var, folds 1-3): the two-phase run and the original with the
readers in place of the context-level correction (``readers.fold_arms``), each seed's total error in % of the
design-aware baseline's, and their difference with a 95% line interval (``scsailor.resolution.comparison.bootstrap``;
for the seed mean, the seeds' per-line differences averaged within lines, ``pool_seeds``). The original compared with
(``compared_with``) is the same-machine refit or the original fit (origin resolution's +Var case,
``calibration.plus_var``). On the calibration-supported units, the calibrated original (``calibration.case``) and
the two-phase run with the readers, in % of that unit set's design-aware baseline. ``two_phase_metrics_table``: the
two-phase run with the readers minus the original with the readers on the nine metrics, all held-out units, with 90%
line intervals (``metrics.draws``).

**Retrieval** (``retrieval_table``; fold 1, Mean, same-machine refits): the reading of ``retrieval`` applied to State
and PrePR-CT: per held-out context, the change that training on the planted null (task ``c``) makes to the prediction
(``retrieval.planted_change``), its plate-centred correlation with the cue of the nearest training contexts on the same
plate (``retrieval.similarities``), pooled over each model's three seeds, and the permutation probability over 2,000
reassignments of the cues among the training lines (``retrieval.pooled``), drawn from the permutation generator
(``retrieval.generator``).
"""
from functools import cache

import numpy as np
import pandas as pd

from scsailor.resolution import grids
from scsailor.resolution import metrics as nine

from ..origin_resolution import SEEDS, tahoe
from ..origin_resolution.other_metrics import mean_response, references
from ..origin_resolution.probes import STATE_FIT
from . import (arms, calibration, data, fits, metrics, programs, ranking, readers, retrieval, selection, storage,
               two_phase)
from .data import SEED_MEAN

__all__ = ["LEVELS", "INTERVENTIONS", "COMPARISONS", "TWO_PHASE", "fit_file", "training_growth",
           "intervention_readings", "interventions_table", "two_phase_folds", "two_phase_table",
           "two_phase_metrics_table", "RETRIEVAL_SEEDS", "READINGS", "POOLED_BEFORE", "retrieval_table"]

LEVELS = {'L1': 'Mean', 'L2': '+Var'}
# (model, level, fit, seeds) of the target-intervention table, in its row order
INTERVENTIONS = (('state', 'L2', 'original fits', SEEDS), ('state', 'L1', 'original fits', SEEDS),
                 ('state', 'L1', 'same-machine refits', SEEDS), ('state', 'L2', 'input-relative', SEEDS),
                 ('state', 'L2', 'readable-only', SEEDS),
                 ('cpa', 'L1', 'original fits', SEEDS), ('cpa', 'L1', 'same-machine refits', (31002,)),
                 ('cpa', 'L1', 'input-relative', SEEDS), ('cpa', 'L1', 'readable-only', SEEDS),
                 ('prepr', 'L1', 'original fits', SEEDS), ('prepr', 'L1', 'same-machine refits', SEEDS),
                 ('prepr', 'L1', 'input-relative', SEEDS), ('prepr', 'L1', 'readable-only', SEEDS))
COMPARISONS = ('same-machine refits', 'original fits')   # the originals of the two-phase retraining
TWO_PHASE = 'two-phase + readers - original + readers'


def fit_file(fit, model, seed, level, fold='F1', task='natural'):
    """The file of one fit (``fit``: 'original fits', 'same-machine refits' or a retraining design)."""
    if fit == 'original fits':
        if model == 'state' and level == 'L2':
            return STATE_FIT / f'st_anchored_train28_L2_{seed}/PREDICTIONS_step5000.npz'
        return data.ladder_file(fold, model, seed)
    return fits.tahoe('original' if fit == 'same-machine refits' else fit, model, seed, fold, task, level)


def training_growth(path):
    """A fit's prediction on fold 1's training units (``storage.sides``), at the growth genes (``programs.growth``)."""
    _, G = programs.growth('F1')
    return grids.load(path, 'train', storage.sides('F1')['training lines']['u'])[:, G]


@cache
def _fold1():
    """Fold 1's readings shared by every run: the training side of ``storage.sides`` at the growth genes, the
    program-level weights, the pull, the line ranking and the design-aware baseline's total error."""
    d, D = data.fold_data('F1'), data.design('F1')
    G, mix = selection.mix('F1')
    x = storage.sides('F1')['training lines']
    train = {k: x[k][:, G] for k in ('R', 'E', 's', 'A0')}
    base = arms.per_line_total(d['b'] + d['kappa'] * d['r_plate_unit'], D, d['u']['truth']).sum()
    return train, x['u'], mix, ranking.Pull('F1', G, mix), ranking.growth_response('F1'), base


def intervention_readings(held, train):
    """The readings of one run: ``held``, its prediction on fold 1's held-out units; ``train``, on the training units
    at the growth genes."""
    t, u, mix, pull, growth, base = _fold1()
    c = storage.correction(train, t['A0'], u)
    every = np.ones(c.shape[1], bool)
    D, truth = data.design('F1'), data.fold_data('F1')['u']['truth']
    return {'train_growth_slope_unreadable': storage.slopes(c, t['R'], t['E'], t['s'], every)[1],
            'train_program_slope_sampling': storage.program_slopes(c, t['R'], t['E'], t['s'], every, mix)[2],
            'pull_slope': pull(held)[0], 'lines_spearman': growth.ranking(held),
            'total_points': 100 * D.energy(np.maximum(held, 0) - truth) / base}


def interventions_table():
    """One row per model, level, fit and seed (and the seed mean): the readings of ``intervention_readings``; the seed
    mean rows also give the fit's number of seeds. Written with ``to_csv(**data.WRITE)``."""
    u = data.fold_data('F1')['u']
    rows, means = [], []
    for model, level, fit, seeds in INTERVENTIONS:
        info = {'model': data.MODELS[model], 'level': LEVELS[level], 'fit': fit}
        per_seed = []
        for s in seeds:
            path = fit_file(fit, model, s, level)
            per_seed.append(intervention_readings(grids.load(path, 'outer', u), training_growth(path)))
            rows.append({**info, 'seeds': None, 'seed': s, **per_seed[-1]})
        means.append({**info, 'seeds': len(seeds), 'seed': SEED_MEAN,
                      **{k: float(np.mean([r[k] for r in per_seed])) for k in per_seed[0]}})
    return pd.DataFrame(rows + means)


@cache
def two_phase_folds(fold):
    """Per seed, State's two-phase run and its two originals (``COMPARISONS``) on the fold's held-out units, each with
    the readers in place of the context-level correction, and the run's raw prediction."""
    u = data.fold_data(fold)['u']
    original_fits = calibration.plus_var(fold, 'state')['p']
    out = {}
    for s in SEEDS:
        raw = {fit: grids.load(fit_file(fit, 'state', s, 'L2', fold), 'outer', u)
               for fit in ('two-phase', 'same-machine refits')}
        raw['original fits'] = np.asarray(original_fits[s], dtype=np.float64)
        out[s] = {k: (P, readers.fold_arms(fold, P)['readers']) for k, P in raw.items()}
    return out


def two_phase_table():
    """One row per fold, original compared with (``compared_with``: ``COMPARISONS``) and seed (and the seed mean):
    ``two_phase.comparison_rows`` of State's two-phase run against that original, both with the readers. Written with
    ``to_csv(**data.WRITE)``."""
    rows = []
    for fold in data.FOLDS:
        runs, case = two_phase_folds(fold), calibration.case(fold, 'state')
        for comparison in COMPARISONS:
            pairs = {s: (runs[s]['two-phase'][1], runs[s][comparison][1]) for s in SEEDS}
            rows += two_phase.comparison_rows({'fold': fold, 'compared_with': comparison}, pairs, fold, case)
    return pd.DataFrame(rows)


def two_phase_metrics_table():
    """One row per fold, original compared with (``compared_with``: ``COMPARISONS``), metric and seed (and the seed
    mean): the gain of the two-phase run with the readers over the original with the readers on the nine metrics, all
    held-out units. Written with ``to_csv(**data.WRITE)``."""
    rows = []
    for fold in data.FOLDS:
        d, f = data.fold_data(fold), tahoe.files(fold)
        ref = references({'u': d['u']}, {'bags': d['bags']}, 'heldout',
                         mean_response(f['constants'], f['train_targets']))

        def line(x):
            return nine.per_line(nine.unit_values(x, d['u'], ref), d['u'], ref)

        runs = two_phase_folds(fold)
        for comparison in COMPARISONS:
            lines = {s: {'original': line(runs[s][comparison][0]), 'original + readers': line(runs[s][comparison][1]),
                         'two-phase + readers': line(runs[s]['two-phase'][1])} for s in SEEDS}
            info = {'fold': fold, 'model': data.MODELS['state'], 'level': '+Var', 'compared_with': comparison}
            contrast = {TWO_PHASE: ('two-phase + readers', 'original + readers')}
            rows += metrics.case_gains(info, lines, ref['n'], contrast)
    return pd.DataFrame(rows)


RETRIEVAL_SEEDS = {'state': SEEDS, 'prepr': SEEDS}   # the models of the table and each one's seeds
# The order in which the permutation generator serves the readings (``retrieval.generator``): first single-fit readings
# of each run (State and PrePR-CT three seeds each, CPA one) and of the two three-seed averages, under six neighbour
# weightings; then the pooled readings per group (State, CPA, PrePR-CT) and weighting, of which the table prints these.
READINGS = 9 * 6
POOLED_BEFORE = {('state', retrieval.PROFILE): 0, ('state', retrieval.GROWTH): 4,
                 ('prepr', retrieval.PROFILE): 12, ('prepr', retrieval.GROWTH): 16}


def retrieval_table():
    """One row per model (its seeds at Mean, same-machine refits) and printed similarity: the number of fits and the
    pooled retrieval reading (``retrieval.retrieval_table``: 'r_centred', 'p_permutation'). Written with
    ``to_csv(**data.WRITE)``."""
    return retrieval.retrieval_table(RETRIEVAL_SEEDS, lambda m, s: fit_file('same-machine refits', m, s, 'L1'),
                                     READINGS, POOLED_BEFORE)
