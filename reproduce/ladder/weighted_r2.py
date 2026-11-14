"""Adapted weighted R2 (Supplementary Note S3) across the information ladder: Supplementary Fig. 1 and Supplementary
Data 2, from the saved scores under ``natural-scores/weighted-r2/``.

The saved scores hold each fit's adapted weighted R2, total and contrast versions (``part-*.csv``: one row per fit and
the baseline and no-change rows of every fold, with the prediction units a singleton context leaves out), and Parse
PBMC's on all twelve donors (``parse_pbmc_donors.csv``, rows with ``support`` 'twelve', each with its fold baseline's).
Parse PBMC's no-change prediction on twelve donors is scored here (``adapted_r2``), from the deposited held-out truth
and input sets. A fold point is the mean over the kept fits (seeds) of a model, level and fold of their scores, minus
the fold baseline's; a fold mean is the mean over folds.

The files name the attention model ``state`` and State ``state_pub``.
"""
import csv
from functools import cache

import numpy as np
import pandas as pd

from scsailor.natural.score import average_valid
from scsailor.resolution import metrics
from scsailor.resolution.stages import bag_of

from ..origin_resolution import bag_means, heldout_units, parse
from .fits import CODE, LEVELS, SCORES

__all__ = ["adapted_r2", "fit_scores", "reference_scores", "plotted_table", "references_table"]

FOLDER = SCORES / 'weighted-r2'
DATASETS = {'tahoe': 'Tahoe-100M', 'gse': 'GSE306429', 'parse': 'Parse PBMC', 'replogle': 'Replogle–Nadig',
            'jiang': 'Jiang et al.'}
LEVEL_KEY = {label: key for key, label in LEVELS.items()}
VERSIONS = {'Total': 'weighted_r2', 'Contrast': 'weighted_r2_contrast'}


def adapted_r2(prediction, u):
    """{'weighted_r2', 'weighted_r2_contrast'}: the adapted weighted R2 of a prediction, clipped at zero, on prediction
    units ``u``, each the weighted mean over the units with a defined score (a unit alone in its context has none). No
    target gene is left out: this serves Tahoe-100M and Parse PBMC, where no gene is targeted."""
    x = np.maximum(prediction, 0)
    ybar = metrics.context_mean(u['truth'], u['weight'], u['context'])
    d = u['truth'] - ybar
    gene_weight = metrics.effect_weights(d)
    changes = {'weighted_r2': x - ybar, 'weighted_r2_contrast': x - metrics.context_mean(x, u['weight'], u['context'])}
    return {name: average_valid(metrics.r2_units(d, change, gene_weight), u['weight'])['value']
            for name, change in changes.items()}


def saved_rows():
    """The rows of the saved scores of every screen but Parse PBMC, whose twelve-donor scores are read from
    ``parse_pbmc_donors.csv``; and the twelve-donor rows of that file."""
    rows = []
    for part in sorted(FOLDER.glob('part-*.csv')):
        with open(part) as handle:
            rows += [r for r in csv.DictReader(handle) if DATASETS[r['dataset']] != 'Parse PBMC']
    with open(FOLDER / 'parse_pbmc_donors.csv') as handle:
        return rows, [r for r in csv.DictReader(handle) if r['support'] == 'twelve']


@cache
def fit_scores():
    """(screen, fold, model, seed, level) -> {'weighted_r2', 'weighted_r2_contrast'} of every fit."""
    rows, parse_rows = saved_rows()
    out = {(DATASETS[r['dataset']], r['fold'], CODE[r['model']], r['seed'], r['level']):
           {'weighted_r2': float(r['r2w_standard']), 'weighted_r2_contrast': float(r['r2w_contrast'])}
           for r in rows if r['kind'] == 'fit'}
    return out | {('Parse PBMC', r['group'], CODE[r['model']], r['seed'], LEVEL_KEY[r['level']]):
                  {'weighted_r2': float(r['total']), 'weighted_r2_contrast': float(r['contrast'])} for r in parse_rows}


@cache
def reference_scores():
    """(screen, fold, reference) -> {'weighted_r2', 'weighted_r2_contrast', 'units_left_out'} of the baseline and of no
    change, Parse PBMC on all twelve donors."""
    rows, parse_rows = saved_rows()
    out = {(DATASETS[r['dataset']], r['fold'], r['kind']):
           {'weighted_r2': float(r['r2w_standard']), 'weighted_r2_contrast': float(r['r2w_contrast']),
            'units_left_out': int(r['units_dropped'])} for r in rows if r['kind'] != 'fit'}
    out |= {('Parse PBMC', r['group'], 'baseline'):
            {'weighted_r2': float(r['baseline_total']), 'weighted_r2_contrast': float(r['baseline_contrast']),
             'units_left_out': int(r['undefined_units'])} for r in parse_rows}
    for group in parse.FOLDS:   # no change: each held-out context's input-set mean
        f = parse.files(group)
        u = heldout_units(f['outer'])
        # Parse PBMC has no singleton context, so no prediction unit is left out (Supplementary Note S3)
        out[('Parse PBMC', group, 'no_change')] = adapted_r2(bag_of(bag_means(f['inputs'])['heldout'], u)[0], u) | \
            {'units_left_out': 0}
    return out


def plotted_table(fit_table):
    """Supplementary Fig. 1: per screen, model, version, fold and level, the mean over the kept fits (seeds) of their
    score minus the fold baseline's (fold points), and per level the mean over folds (``fold mean``)."""
    refs = reference_scores()
    kept = fit_table[fit_table['fit_status'] == 'kept']
    rows = []
    for (screen, model), fits in kept.groupby(['screen', 'model'], sort=False):
        for version, column in VERSIONS.items():
            folds = {}
            for (fold, level), seeds in fits.groupby(['fold', 'level'], sort=False):
                baseline = refs[(screen, fold, 'baseline')][column]
                value = float(np.mean(seeds[column]))
                folds.setdefault(level, []).append(value - baseline)
                rows.append(dict(screen=screen, model=model, version=version, fold=fold, level=level,
                                 seeds=';'.join(sorted(seeds['seed'])), weighted_r2=value,
                                 baseline_weighted_r2=baseline, difference=value - baseline))
            rows += [dict(screen=screen, model=model, version=version, fold='fold mean', level=level,
                          difference=float(np.mean(diffs))) for level, diffs in folds.items()]
    return pd.DataFrame(rows)


def references_table():
    """Supplementary Data 2's references: the baseline's and no change's scores of every fold, with the prediction units
    left out of every score of the fold."""
    rows = [dict(screen=screen, fold=fold, reference=kind.replace('_', ' '), **v)
            for (screen, fold, kind), v in reference_scores().items()]
    rank = {name: i for i, name in enumerate(DATASETS.values())}
    order = lambda s: s.map(rank) if s.name == 'screen' else s   # noqa: E731
    table = pd.DataFrame(rows).sort_values(['screen', 'fold', 'reference'], key=order, kind='stable')
    return table.reset_index(drop=True)
