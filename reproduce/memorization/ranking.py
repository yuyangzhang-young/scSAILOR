"""Line ranking by growth-program response on Tahoe-100M folds 1-3: how each model ranks a fold's held-out lines by
their growth-program response, the measured plate reference ('Plate against plate'), and the pull toward the training
lines' control-pool growth level.

Terms as in ``data``, ``readers`` and ``selection``. A line's growth-program response on one plate is the mean, over
the fold's pairs read (``selection.pairs``), of the program response: measured, the truth minus the control-pool mean;
predicted, the clipped prediction minus the input-set mean (``Growth``).

**Predictors**, each training seed: the models' predictions on every held-out unit (``arms.originals``: the Mean-level
fits, State's +Var fits), 'original', and the same with the readers (``readers.fold_arms``), 'readers'.

**Line ranking** across a fold's 14 held-out lines (Spearman correlations; ``growth_ranking_table``): each predictor's
prediction from one plate's input cells against the other plate's measurement, both directions averaged (``cross``);
the plate reference, the measured response on one plate against the other (``replicate``), with a 95% interval from
``RESAMPLES`` resamples of the lines (``interval``, seed ``BOOT_SEED``). ``growth_lines_table`` holds the line
responses. The pair-median growth relation (``pair_median``) ranks the lines per pair instead, both directions
averaged, and takes the median over the pairs.

**Pull** (``pull_table``): per held-out context, the gap between the mean control-pool level of the same plate's
training contexts on the growth genes and the context's input-set mean (``level_gap``), and the model's context-level
correction ``proj_ctx(P - A0)`` on the growth genes. Both are read at program level with the pairs' mix of programs
(``selection.mix``) and centred per plate; the pull is the slope ``beta = sum(a c) / sum(c^2)`` of the correction on the
gap (``pull_slope``). Without pull (``without_pull``), ``beta`` times the gene-wise gap is taken out of the
prediction's growth genes, and the line ranking read again (after clipping).

**Seed means** (``line_ranking_table``): per fold and model, the means of the seeds' rankings and pulls, read from
the ranking and pull tables as written (the paper's seed means were taken from them at six significant digits);
``fold_means``: their means over folds, read from the seed-mean table as written.
"""
from functools import cache

import numpy as np
import pandas as pd

from ..origin_resolution import tahoe
from . import arms, data, programs, readers, selection
from .selection import spearman_cols

__all__ = ["PLATES", "RESAMPLES", "BOOT_SEED", "PREDICTORS", "PLATE_REFERENCE", "rho", "replicate", "cross",
           "interval", "line_means", "pair_median", "Growth", "growth_response", "growth_lines_table",
           "growth_ranking_table", "level_gap", "pull_slope", "without_pull", "pull_table", "line_ranking_table",
           "fold_means"]

PLATES = list(tahoe.PLATES)
RESAMPLES, BOOT_SEED = 2000, 0
PREDICTORS = ('original', 'readers')
PLATE_REFERENCE = 'Plate against plate'   # the measured response on one plate against the other


def rho(a, b):
    """The Spearman correlation of two vectors."""
    return float(spearman_cols(np.asarray(a, float)[:, None], np.asarray(b, float)[:, None])[0])


def replicate(Y, i):
    """Spearman correlation of Y's two plates (lines x plates) on the lines ``i``."""
    return rho(Y[i, 0], Y[i, 1])


def cross(X, Y, i):
    """X from one plate against Y from the other, both directions averaged, on the lines ``i``."""
    return (rho(X[i, 0], Y[i, 1]) + rho(X[i, 1], Y[i, 0])) / 2


def interval(f, n):
    """95% interval of the reading ``f`` (of a vector of line indices) over ``RESAMPLES`` resamples of the n lines."""
    draws = np.random.default_rng(BOOT_SEED).integers(0, n, size=(RESAMPLES, n))
    return np.nanpercentile([f(i) for i in draws], [2.5, 97.5])


def line_means(R, pairs, line, plate, code):
    """Lines x plates: the mean over the pairs (drug code, column of R) of R."""
    frame = pd.DataFrame({'line': line, 'plate': plate})
    t = pd.concat([frame[code == q].assign(v=R[code == q, j]) for q, j in pairs]).groupby(['line', 'plate']).v
    return t.mean().unstack('plate')[PLATES]


def pair_median(predicted, measured, pairs):
    """The pair-median growth relation: per pair (drug code, program index), the Spearman correlation across the drug's
    lines of one plate's predicted response with the other plate's measured one, both directions averaged; then the
    median over the pairs. ``predicted``, ``measured``: ``selection.by_plate`` of units x programs responses."""
    def relation(q, j):
        _, predicted_first, predicted_second = predicted[q]   # each: the drug's lines x programs, on one plate
        _, measured_first, measured_second = measured[q]
        return (rho(predicted_first[:, j], measured_second[:, j])
                + rho(predicted_second[:, j], measured_first[:, j])) / 2

    return float(np.median([relation(q, j) for q, j in pairs]))


class Growth:
    """A fold's held-out lines read on its pairs: the measured growth-program response per line and plate, and
    predicted ones."""

    def __init__(self, fold):
        self.d = d = data.fold_data(fold)
        masks, _ = programs.growth(fold)
        self.W = selection.weights(masks)
        self.pairs = selection.pairs(fold)
        self.G, self.mix = selection.mix(fold)
        measured_units = (d['u']['truth'] - d['C']) @ self.W
        self.measured_pairs = selection.by_plate(measured_units, d['line'], d['plate'], d['code'])
        measured = self.means(measured_units)
        self.lines = measured.index.to_numpy()
        self.measured = measured.to_numpy()
        self.every_line = np.arange(len(self.lines))

    def means(self, R):
        """Lines x plates: ``line_means`` of a units x programs response over the fold's pairs."""
        return line_means(R, self.pairs, self.d['line'], self.d['plate'], self.d['code'])

    def responses(self, P, rows=None):
        """Units x programs: the predicted program response (clip(P) - b) @ W; P holds the units of ``rows`` (all if
        None), and the other units are NaN."""
        b = self.d['b']
        R = np.full((len(b), self.W.shape[1]), np.nan)
        idx = np.arange(len(b)) if rows is None else np.flatnonzero(rows)
        R[idx] = (np.maximum(P, 0) - b[idx]) @ self.W
        return R

    def predicted(self, R):
        """Lines x plates from a units x programs response."""
        return self.means(R).loc[self.lines, PLATES].to_numpy()

    def ranking(self, P):
        """The line ranking of a prediction on every held-out unit."""
        return cross(self.predicted(self.responses(P)), self.measured, self.every_line)

    def pairs_median(self, R):
        """The pair-median growth relation (``pair_median``) of a units x programs response."""
        d = self.d
        return pair_median(selection.by_plate(R, d['line'], d['plate'], d['code']), self.measured_pairs, self.pairs)


@cache
def growth_response(fold):
    """The fold's ``Growth``."""
    return Growth(fold)


def training_levels(fold):
    """The fold's training contexts (sorted) and their control-pool means on the growth genes."""
    _, G = programs.growth(fold)
    bags = data.fold_data(fold)['bags']['train']
    contexts = sorted(bags)
    # in C order, as the matrix product of ``level_gap`` takes its sums in the order of the paper's tables (a column
    # selection alone would give Fortran order)
    return contexts, np.ascontiguousarray(data.pool_means(bags, contexts)[:, G])


def level_gap(level, train_contexts, own, held_contexts):
    """Per held-out context (rows of ``own``): the mean of the training contexts' ``level`` on its plate, minus
    ``own``."""
    same_plate = (data.plates(held_contexts)[:, None] == data.plates(train_contexts)[None]).astype(float)
    return same_plate / same_plate.sum(axis=1, keepdims=True) @ level - own


def pull_slope(correction, gap, mix, plates):
    """The slope of the correction on the gap (contexts x growth genes), both at program level (``mix``) and centred per
    plate: sum(a c) / sum(c^2)."""
    a, c = (readers.centre((X @ mix)[:, None], plates)[:, 0] for X in (correction, gap))
    return float(np.sum(a * c) / np.sum(c * c))


def without_pull(P, slope, gap, G, context_index):
    """``P`` with ``slope`` times each unit's context gap (``gap``: contexts x growth genes; ``context_index``: each
    unit's row of ``gap``) taken out of its growth genes ``G``."""
    Q = P.copy()
    Q[:, G] -= slope * gap[context_index]
    return Q


class Pull:
    """A fold's pull: the gap of each held-out context's growth level (``level_gap``), and a prediction's pull slope
    and the prediction without it."""

    def __init__(self, fold, G, mix):
        d = data.fold_data(fold)
        self.D, self.G, self.mix, self.A0 = data.design(fold), G, mix, d['A0']
        held, self.first, self.unit_ctx = np.unique(d['u']['context'], return_index=True, return_inverse=True)
        self.plates = data.plates(held)
        train, level = training_levels(fold)
        self.gap = level_gap(level, train, self.ctx_mean(d['b'][:, G]), held)

    def ctx_mean(self, X):
        """Each held-out context's weighted mean of X over its units."""
        return self.D.project_ctx(X)[self.first]

    def __call__(self, P):
        """The pull slope of the prediction ``P`` and ``P`` without the pull (``without_pull``)."""
        beta = pull_slope(self.ctx_mean((P - self.A0)[:, self.G]), self.gap, self.mix, self.plates)
        return beta, without_pull(P, beta, self.gap, self.G, self.unit_ctx)


@cache
def fold_growth(fold):
    """The fold's rows of the growth lines, growth ranking and pull tables."""
    g = growth_response(fold)
    pull = Pull(fold, g.G, g.mix)
    lines, M = g.lines, g.measured
    n = len(lines)

    def frame_of(series, X, model='', level='', seed=''):
        return pd.DataFrame({'fold': fold, 'line': np.repeat(lines, 2), 'plate': PLATES * n, 'series': series,
                             'model': model, 'level': level, 'seed': seed, 'value': X.reshape(-1)})

    base = {'fold': fold, 'lines': n, 'pairs': len(g.pairs)}
    low, high = interval(lambda i: replicate(M, i), n)
    long = [frame_of('measured', M)]
    ranking = [{**base, 'model': PLATE_REFERENCE, 'level': '', 'predictor': '', 'seed': '',
                'lines_spearman': replicate(M, g.every_line), 'low': low, 'high': high}]
    pulls = []
    for m, level, seeds in arms.originals(fold):
        for s, P in seeds.items():
            for predictor, Q in zip(PREDICTORS, (P, readers.fold_arms(fold, P)['readers'])):
                X = g.predicted(g.responses(Q))
                long.append(frame_of(predictor, X, data.MODELS[m], level, s))
                ranking.append({**base, 'model': data.MODELS[m], 'level': level, 'predictor': predictor, 'seed': s,
                                'lines_spearman': cross(X, M, g.every_line), 'low': np.nan, 'high': np.nan})
            beta, without = pull(P)
            pulls.append({'fold': fold, 'model': data.MODELS[m], 'level': level, 'seed': s, 'pull_slope': beta,
                          'ranking_without_pull': g.ranking(without)})
    return pd.concat(long, ignore_index=True), ranking, pulls


def growth_lines_table():
    """One row per fold, held-out line, plate and series: the measured growth-program response ('measured'), and each
    predictor's (series 'original' or 'readers', with its model, level and seed). Written with
    ``to_csv(**data.WRITE)``."""
    return pd.concat([fold_growth(fold)[0] for fold in data.FOLDS], ignore_index=True)


def growth_ranking_table():
    """One row per fold for the plate reference (with its 95% interval, 'low' and 'high') and per fold, model, level,
    predictor and seed: the line ranking ('lines_spearman'), with the numbers of lines and pairs read. Written with
    ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for fold in data.FOLDS for r in fold_growth(fold)[1]])


def pull_table():
    """One row per fold, model (the Mean fits; State's +Var fits) and seed: the pull ('pull_slope') and the line ranking
    of the prediction without it. Written with ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for fold in data.FOLDS for r in fold_growth(fold)[2]])


def line_ranking_table(ranking, pull):
    """Per fold: the plate reference with its 95% interval ('data', 'data_low', 'data_high'); per fold, model and level:
    the seed means of the pull ('pull_slope') and of the line rankings of the original, of the original without pull
    and of the readers. From the growth ranking (``growth_ranking_table``) and ``pull`` tables as written. Written with
    ``to_csv(**data.WRITE)``."""
    r = data.as_written(ranking).fillna({'level': ''})
    keys = ['fold', 'model', 'level']
    plate = r[r.model == PLATE_REFERENCE].rename(columns={'lines_spearman': 'data', 'low': 'data_low',
                                                          'high': 'data_high'})
    seed_mean = r[r.model != PLATE_REFERENCE].groupby(keys + ['predictor'], sort=False).lines_spearman.mean()
    models = seed_mean.unstack('predictor')[list(PREDICTORS)]
    pulled = data.as_written(pull).groupby(keys, sort=False)[['pull_slope', 'ranking_without_pull']].mean()
    models = pulled.join(models).rename(columns={'ranking_without_pull': 'without_pull'}).reset_index()
    out = pd.concat([plate[keys + ['data', 'data_low', 'data_high']], models], ignore_index=True)
    return out.sort_values('fold', kind='stable')[keys + ['pull_slope', 'original', 'without_pull', 'readers', 'data',
                                                          'data_low', 'data_high']]


def fold_means(line_ranking):
    """Per model of the seed-mean table (``line_ranking_table``, as written): the mean over folds of its rankings (of
    the plate reference: of 'data'). Written with ``to_csv(**data.WRITE)``."""
    t = data.as_written(line_ranking).fillna({'level': ''})
    out = []
    for (model, level), g in t.groupby(['model', 'level'], sort=False):
        columns = ['data'] if model == PLATE_REFERENCE else ['original', 'without_pull', 'readers']
        out.append({'model': model, 'level': level, 'folds': len(g),
                    **{c: float(np.mean(g[c].to_numpy())) for c in columns}})
    return pd.DataFrame(out)
