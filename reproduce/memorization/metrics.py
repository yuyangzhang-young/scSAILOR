"""The arms on the nine metrics, and the summaries of the arms the paper reads, on Tahoe-100M folds 1-3.

**Cases.** Origin resolution's +Var cases (``calibration.case``: the summary-statistics, DeepSets and attention models
and State) on each fold's calibration-supported held-out units (``data.calibration_units``): 4 models x 3 folds = 12
cases. Each training seed's prediction is read in six arms (``calibration.Case.arms``): original, removal, constant,
readers, calibrated and calibrated + readers; beside them, the design-aware baseline.

**Nine metrics** (``scsailor.resolution.metrics``, ``METRICS``), on predictions clipped at zero: each unit's value,
summed by scoring weight per held-out line, then over lines (``per_line``, ``point``); each seed read alone, then the
mean of the seeds ('seed mean'). References: C, the context's whole control-pool mean; R = C + the training drugs'
mean input response (``other_metrics.references``).

**Gains** (``CONTRASTS``): the first prediction minus the second, signed so that positive favours the first; squared
error in % of the original's total. 90% intervals: ``RESAMPLES`` resamples of the held-out lines with replacement
(``draws``, seed ``BOOT_SEED``), the same draws for both predictions, every contrast and every seed; the interval of the
seed mean is that of the mean of the seeds' resampled gains.

**Summaries.** ``summary``: per contrast and metric, how many of the 12 cases' seed-mean gains are positive.
``absolute_values``: per prediction, each metric as the mean over the 12 cases of the seed means, squared error as the
four-arms table's total in % of the design-aware baseline's. ``arm_changes``: per case and arm, the change from the
original in total, common and contrast error, in % of the original's total, the mean of the seeds. Both read their
input tables as written.
"""
import itertools
from functools import cache

import numpy as np
import pandas as pd

from scsailor.resolution import metrics

from ..origin_resolution import SEEDS, tahoe
from ..origin_resolution.other_metrics import mean_response, references
from . import calibration, data, readers
from .data import SEED_MEAN

__all__ = ["METRICS", "CONTRASTS", "BOOT_SEED", "RESAMPLES", "PREDICTIONS", "draws", "case_lines",
           "line_sums", "case_values", "case_gains", "values_table", "gains_table", "summary", "absolute_values",
           "CHANGED_ARMS", "arm_changes"]

METRICS = ('mse', 'weighted_r2', 'weighted_r2_contrast', 'pearson_delta', 'spearman_delta', 'direction_top50',
           'cosine_pds', 'systema_pearson_delta', 'systema_pearson_delta_observed')
CONTRASTS = {'readers - original': ('readers', 'original'), 'readers - calibrated': ('readers', 'calibrated'),
             'calibrated + readers - calibrated': ('calibrated + readers', 'calibrated'),
             'readers - constant': ('readers', 'constant'), 'calibrated - original': ('calibrated', 'original'),
             'removal - original': ('removal', 'original'), 'constant - original': ('constant', 'original'),
             'calibrated + readers - original': ('calibrated + readers', 'original')}
BOOT_SEED, RESAMPLES = 20261102, 10000
PREDICTIONS = ('design-aware baseline', *readers.ARMS, *readers.CALIBRATED_ARMS)
CHANGED_ARMS = ('calibrated', 'removal', 'constant', 'readers', 'calibrated + readers')   # the arms of ``arm_changes``


@cache
def draws(n):
    """The line resamples: ``RESAMPLES`` draws of n held-out lines with replacement, as each line's count per draw."""
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, size=(RESAMPLES, n))
    return np.stack([np.bincount(d, minlength=n) for d in idx]).astype(np.float64)


def case_lines(u, ref, baseline, arms):
    """Per seed, each prediction's per-line sums of the nine metrics (``metrics.per_line``) on the units ``u`` (``ref``:
    ``other_metrics.references`` on them): the design-aware ``baseline``, the same for every seed, and the arms
    ``arms(seed)`` (name -> prediction)."""
    def line(x):
        """The per-line sums of the nine metrics of the prediction ``x``."""
        return metrics.per_line(metrics.unit_values(x, u, ref), u, ref)

    fixed = {'design-aware baseline': line(baseline)}
    return {s: fixed | {name: line(x) for name, x in arms(s).items()} for s in SEEDS}


@cache
def fold_reference(fold):
    """The metrics' references on the fold's calibration-supported units."""
    d, f = data.fold_data(fold), tahoe.files(fold)
    _, u, _ = data.calibration_units(fold)
    return references({'u': u}, {'bags': d['bags']}, 'heldout', mean_response(f['constants'], f['train_targets']))


@cache
def line_sums(fold, model):
    """``case_lines`` of one +Var case."""
    c = calibration.case(fold, model)
    return case_lines(data.calibration_units(fold)[1], fold_reference(fold), c.baseline, c.arms)


def all_cases():
    """The 12 cases: their table keys, per-line sums (``line_sums``) and numbers of held-out lines."""
    for fold, model in itertools.product(data.FOLDS, calibration.PLUS_VAR):
        info = {'fold': fold, 'model': data.MODELS[model], 'level': '+Var', 'units': 'calibration-supported'}
        yield info, line_sums(fold, model), fold_reference(fold)['n']


def case_values(info, lines):
    """One case's rows of the values table: each prediction's nine metrics per seed, then their seed mean."""
    rows = []
    for name in lines[SEEDS[0]]:
        points = {s: {m: metrics.point(lines[s][name][m]) for m in METRICS} for s in SEEDS}
        rows += [{**info, 'seed': s, 'prediction': name, **points[s]} for s in SEEDS]
        rows.append({**info, 'seed': SEED_MEAN, 'prediction': name,
                     **{m: float(np.mean([points[s][m] for s in SEEDS])) for m in METRICS}})
    return rows


def case_gains(info, lines, n):
    """One case's rows of the gains table: every contrast's gain per metric and seed, then their seed mean, with 90%
    intervals over ``n`` held-out lines."""
    K = draws(n)
    rows = []
    for contrast, (first, second) in CONTRASTS.items():
        for m in METRICS:
            seeds, boots = [], []
            for s in SEEDS:
                # ``per_line`` gives (weighted sum, weight) per line; a resample's value is the ratio of the resampled
                # sums, as ``metrics.point`` is of the sums over all lines
                first_lines, second_lines = lines[s][first][m], lines[s][second][m]
                gain = metrics.point(first_lines) - metrics.point(second_lines)
                boot = ((K @ first_lines[0]) / (K @ first_lines[1])
                        - (K @ second_lines[0]) / (K @ second_lines[1]))
                if m == 'mse':
                    scale = 100 / metrics.point(lines[s]['original']['mse'])
                    gain, boot = -scale * gain, -scale * boot
                seeds.append(gain)
                boots.append(boot)
                low, high = np.quantile(boot, [.05, .95])
                rows.append({**info, 'contrast': contrast, 'metric': m, 'seed': s, 'gain': gain, 'ci90_low': low,
                             'ci90_high': high})
            low, high = np.quantile(np.mean(boots, axis=0), [.05, .95])
            rows.append({**info, 'contrast': contrast, 'metric': m, 'seed': SEED_MEAN, 'gain': float(np.mean(seeds)),
                         'ci90_low': low, 'ci90_high': high})
    return rows


def values_table():
    """One row per case, seed (and the seed mean) and prediction (``PREDICTIONS``): the nine metrics. Written with
    ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for info, lines, _ in all_cases() for r in case_values(info, lines)])


def gains_table():
    """One row per case, contrast, metric and seed (and the seed mean): the gain and its 90% interval. Written with
    ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for info, lines, n in all_cases() for r in case_gains(info, lines, n)])


def summary(gains):
    """Per contrast and metric of ``gains`` (``gains_table``): the number of cases (``n``) and how many of their
    seed-mean gains are above 0. Written with ``to_csv(**data.WRITE)``."""
    g = gains[gains.seed == SEED_MEAN]
    return pd.DataFrame([{'contrast': contrast, 'metric': metric, 'n': len(x), 'improved': int((x.gain > 0).sum())}
                         for (contrast, metric), x in g.groupby(['contrast', 'metric'], sort=False)])


def case_mean(table, column):
    """The mean over cases (fold and model, in table order) of each case's mean over its rows."""
    per = {}
    for fold, model, x in zip(table.fold, table.model, table[column]):
        per.setdefault((model, fold), []).append(x)
    return float(np.mean([np.mean(v) for v in per.values()]))


def plus_var_seeds(four_arms):
    """The four-arms table's per-seed rows on the +Var cases, as written."""
    t = data.as_written(four_arms, dtype={'seed': str})
    return t[(t.level == '+Var') & (t.units == 'calibration-supported')]


def absolute_values(four_arms, values):
    """One row per prediction (``PREDICTIONS``): squared error in % of the design-aware baseline's total MSE, from the
    four-arms table (``arms.four_arms_table``), and the other eight metrics, from ``values`` (``values_table``); each
    the mean over the cases of the seed means. Written with ``to_csv(**data.WRITE)``."""
    rows, v = plus_var_seeds(four_arms), data.as_written(values, dtype={'seed': str})
    per_seed = v[v.seed != SEED_MEAN]
    out = []
    for name in PREDICTIONS:
        mse = 100. if name == 'design-aware baseline' else case_mean(rows[rows.arm == name], 'total_points')
        out.append({'prediction': name, 'cases': len(set(zip(rows.fold, rows.model))), 'seeds': len(SEEDS),
                    'mse_pct_of_design_aware': mse,
                    **{m: case_mean(per_seed[per_seed.prediction == name], m) for m in METRICS[1:]}})
    return pd.DataFrame(out)


def arm_changes(four_arms):
    """One row per +Var case (model, fold) and arm (``CHANGED_ARMS``) of the four-arms table: per seed, the arm's total,
    common and contrast error minus the original's, in % of the original's total; then the mean of the seeds. Written
    with ``to_csv(**data.WRITE)``."""
    rows = plus_var_seeds(four_arms)
    parts = rows[['total_points', 'common_points', 'contrast_points']].to_numpy()
    by = dict(zip(zip(rows.model, rows.fold, rows.seed, rows.arm), parts))
    seeds = [str(s) for s in SEEDS]
    out = []
    for model, fold in dict.fromkeys(zip(rows.model, rows.fold)):
        original = np.array([by[(model, fold, s, 'original')] for s in seeds])
        for arm in CHANGED_ARMS:   # in % of the original's total (the first of its three parts)
            change = 100 * (np.array([by[(model, fold, s, arm)] for s in seeds]) - original) / original[:, :1]
            out.append({'model': model, 'fold': fold, 'arm': arm, 'seeds': len(SEEDS),
                        **{f'{p}_pct_of_original': float(v)
                           for p, v in zip(('total', 'common', 'contrast'), change.mean(axis=0))}})
    return pd.DataFrame(out)
