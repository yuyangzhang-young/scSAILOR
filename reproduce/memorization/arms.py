"""The four-arms table on Tahoe-100M folds 1-3 (``four_arms_table``): each model's prediction with its context-level
correction kept (original), removed, or replaced by the training lines' constant or by the readers
(``readers.fold_arms``); on the +Var cases also calibrated toward the design-aware baseline, alone and with the
readers.

Predictions, three training seeds each:

* all held-out units (``originals``): the Mean-level fits of the summary-statistics, DeepSets and attention models, CPA
  and PrePR-CT (``data.ladder``), and State's +Var fits;
* calibration-supported units (``data.calibration_units``): origin resolution's +Var cases (``calibration.case``) of
  the summary-statistics, DeepSets and attention models and State, with their calibrated predictions.

The error of each arm, clip(prediction) - truth, is split into its common component (the context means,
``data.Design.project_ctx``) and its contrast component (the rest), each read, with the total, in % of the design-aware
baseline's total error on the same units (``*_points``). Each unit set has its own baseline: on all
held-out units ``b + kappa r_plate`` with ``data.fold_data``'s kappa; on the calibration-supported units origin
resolution's (``calibration.case``), with the case's kappa. ``seed_means_table``: the mean of the seeds' total error
per case and arm.
"""
import numpy as np
import pandas as pd

from ..origin_resolution import SEEDS
from . import calibration, data, readers

__all__ = ["MEAN_MODELS", "originals", "per_line_total", "score", "four_arms_table", "seed_means_table"]

MEAN_MODELS = ('summary', 'deepsets', 'attention_model', 'cpa', 'prepr')   # Mean-level fits, on all held-out units


def originals(fold):
    """The models' predictions on every held-out unit of the fold: (model, level, seed -> prediction) for the
    Mean-level fits of ``MEAN_MODELS``, then State's +Var fits."""
    for m in MEAN_MODELS:
        yield m, 'Mean', {s: data.ladder(fold, m, s) for s in SEEDS}
    yield 'state', '+Var', calibration.case(fold, 'state').predictions


def per_line_total(P, D, truth):
    """The total error of the prediction ``P`` clipped at zero, per held-out line (``D``'s lines); summed over lines,
    as ``score`` sums each part, it is the design-aware baseline's total of ``four_arms_table``."""
    return D.energy(np.maximum(P, 0) - truth, per_line=True)


def score(info, predictions, D, truth, base):
    """One row per arm (``predictions``: arm -> units x genes on the units of ``D``): its total error and the error's
    common and contrast components, in % of ``base``, each summed per held-out line and then over lines."""
    rows = []
    for arm, Q in predictions.items():
        E = np.maximum(Q, 0) - truth
        total, common = D.energy(E, per_line=True), D.energy(D.project_ctx(E), per_line=True)
        parts = {'total': total, 'common': common, 'contrast': total - common}
        rows.append({**info, 'arm': arm, **{f'{k}_points': 100 * v.sum() / base for k, v in parts.items()}})
    return rows


def four_arms_table():
    """One row per fold, model, seed, unit set and arm. Written with ``to_csv(**data.WRITE)``."""
    rows = []
    for fold in data.FOLDS:
        d, D = data.fold_data(fold), data.design(fold)
        truth = d['u']['truth']
        base = per_line_total(d['b'] + d['kappa'] * d['r_plate_unit'], D, truth).sum()
        for m, level, predictions in originals(fold):
            for s, P in predictions.items():
                info = {'fold': fold, 'model': data.MODELS[m], 'level': level, 'seed': s, 'units': 'all held-out'}
                rows += score(info, readers.fold_arms(fold, P), D, truth, base)
        _, uk, Dk = data.calibration_units(fold)
        for m in calibration.PLUS_VAR:
            c = calibration.case(fold, m)
            base_k = per_line_total(c.baseline, Dk, uk['truth']).sum()
            for s in SEEDS:
                info = {'fold': fold, 'model': data.MODELS[m], 'level': '+Var', 'seed': s,
                        'units': 'calibration-supported'}
                rows += score(info, c.arms(s), Dk, uk['truth'], base_k)
    return pd.DataFrame(rows)


def seed_means_table(four_arms):
    """One row per fold, model, level, unit set and arm of the four-arms table: the mean of the seeds' total error
    ('total_points'), from the table as written (the paper rounds these six-digit means). Written with
    ``to_csv(**data.WRITE)``."""
    keys = ['fold', 'model', 'level', 'units', 'arm']
    return data.as_written(four_arms).groupby(keys, sort=False).total_points.mean().reset_index()
