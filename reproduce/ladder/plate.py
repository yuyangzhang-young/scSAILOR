"""The plate-aware baseline on Tahoe-100M (Results 1, Supplementary Note S5.2, Supplementary Table 7, and Supplementary
Data 2's Tahoe plate comparison), from the fold data under the data root; no model enters.

The plate-aware baseline is the held-out context's input-set mean plus the training lines' drug average on the
context's plate; the baseline is the input-set mean plus the drug average over both plates. Both are clipped at zero
and scored like the models (``scsailor.resolution.scoring.scores``; the adapted weighted R2 of Supplementary Note S3,
``weighted_r2.adapted_r2``). The fold data are read through the memorization layer's ``reproduce.memorization.data
.fold_data``, which takes the plate from the context's label; the plate read from each held-out input set agrees with
its label in every held-out context (Supplementary Note S5.2), so the values are those of the plate read from the input
cells.
"""
import numpy as np
import pandas as pd

from scsailor.resolution.scoring import scores

from . import fits
from . import weighted_r2 as r2w

__all__ = ["reductions", "plate_table", "supp_table_7", "comparison_table"]


def reductions(prediction, baseline, u):
    """A prediction against the baseline on prediction units ``u``, both clipped at zero: relative MSE, and the
    reductions of common and contrast MSE (minus ``fits.score_columns``' model-minus-baseline components) in % of the
    baseline's total MSE."""
    s = fits.score_columns(*(tuple(scores(x, u)[k] for k in fits.MSE) for x in (prediction, baseline)))
    return {'relative_mse': s['relative_mse'], 'common_reduction': -s['delta_common_points'],
            'contrast_reduction': -s['delta_contrast_points']}


def plate_table():
    """Per fold: the plate-aware baseline's ``reductions`` and its adapted weighted R2 (total version)."""
    from ..memorization.data import fold_data
    rows = []
    for fold in fits.FOLDS['Tahoe-100M']:
        d = fold_data(fold)
        plate_aware = d['b'] + d['r_plate_unit']
        rows.append(dict(fold=fold, **reductions(plate_aware, d['A0'], d['u']),
                         weighted_r2=r2w.adapted_r2(plate_aware, d['u'])['weighted_r2']))
    return pd.DataFrame(rows)


def supp_table_7(points, plate_rows):
    """Supplementary Table 7 from the ladder's ``points`` (``ladder.summarize``) and ``plate_table``'s rows: the
    plate-aware baseline's relative MSE per fold and its reductions averaged over folds; per model the fold points at
    Cells, the lowest fold point over the levels with its level, and the reductions at Cells (minus the components'
    fold means)."""
    rows = [dict(model='Plate-aware baseline',
                 **{f'relative_mse_{f}': v for f, v in zip(plate_rows['fold'], plate_rows['relative_mse'])},
                 common_reduction=float(plate_rows['common_reduction'].mean()),
                 contrast_reduction=float(plate_rows['contrast_reduction'].mean()))]
    tahoe = points[points['screen'] == 'Tahoe-100M']
    for model, mine in tahoe.groupby('model', sort=False):
        row = dict(model=model)
        for fold in plate_rows['fold']:
            at = mine[mine['fold'] == fold].set_index('level')['relative_mse']
            row |= {f'relative_mse_{fold}': at['Cells'], f'lowest_{fold}': at.min(),
                    f'lowest_level_{fold}': at.idxmin()}
        cells = mine[(mine['fold'] == 'fold mean') & (mine['level'] == 'Cells')].iloc[0]
        rows.append(row | dict(common_reduction=-cells['delta_common_points'],
                               contrast_reduction=-cells['delta_contrast_points']))
    return pd.DataFrame(rows)


def comparison_table(fit_table, plate_rows):
    """Supplementary Data 2's Tahoe plate comparison: per fold, model and level, the mean over the kept fits (seeds) of
    their adapted weighted R2 beside the plate-aware baseline's (``plate_table``) and the baseline's (Supplementary Fig.
    1's), and whether the model's is higher."""
    kept = fit_table[(fit_table['screen'] == 'Tahoe-100M') & (fit_table['fit_status'] == 'kept')]
    plate_aware = plate_rows.set_index('fold')['weighted_r2']
    rows = []
    for (fold, model, level), seeds in kept.groupby(['fold', 'model', 'level'], sort=False):
        value = float(np.mean(seeds['weighted_r2']))
        rows.append(dict(fold=fold, model=model, level=level, seeds=len(seeds), weighted_r2=value,
                         plate_aware_weighted_r2=plate_aware[fold],
                         baseline_weighted_r2=r2w.reference_scores()[('Tahoe-100M', fold, 'baseline')]['weighted_r2'],
                         above_plate_aware=value > plate_aware[fold]))
    return pd.DataFrame(rows)
