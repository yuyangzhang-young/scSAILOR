"""The information ladder's values: per screen, model, fold and level the mean over the kept fits (seeds), and the mean
over folds (Figs. 1e-f and 2b-d), from the per-fit table; Parse PBMC on twelve and on ten donors (Supplementary Table
2); the fits the fit rules set aside and what reinstating them changes; no change against the models.

A fold point is the mean over the kept fits of a model, level and fold; a fold mean is the mean of a level's fold
points. Component values are model minus baseline in % of the baseline's total MSE (``delta_*_points``). A change from
Mean is the level's fold point minus the Mean level's fold point of the same model and fold, each over its own kept
fits (the two seed sets differ where a fit is set aside at one level only, or a replacement seed enters); its fold mean
is the mean of the folds' changes.
"""
import numpy as np
import pandas as pd

from . import fits

__all__ = ["summarize", "parse_donors_table", "exclusions_table", "sensitivity_table", "no_change_table"]

VALUES = ('relative_mse', 'delta_common_points', 'delta_contrast_points')
LEFT_OUT_DONORS = ('Donor2', 'Donor3')   # left out of scoring in Parse PBMC's ten-donor sensitivity analysis


def summarize(table):
    """Fold points and fold means of ``VALUES`` per screen, model and level of a per-fit table, with the changes of the
    two components from Mean (module docstring)."""
    kept = table[table['fit_status'] == 'kept']
    keys = ['screen', 'model', 'fold', 'level']
    points = kept.groupby(keys, sort=False)[list(VALUES)].mean().reset_index()
    points['seeds'] = kept.groupby(keys, sort=False)['seed'].agg(lambda s: ';'.join(sorted(s))).values
    mean_level = points[points['level'] == 'Mean'].set_index(['screen', 'model', 'fold'])
    at = pd.MultiIndex.from_frame(points[['screen', 'model', 'fold']])
    for part in ('common', 'contrast'):
        points[f'{part}_change_from_mean'] = (points[f'delta_{part}_points'].values -
                                              mean_level.loc[at, f'delta_{part}_points'].values)
    columns = list(VALUES) + ['common_change_from_mean', 'contrast_change_from_mean']
    means = points.groupby(['screen', 'model', 'level'], sort=False)[columns].mean().reset_index()
    means['fold'] = 'fold mean'
    return pd.concat([points, means], ignore_index=True)[keys + ['seeds'] + columns]


def parse_donors_table(fit_table):
    """Parse PBMC's fits scored on all twelve donors (``fit_table``'s rows) and with ``LEFT_OUT_DONORS`` left out of
    scoring, each ``summarize``d. The fit rules are applied within each evaluation; Parse PBMC's files hold no training
    MSE, so only the outlier rule can act, and it sets no fit aside in either."""
    views = (('all twelve donors', fit_table[fit_table['screen'] == 'Parse PBMC']),
             ('ten donors', fits.named(fits.table(['Parse PBMC'], LEFT_OUT_DONORS))))
    return pd.concat([summarize(t).assign(evaluation=name) for name, t in views], ignore_index=True)


def exclusions_table(fit_table):
    """The fits the fit rules set aside, with their relative MSE."""
    return fit_table.loc[fit_table['fit_status'] != 'kept', ['fit_id', 'screen', 'fold', 'model', 'seed', 'level',
                                                             'fit_status', 'relative_mse']].reset_index(drop=True)


def sensitivity_table(fit_table):
    """For every screen, model and level with a fit set aside as an outlier, the fold mean of relative MSE with the
    rules applied and with the outliers reinstated."""
    outlier = fit_table['fit_status'].str.startswith('excluded: outlier')
    reinstated = fit_table.assign(fit_status=np.where(outlier, 'kept', fit_table['fit_status']))
    keys = ['screen', 'model', 'level']
    at = fit_table.loc[outlier, keys].drop_duplicates()
    value = lambda t: summarize(t).query("fold == 'fold mean'").merge(at)[keys + ['relative_mse']]   # noqa: E731
    return value(fit_table).merge(value(reinstated), on=keys, suffixes=('', '_outliers_reinstated'))


def no_change_table(points):
    """Per fold of GSE306429, Replogle–Nadig and Jiang et al.: no change's relative MSE (``fits.no_change``) and the
    lowest fold point of any model and level among the ladder's ``points`` (``summarize``), with its model and
    level."""
    rows = []
    for screen in ('GSE306429', 'Replogle–Nadig', 'Jiang et al.'):
        for fold in fits.FOLDS[screen]:
            best = points[(points['screen'] == screen) & (points['fold'] == fold)].sort_values('relative_mse').iloc[0]
            rows.append(dict(screen=screen, fold=fold, no_change=fits.no_change(screen, fold),
                             lowest_model=best['relative_mse'], lowest_model_name=best['model'],
                             lowest_model_level=best['level']))
    return pd.DataFrame(rows)
