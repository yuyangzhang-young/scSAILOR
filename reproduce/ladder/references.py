"""Model-free references on Parse PBMC and Jiang et al. (Supplementary Note S5.3, Results 2), on the held-out prediction
units the models are scored on, from the fold data under the data root; no model enters.

* no change ``b``: the mean of the held-out context's input set;
* the baseline ``a``: ``b`` plus the average training response;
* the stratum mean ``A``: ``b`` plus the average training response within the prediction unit's stratum (cell type
  on Parse PBMC, stimulus on Jiang et al.); a perturbation absent from the stratum keeps the overall average
  (``scsailor.resolution.net_cost.stratum_shift``);
* the design-aware baseline ``b + kappa (A - b)``: the best scale ``kappa`` is the weighted least-squares slope of
  truth - ``b`` on ``A - b``, clipped to 0-1 (``scsailor.resolution.net_cost.calibrate_design``), over the training
  units of the calibration lines whose perturbation the fitted lines hold, with ``A`` built from the fitted lines'
  responses (the nested split: the training lines, sorted, permuted with numpy ``default_rng(2026100803)``; six
  donors fitted and two calibration donors on Parse PBMC, three fitted and two calibration cell lines on Jiang et al.).

The held-out references use the average training responses over all training lines (fitted and calibration). Predictions
are clipped at zero before scoring. The strongest reference of a fold is the baseline, stratum mean or design-aware
baseline with the lowest total MSE; on a tie, the first of these (on Parse PBMC G1 the best scale is 1, and the
stratum mean and design-aware baseline are equal: the stratum mean is marked).
"""
import numpy as np
import pandas as pd

from scsailor.resolution.net_cost import calibrate_design, design_baseline
from scsailor.resolution.scoring import retained, scores
from scsailor.resolution.stages import bag_of

from ..origin_resolution import bag_means, heldout_units, jiang, parse, stratum_design_shift
from . import fits

__all__ = ["nested_split", "predictions", "references_table", "models_table"]

SPLIT_SEED = 2026100803
FITTED = {'Parse PBMC': 6, 'Jiang et al.': 3}   # lines fitted in the nested split
REFERENCES = ('no change', 'baseline', 'stratum mean', 'design-aware baseline')


def nested_split(lines, fitted):
    """(fitted lines, calibration lines) of the nested split of a fold's training ``lines``."""
    order = np.random.default_rng(SPLIT_SEED).permutation(sorted(lines)).tolist()
    return order[:fitted], order[fitted:]


def fold_inputs(screen, fold):
    """A fold's training units, held-out prediction units, input-set and pool means, and the line and stratum of a
    context."""
    binding = parse if screen == 'Parse PBMC' else jiang
    f = binding.files(fold)
    return {'train': binding.fold_training_units(fold), 'held': heldout_units(f['outer']),
            'bags': bag_means(f['inputs']), 'line_of': binding.line_of, 'stratum_of': binding.stratum_of}


def predictions(inputs, units, side, lines):
    """(b, a, A) for prediction ``units`` (input sets from ``side``), with the training responses of the training
    ``lines``; ``inputs`` as ``fold_inputs`` returns them."""
    t = inputs['train']
    keep = np.isin([inputs['line_of'](c) for c in t['context']], lines)
    change, w, q = t['truth'] - bag_of(inputs['bags']['train'], t)[1], t['weight'], t['p_code']
    r = {int(k): np.average(change[keep & (q == k)], axis=0, weights=w[keep & (q == k)]) for k in np.unique(q[keep])}
    b = bag_of(inputs['bags'][side], units)[0]
    a = b + np.stack([r[int(k)] for k in units['p_code']])
    return b, a, a + stratum_design_shift(units, t, inputs['bags']['train'], lines, inputs['line_of'],
                                          inputs['stratum_of'])


def fold_references(screen, fold):
    """kappa, the number of held-out prediction units and the (total, common, contrast) MSE of each reference on
    them."""
    c = fold_inputs(screen, fold)
    t = c['train']
    lines = np.array([c['line_of'](x) for x in t['context']])
    fit, calibration = nested_split(set(lines), FITTED[screen])
    fitted_codes = np.unique(t['p_code'][np.isin(lines, fit)])
    cal = retained(t, np.isin(lines, calibration) & np.isin(t['p_code'], fitted_codes))
    b, _, A = predictions(c, cal, 'train', fit)
    kappa = calibrate_design({'u': cal, 'b': b, 'A': A})
    b, a, A = predictions(c, c['held'], 'heldout', fit + calibration)
    out = {name: scores(p, c['held']) for name, p in
           zip(REFERENCES, (b, a, A, design_baseline({'b': b, 'A': A}, kappa)))}
    return kappa, len(c['held']['truth']), {n: tuple(s[k] for k in fits.MSE) for n, s in out.items()}


def references_table():
    """Per screen, fold and reference: relative MSE (over the baseline's total MSE), the common and contrast MSE minus
    the baseline's in % of its total MSE (``fits.score_columns``), the best scale, and whether it is the strongest
    reference."""
    rows = []
    for screen in FITTED:
        for fold in fits.FOLDS[screen]:
            kappa, units, mse = fold_references(screen, fold)
            strongest = min(REFERENCES[1:], key=lambda n: mse[n][0])
            for name in REFERENCES:
                s = fits.score_columns(mse[name], mse['baseline'])
                rows.append(dict(screen=screen, fold=fold, units=units, reference=name, relative_mse=s['relative_mse'],
                                 delta_common_points=s['delta_common_points'],
                                 delta_contrast_points=s['delta_contrast_points'],
                                 best_scale=kappa if name == 'design-aware baseline' else np.nan,
                                 strongest=name == strongest))
    return pd.DataFrame(rows)


def models_table(points, refs):
    """Per screen, model, level and fold of Parse PBMC and Jiang et al. (and the mean over folds): the model's MSE minus
    the strongest reference's, in % of the baseline's total MSE, i.e. 100 x (the model's fold point of relative MSE in
    the ladder's ``points`` minus the strongest reference's relative MSE in ``refs``)."""
    strongest = refs[refs['strongest']].set_index(['screen', 'fold'])['relative_mse']
    rows = []
    for (screen, model, level), mine in points[points['screen'].isin(FITTED)].groupby(['screen', 'model', 'level'],
                                                                                      sort=False):
        mine = mine[mine['fold'] != 'fold mean']
        diffs = [100 * (v - strongest[(screen, f)]) for f, v in zip(mine['fold'], mine['relative_mse'])]
        rows += [dict(screen=screen, model=model, level=level, fold=f, minus_strongest=d)
                 for f, d in zip(mine['fold'], diffs)]
        rows.append(dict(screen=screen, model=model, level=level, fold='fold mean',
                         minus_strongest=float(np.mean(diffs))))
    return pd.DataFrame(rows)
