"""The separate growth-gene correction of origin resolution's calibrated models, and what the routine read-outs and the
pair-median growth relation see of it.

Terms as in ``data``, ``readers``, ``calibration`` and ``ranking``. A unit's response is ``T - b`` (truth minus the
input-set mean).

**Linear reader** (``linear_reader``). For each drug, kernel ridge from a context's per-gene input-set means to the
response's deviation ``Y = T - b - ybar``, ``ybar`` the drug's mean response over the training contexts on the unit's
plate. The means are centred per plate on the training contexts, divided by their SD over them and scaled to a unit
mean squared norm over them (``mean_kernel``). One penalty for every drug, from ``PENALTIES``, is chosen by leaving one
training line out (both plates together), the drug's plate means recomputed without it (``LinearReader``). A held-out
unit's prediction is ``b + ybar`` plus the fitted deviation, in single precision (the paper's tables read it from a
saved single-precision copy).

**Growth-gene correction** (``growth_correction``). On the growth genes ``G`` (``programs``), a prediction's
context-level correction relative to the design-aware baseline ``Th`` is replaced by the linear reader's ``R``:
``X[:, G] += proj_ctx(R - Th)[:, G] - proj_ctx(X - Th)[:, G]``. A context's differences between drugs are kept.

**Growth relation.** The pair-median growth relation (``ranking.pair_median``): over the fold's drug-program pairs
(``selection.pairs``), the median of the cross-plate Spearman correlation across held-out lines of the predicted
program response (``clip(P) - b``) with the measured one, both directions averaged.

**Tables**, on origin resolution's 12 +Var cases (``calibration.case``: four models, folds 1-3, the
calibration-supported units, each seed calibrated with its own amounts): ``readouts_table``, per case, seed and arm
(``SEED_ARMS``, then ``FIXED_ARMS``, the same for every seed), the seven routine read-outs (``Readouts``, of the
prediction clipped at zero), its total error in % of the design-aware baseline's and its growth relation;
``changes_table``, the change of 'calibrated + growth-gene correction' from 'calibrated', per seed and for the seed
mean ('seed mean': the seeds' mean change; the total error's change in % of the design-aware baseline's total, with
origin resolution's line bootstrap on the seeds pooled within each line, and in % of the calibrated prediction's).
"""
from functools import cache

import numpy as np
import pandas as pd

from scsailor.natural.score import row_correlation
from scsailor.resolution import comparison

from ..origin_resolution import SEEDS, tahoe
from . import calibration, data, ranking, readers

__all__ = ["PENALTIES", "READOUTS", "CHANGE_READOUTS", "SEED_ARMS", "FIXED_ARMS", "mean_kernel", "lolo_errors",
           "LinearReader", "linear_reader", "growth_correction", "Readouts", "seed_arms", "change_row", "seed_mean_row",
           "readouts_table", "changes_table"]

PENALTIES = np.logspace(-2, 3, 11)
READOUTS = ('systema_pearson_delta_observed_unweighted', 'pearson_delta_unweighted', 'centroid_accuracy', 'r_template',
            'r_residual', 'r_gamma', 'r_beta')   # the seven routine read-outs (``Readouts``)
CHANGE_READOUTS = ('systema_pearson_delta_observed_unweighted', 'centroid_accuracy', 'pearson_delta_unweighted',
                   'r_template', 'r_residual', 'r_gamma', 'r_beta',
                   'growth_biology_median')   # the change table's read-outs, in its column order
CORRECTED = 'calibrated + growth-gene correction'
SEED_ARMS = ('original', 'original + growth-gene correction', 'calibrated', CORRECTED)   # per training seed
FIXED_ARMS = ('design-aware baseline', 'linear reader')   # the same for every seed


# ------------------------------------------------------------------------------------------------- linear reader

def mean_kernel(means, plate, train):
    """The linear kernel of the contexts' per-gene input-set means, centred per plate on the ``train`` rows, divided by
    their SD over them and scaled to a unit mean squared norm over them."""
    Z = readers.centre_scale(means, plate, train)
    Z /= np.sqrt(np.sum(Z[train] ** 2) / train.sum())
    return Z @ Z.T


def lolo_errors(K, targets, penalties):
    """Per penalty, the mean squared error over units and genes of leaving each training line out of each drug's
    reader (both plates together), the drug's plate means recomputed without the line."""
    err = np.zeros(len(penalties))
    count = 0
    for t in targets.values():
        for held in np.unique(t['line']):
            out = t['line'] == held
            Ytr, Yte = t['Y'][~out], t['Y'][out]
            for p in tahoe.PLATES:
                mo, mi = out & (t['plate'] == p), (~out) & (t['plate'] == p)
                n = mo.sum() + mi.sum()
                if mo.any() and n > 1:
                    # one unit per line, plate and drug; Y is centred, so without the line the plate mean of the other
                    # n - 1 is -Y_out / (n - 1), and every target moves by Y_out / (n - 1)
                    Ytr[t['plate'][~out] == p] += t['Y'][mo].sum(axis=0) / (n - 1)
                    Yte[t['plate'][out] == p] = t['Y'][mo] * n / (n - 1)
            pred = readers.kernel_ridge(K, t['rows'][~out], Ytr, t['rows'][out], penalties)
            err += np.sum((pred - Yte[None]) ** 2, axis=(1, 2))
            count += Yte.size
    return err / count


class LinearReader:
    """Per drug, kernel ridge (kernel ``K``) from a training context's input set to the deviation of its response
    ``delta`` (training units x genes) from the drug's plate mean; one penalty for every drug, chosen by
    ``lolo_errors`` (kept in ``errors``). On Tahoe-100M every held-out drug has training units."""

    def __init__(self, K, delta, line, plate, code, rows):
        """line, plate, code: each training unit's line, plate and drug; rows: its context's kernel row."""
        self.K, self.targets = K, {}
        for q in np.unique(code):
            m = np.flatnonzero(code == q)
            on, Y, ybar = plate[m], np.empty_like(delta[m]), {}
            for p in tahoe.PLATES:
                ybar[p] = delta[m][on == p].mean(axis=0)
                Y[on == p] = delta[m][on == p] - ybar[p]
            self.targets[int(q)] = {'rows': rows[m], 'line': line[m], 'plate': on, 'Y': Y, 'ybar': ybar}
        self.errors = lolo_errors(K, self.targets, PENALTIES)
        self.penalty = PENALTIES[int(np.argmin(self.errors))]

    def predict(self, b, rows, plate, code):
        """Per unit (input-set mean ``b``, kernel row, plate, drug): ``b`` plus the drug's plate mean plus the fitted
        deviation."""
        out = np.empty_like(b)
        for q in np.unique(code):
            t, m = self.targets[int(q)], np.flatnonzero(code == q)
            ybar = np.stack([t['ybar'][p] for p in plate[m]])
            out[m] = b[m] + ybar + readers.kernel_ridge(self.K, t['rows'], t['Y'], rows[m], [self.penalty])[0]
        return out


@cache
def linear_reader(fold):
    """The linear reader's prediction of every held-out unit of ``data.fold_data(fold)``, rounded to single precision
    as the paper's tables read it (from a saved single-precision copy)."""
    d = data.fold_data(fold)
    tr, u = d['train'], d['u']
    contexts, train, means = data.all_contexts(d['bags'])
    K = mean_kernel(means, data.plates(contexts), train)
    row = {c: i for i, c in enumerate(contexts)}
    reader = LinearReader(K, tr['u']['truth'] - tr['b'], tr['line'], tr['plate'], tr['u']['p_code'],
                          np.array([row[c] for c in tr['u']['context']]))
    P = reader.predict(d['b'], np.array([row[c] for c in u['context']]), d['plate'], d['code'])
    return P.astype(np.float32).astype(np.float64)


def growth_correction(X, R, Th, D, G):
    """``X`` with its context-level correction relative to ``Th`` on the genes ``G`` replaced by ``R``'s (contexts and
    weights of the decomposition ``D``)."""
    out = X.copy()
    out[:, G] += D.project_ctx(R - Th)[:, G] - D.project_ctx(X - Th)[:, G]
    return out


# ------------------------------------------------------------------------------------------------- read-outs

class Readouts:
    """The seven routine read-outs of a prediction ``P`` against truth ``T`` (``__call__``), with the context's
    control-pool mean ``C`` (``delta = T - C``, ``dhat = P - C``), units in the error readings ``D``, context index
    ``ctx``:

    * systema_pearson_delta_observed_unweighted: mean over units of the correlation over genes of P - O and T - O, O
      the context's unweighted mean truth. This is the nine metrics' ``systema_pearson_delta_observed`` (Systema
      Pearson Delta (observed)) read without weights, not their ``systema_pearson_delta``, which subtracts C + the mean
      training response;
    * pearson_delta_unweighted: the same for dhat and delta, the nine metrics' ``pearson_delta`` (Pearson Delta) read
      without weights;
    * centroid_accuracy: mean over units of the share of the context's other drugs whose truth lies farther from P
      than the unit's own;
    * r_template: mean over contexts of the correlation over drugs of the projections of delta and dhat on the
      template (the context's mean delta);
    * r_residual: mean over units of the correlation over genes of the template-removed residuals;
    * r_gamma, r_beta: mean over units of the correlation over genes of dhat with the interaction (line-specific) part
      of delta, and with its drug main effect (the weighted drug mean of delta's within-context contrast).

    The parts that depend on the truth alone are computed once."""

    def __init__(self, T, C, D, ctx):
        def unweighted_context_mean(X):
            """Each unit's unweighted mean of ``X`` over its context's units."""
            return data.context_means(X, ctx, np.ones(len(ctx)))[1][ctx]

        self.C = C
        self.O = unweighted_context_mean(T)
        self.truth_minus_O = T - self.O
        self.delta = T - C
        self.template = unweighted_context_mean(self.delta)
        self.template_norm2 = (self.template * self.template).sum(axis=1)
        # each unit's coefficient of delta on its context's template
        self.delta_coef = (self.delta * self.template).sum(axis=1) / self.template_norm2
        self.residual = self.delta - self.delta_coef[:, None] * self.template
        self.gamma = D.interaction_of(self.delta)
        contrast = self.delta - D.project_ctx(self.delta)
        drug = pd.factorize(D.u['p_code'])[0]
        self.beta = data.context_means(contrast, drug, D.w)[1][drug]
        self.groups = [np.flatnonzero(ctx == c) for c in np.unique(ctx)]
        self.truth_by_context = [T[i] for i in self.groups]
        self.truth_norm2_by_context = [(t ** 2).sum(axis=1)[None, :] for t in self.truth_by_context]

    def __call__(self, P):
        """The read-outs of the prediction ``P`` (units x genes), by name."""
        out = {}
        out['systema_pearson_delta_observed_unweighted'] = float(np.nanmean(row_correlation(P - self.O,
                                                                                            self.truth_minus_O)))
        dhat = P - self.C
        out['pearson_delta_unweighted'] = float(np.nanmean(row_correlation(dhat, self.delta)))
        acc = []
        for i, t, t2 in zip(self.groups, self.truth_by_context, self.truth_norm2_by_context):
            # squared distance of each prediction to each truth in the context (rows: predictions)
            d2 = (P[i] ** 2).sum(axis=1)[:, None] + t2 - 2 * P[i] @ t.T
            acc.append((d2 > np.diag(d2)[:, None]).sum(axis=1) / (len(i) - 1))
        out['centroid_accuracy'] = float(np.mean(np.concatenate(acc)))
        dhat_coef = (dhat * self.template).sum(axis=1) / self.template_norm2
        out['r_template'] = float(np.nanmean([np.corrcoef(self.delta_coef[i], dhat_coef[i])[0, 1]
                                              for i in self.groups]))
        out['r_residual'] = float(np.nanmean(row_correlation(dhat - dhat_coef[:, None] * self.template,
                                                             self.residual)))
        out['r_gamma'] = float(np.nanmean(row_correlation(dhat, self.gamma)))
        out['r_beta'] = float(np.nanmean(row_correlation(dhat, self.beta)))
        return out


@cache
def fold_readouts(fold):
    """The read-outs on the fold's calibration-supported units (``Readouts``)."""
    keep, u, D = data.calibration_units(fold)
    return Readouts(u['truth'], data.fold_data(fold)['C'][keep], D, pd.factorize(u['context'])[0])


# ------------------------------------------------------------------------------------------------- tables

def seed_arms(P, calibrated, R, Th, D, G):
    """One seed's predictions of ``SEED_ARMS``: its original prediction ``P`` and its calibrated prediction, each alone
    and with the growth-gene correction (``growth_correction``: linear reader ``R``, design-aware baseline ``Th``,
    growth genes ``G``)."""
    return dict(zip(SEED_ARMS, (P, growth_correction(P, R, Th, D, G), calibrated,
                                growth_correction(calibrated, R, Th, D, G))))


def change_row(readings, line_errors, line_names, base):
    """One seed's change of 'calibrated + growth-gene correction' from 'calibrated' (``readings``: arm -> read-outs;
    ``line_errors``: arm -> total error per held-out line, lines ``line_names``): each read-out's difference
    (``CHANGE_READOUTS``), and the total error's, in % of the design-aware baseline's total ``base`` with its 95% line
    interval (origin resolution's line bootstrap) and in % of the calibrated prediction's total. Returns the row and
    the per-line change (line -> change) that the seed mean pools."""
    diff = line_errors[CORRECTED] - line_errors['calibrated']
    per_line = dict(zip(line_names, diff))
    iv = comparison.bootstrap(per_line, base)
    row = {**{k: readings[CORRECTED][k] - readings['calibrated'][k] for k in CHANGE_READOUTS},
           'total_error_points': iv['estimate_pct'], 'low': iv['interval_pct'][0], 'high': iv['interval_pct'][1],
           'total_error_pct_of_calibrated': 100 * diff.sum() / line_errors['calibrated'].sum()}
    return row, per_line


def seed_mean_row(rows, per_line, base):
    """The seed mean of the seeds' ``change_row`` rows: their mean changes, and the total error's change with origin
    resolution's line bootstrap on the seeds pooled within each line (``per_line``: seed -> line -> change)."""
    means = pd.DataFrame(rows)[list(CHANGE_READOUTS) + ['total_error_pct_of_calibrated']].mean().to_dict()
    iv = comparison.bootstrap(*comparison.pool_seeds([(per_line[s], base) for s in SEEDS]))
    return {**means, 'total_error_points': iv['estimate_pct'], 'low': iv['interval_pct'][0],
            'high': iv['interval_pct'][1]}


@cache
def case_rows(fold, model):
    """One +Var case's read-out rows (``SEED_ARMS`` per seed, then ``FIXED_ARMS``) and change rows (per seed, then the
    seed mean)."""
    keep, u, D = data.calibration_units(fold)
    R = linear_reader(fold)[keep]
    c = calibration.case(fold, model)
    growth, readouts = ranking.growth_response(fold), fold_readouts(fold)
    Th, T = c.baseline, u['truth']
    # the design-aware baseline's total error in one sum over units, the order of the paper's tables (the four-arms
    # table sums per line first)
    base = D.energy(np.maximum(Th, 0) - T)

    def read(X):
        """The read-outs of ``X`` clipped at zero, its total error in % of ``base`` and its growth relation."""
        P = np.maximum(X, 0)
        return {**readouts(P), 'total_error_pct_of_design_aware': 100 * D.energy(P - T) / base,
                'growth_biology_median': growth.pairs_median(growth.responses(P, keep))}

    def line_errors(X):
        """The total error of ``X`` clipped at zero, per held-out line."""
        return D.energy(np.maximum(X, 0) - T, per_line=True)

    info = {'fold': fold, 'level': '+Var', 'model': data.MODELS[model]}
    fixed = dict(zip(FIXED_ARMS, (read(Th), read(R))))
    rows, changes, per_line = [], [], {}
    for s in SEEDS:
        arms = seed_arms(c.predictions[s][keep], c.calibrated[s], R, Th, D, growth.G)
        readings = {arm: read(X) for arm, X in arms.items()}
        rows += [{**info, 'seed': s, 'arm': arm, 'n_units': len(T), **v} for arm, v in (readings | fixed).items()]
        errors = {arm: line_errors(arms[arm]) for arm in ('calibrated', CORRECTED)}
        change, per_line[s] = change_row(readings, errors, D.line_names, base)
        changes.append({**info, 'seed': s, **change})
    changes.append({**info, 'seed': data.SEED_MEAN, **seed_mean_row(changes, per_line, base)})
    return rows, changes


def case_tables():
    """``case_rows`` of the 12 +Var cases."""
    return [case_rows(fold, model) for fold in data.FOLDS for model in calibration.PLUS_VAR]


def readouts_table():
    """One row per case, seed and arm (``SEED_ARMS``, ``FIXED_ARMS``): the number of calibration-supported units
    ('n_units'), the routine read-outs (``READOUTS``), the total error in % of the design-aware baseline's
    ('total_error_pct_of_design_aware') and the pair-median growth relation ('growth_biology_median'). Written with
    ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for rows, _ in case_tables() for r in rows])


def changes_table():
    """One row per case and seed (and the seed mean): the change of 'calibrated + growth-gene correction' from
    'calibrated' in each read-out (``CHANGE_READOUTS``) and in total error, in % of the design-aware baseline's total
    with its 95% line interval ('total_error_points', 'low', 'high') and in % of the calibrated prediction's total.
    Written with ``to_csv(**data.WRITE)``."""
    table = pd.DataFrame([r for _, changes in case_tables() for r in changes])
    table = table.assign(units='calibration-supported', predictor=CORRECTED, against='calibrated')
    return table[['fold', 'level', 'model', 'units', 'predictor', 'against', 'seed', *CHANGE_READOUTS,
                  'total_error_points', 'low', 'high', 'total_error_pct_of_calibrated']]
