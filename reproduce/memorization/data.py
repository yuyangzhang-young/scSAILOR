"""One Tahoe-100M fold's held-out units, the models' saved predictions, and the readings of a prediction's error.

A fold has 14 held-out lines on two replicate plates and 93 drugs; a unit is one line, plate and drug. A unit's truth
``T`` is the mean of its treated cells; ``b`` is the mean of the 256 control cells a model receives (its input set), and
``C`` the mean of the context's whole control pool (a context is one line on one plate). ``r`` is the training lines'
drug average (the average training response), ``b + r`` the anchor, and ``b + kappa r_plate`` the design-aware baseline
(``r_plate``: the drug average on the unit's plate; ``kappa``: the paper's lambda_r). Predictions are clipped at zero
before scoring.

An error ``X`` is read in the weighted inner product of the natural task's score (``Design``): its energy is the
weighted mean over units of the mean over genes of ``X^2``; its common part is each context's weighted mean over its
drugs; its interaction part is the rest after a weighted least-squares fit by context + drug x plate, the line-specific
part of a drug's effect. The calibration-supported units (``calibration_units``) are those whose drug occurs on the
calibration lines, where origin resolution's calibration is defined.

The data root (``SCSAILOR_DATA``, as for ``reproduce.origin_resolution``) holds the fold files and, in
``tahoe-100m-ladder/<fold>/<model>_<seed>_L1.npz``, each model's saved Mean-level predictions on the held-out and the
training lines at its fixed training length. Every table is written with ``to_csv(**WRITE)``; a table computed from
other tables reads them as written (``as_written``).
"""
import ast
import io
from functools import cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scsailor.resolution import grids
from scsailor.resolution.scoring import retained
from scsailor.resolution.stages import bag_of

from ..origin_resolution import ROOT, bag_means
from ..origin_resolution import tahoe

__all__ = ["FOLDS", "MODELS", "LADDER", "SEED_MEAN", "WRITE", "lines", "plates", "context_means", "drug_label",
           "input_means", "pool_means", "all_contexts", "design_kappa", "fold_data", "Design", "design", "supported",
           "calibration_units", "ladder_file", "ladder", "as_written"]

FOLDS = ('F1', 'F2', 'F3')
MODELS = {'summary': 'Summary-statistics', 'deepsets': 'DeepSets', 'attention_model': 'Attention', 'state': 'State',
          'cpa': 'CPA', 'prepr': 'PrePR-CT'}   # code name -> the name the tables print
LADDER = ROOT / 'tahoe-100m-ladder'
SEED_MEAN = 'seed mean'   # the label of the rows that hold the mean of the seeds' values
WRITE = {'index': False, 'float_format': '%.6g'}   # to_csv arguments of every table


def lines(contexts):
    """The line of each context key ('line|plate')."""
    return np.array([c.split('|')[0] for c in contexts])


def plates(contexts):
    """The plate of each context key ('line|plate')."""
    return np.array([c.split('|')[1] for c in contexts])


def context_means(X, group, weight):
    """The groups (sorted; ``group``: each row's, e.g. its context) and per group the weighted mean of ``X``'s rows."""
    labels, inverse = np.unique(group, return_inverse=True)
    M = np.zeros((len(labels), X.shape[1]))
    np.add.at(M, inverse, weight[:, None] * X)
    return labels, M / np.bincount(inverse, weights=weight)[:, None]


def drug_label(raw):
    """'[('Adagrasib', 0.05, 'uM')]' -> 'Adagrasib 0.05 uM'."""
    name, dose, unit = ast.literal_eval(raw)[0]
    return f'{name} {dose:g} {unit}'


def input_means(side_bags, contexts):
    """The input-set means of ``contexts`` (one row each, in that order), from one side of ``fold_data``'s ``bags``."""
    return np.stack([side_bags[c][0] for c in contexts])


def pool_means(side_bags, contexts):
    """The control-pool means of ``contexts`` (one row each, in that order), from one side of ``fold_data``'s
    ``bags``."""
    return np.stack([side_bags[c][1] for c in contexts])


def all_contexts(bags):
    """Every context of ``fold_data``'s ``bags``, the training contexts first (each side sorted); whether each is a
    training context; and their input-set means (contexts x genes)."""
    train, held = sorted(bags['train']), sorted(bags['heldout'])
    means = np.concatenate([input_means(bags['train'], train), input_means(bags['heldout'], held)])
    return train + held, np.arange(len(train) + len(held)) < len(train), means


def design_kappa(x, y, w):
    """The design-aware amount kappa (the paper's lambda_r): the weighted least-squares slope without intercept of ``y``
    (truth - b; units x genes) on ``x`` (the plate-aware drug effect), truncated to [0, 1]; ``w``: the units' weights.
    This is origin resolution's estimator (``net_cost.calibrate_design``, which reads x as A - b) with the sums taken
    in the order of the paper's tables; the two can differ in the last bit (they do on fold 3)."""
    return float(np.clip(np.sum(w[:, None] * x * y) / np.sum(w[:, None] * x * x), 0, 1))


@cache
def fold_data(fold):
    """One fold's units and the quantities the tables read, by key:

    * ``u``: the held-out units, with ``truth``, ``context``, ``p_code`` and ``weight``;
    * per held-out unit (rows in the order of ``u``): ``line``, ``plate``, the drug ``code``, its context's input-set
      mean ``b`` and pool mean ``C``, and its anchor ``A0`` = b + r;
    * ``r``: the training lines' drug average (drugs x genes) as the models received it, stored in single precision;
      ``r_double``: the same average in double precision (origin resolution's ``tahoe.drug_effects`` on the 28
      training lines), from which the readers' offsets are taken; ``r_plate_unit``: per held-out unit, the drug
      average on its plate;
    * ``kappa``: the design-aware amount (``design_kappa``) on the calibration lines, with the nested chain's
      plate-aware drug effects; no model and no held-out truth enter. Origin resolution's +Var cases use their own
      kappas (``calibration.case``), read with each model's saved anchor in place of b + r: AMOUNTS.json's kappas
      differ from this one by at most 2e-8 (folds 2-3; on fold 1 all are 1);
    * ``names``: each drug code's label; ``line_order``: the fold's line order, of which the first ``tahoe.NESTED``
      lines are the nested fit and the rest the calibration lines;
    * ``train``: the training units ``u``, and per unit ``line``, ``plate``, ``b``, ``C`` and ``A0`` as above;
    * ``bags``: side ('train', 'heldout') -> context -> (input-set mean, pool mean); read with ``input_means`` and
      ``pool_means``."""
    f = tahoe.files(fold)
    u = tahoe.units(fold, 'outer')
    bags = bag_means(f['inputs'])
    b, C = bag_of(bags['heldout'], u)
    with np.load(f['constants']) as c:
        r = c['r_input'].astype(np.float64)
    with np.load(f['train_targets']) as z:
        names = np.array([drug_label(x) for x in z['p_ids'].astype(str)])
    line_order = tahoe.line_order(fold)
    # drug effects of the 28 training lines (full chain) and of the 21 nested-fit lines (for kappa)
    r_double, r_plate_full = tahoe.chain_effects(fold, False)
    _, r_plate_nested = tahoe.chain_effects(fold, True)
    line, plate, code = lines(u['context']), plates(u['context']), u['p_code']
    train = tahoe.units(fold, 'train')
    tline, tplate = lines(train['context']), plates(train['context'])
    tb, tC = bag_of(bags['train'], train)
    cal = np.isin(tline, line_order[tahoe.NESTED:])
    x = np.stack([r_plate_nested[p][q] for p, q in zip(tplate[cal], train['p_code'][cal])])
    kappa = design_kappa(x, train['truth'][cal] - tb[cal], train['weight'][cal])
    r_plate = np.stack([r_plate_full[p][q] for p, q in zip(plate, code)])
    return {'u': u, 'line': line, 'plate': plate, 'code': code, 'b': b, 'C': C, 'A0': b + r[code], 'r': r,
            'r_double': r_double, 'r_plate_unit': r_plate, 'kappa': kappa, 'names': names, 'line_order': line_order,
            'train': {'u': train, 'line': tline, 'plate': tplate, 'b': tb, 'C': tC, 'A0': tb + r[train['p_code']]},
            'bags': bags}


class Design:
    """The error readings on one set of units (a fold's held-out units, or a subset with its own weights)."""

    def __init__(self, u):
        n = len(u['truth'])
        self.u = u
        self.w = u['weight'] / u['weight'].sum()
        line, plate = lines(u['context']), plates(u['context'])
        self.line_names = np.array(sorted(set(line)))
        self.line_codes = np.searchsorted(self.line_names, line)
        drug_plate = pd.factorize(pd.Series(plate) + '#' + pd.Series(u['p_code']).astype(str))[0]

        def ind(g):
            return sp.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, g.max() + 1))

        self.Zc = ind(pd.factorize(u['context'])[0])
        self.Z = sp.hstack([self.Zc, ind(drug_plate)]).tocsr()   # context + drug x plate
        # a pseudo-inverse: the context and the drug x plate indicators are collinear (on each plate both sum to one)
        self.Gi = np.linalg.pinv((self.Z.T @ sp.diags(self.w) @ self.Z).toarray(), rcond=1e-10)

    def project_ctx(self, X):
        """Each unit's context mean of ``X`` over the context's drugs (the common part)."""
        s = self.Zc.T @ (self.w[:, None] * X)
        return self.Zc @ (s / np.asarray(self.Zc.T @ self.w)[:, None])

    def energy(self, X, per_line=False):
        """Weighted mean over units of the mean over genes of X^2 (the score's scale); optionally per held-out line
        (contributions that sum to the total)."""
        e = self.w * np.mean(X * X, axis=1)
        return np.bincount(self.line_codes, weights=e, minlength=len(self.line_names)) if per_line else float(e.sum())

    def interaction_of(self, X):
        """The line-specific (interaction) part of X: its contrast to the context means minus the contrast's weighted
        least-squares fit by context + drug x plate."""
        X = X - self.project_ctx(X)
        return X - self.Z @ (self.Gi @ (self.Z.T @ (self.w[:, None] * X)))


@cache
def design(fold):
    """The error readings on the fold's held-out units."""
    return Design(fold_data(fold)['u'])


def supported(train_codes, train_lines, calibration_lines, codes):
    """Which units (drug codes ``codes``) have their drug among the training units of the calibration lines."""
    return np.isin(codes, np.unique(train_codes[np.isin(train_lines, calibration_lines)]))


@cache
def calibration_units(fold):
    """The fold's calibration-supported units: which of ``fold_data(fold)``'s units they are, the units (weights
    recomputed by the datasets' rule) and their error readings (``Design``)."""
    d = fold_data(fold)
    tr = d['train']
    keep = supported(tr['u']['p_code'], tr['line'], d['line_order'][tahoe.NESTED:], d['u']['p_code'])
    u = retained(d['u'], keep)
    return keep, u, Design(u)


def ladder_file(fold, model, seed, step=None):
    """The file of a model's saved Mean-level predictions on the ladder, at its fixed training length or, for the fold-1
    runs read at other lengths, after ``step`` updates."""
    return LADDER / fold / f"{model}_{seed}_L1{'' if step is None else f'_step{step}'}.npz"


def ladder(fold, model, seed):
    """A model's saved Mean-level held-out prediction (units of ``fold_data(fold)``)."""
    return grids.load(ladder_file(fold, model, seed), 'outer', fold_data(fold)['u'])


def as_written(table, **read):
    """The table as read back (``pd.read_csv(..., **read)``) from its CSV written with ``to_csv(**WRITE)``."""
    return pd.read_csv(io.StringIO(table.to_csv(**WRITE)), **read)
