"""Readers of a held-out context's control cells, and the arms that replace a model's context-level correction.

Terms as in ``data``: a context is a line on one plate, ``b`` the mean of its input set, ``r`` the training lines' drug
average and ``A0 = b + r`` the anchor.

**Offsets.** A training context's offset (the paper's source context response), per gene, is the weighted mean over
its training units of the target minus the context's control-pool mean minus the drug average (the weighted mean of
that change over every training unit of the drug; ``data.fold_data``'s ``r_double``): the context's shift of every
drug's response beyond the drug average.

**Readers.** Fitted on the 28 training lines' offsets, they predict a held-out context's offset from its own input set:

* gene-wise readers: for each gene, ridge regression of its offset on the gene's own input-set mean;
* the growth reader, on the growth genes (``programs``): kernel ridge (linear kernel) on the three growth programs'
  mean expression in the input set and a gut-lineage indicator (the line's organ in Tahoe-100M's cell-line metadata
  is Bowel or Esophagus/Stomach). The storage regression (``storage``) uses the readers without the gut-lineage
  indicator.

The growth reader's features are centred per plate on the training contexts, then divided by their SD over the
training contexts (``centre_scale``); the gene-wise readers divide each gene's input-set mean, centred per plate, by
the SD over the training contexts of the uncentred means (plus 1e-9). Offsets are centred per plate. Each penalty is
chosen from ``PENALTIES`` by leaving one training line out (both plates together): both readers recompute the plate
means of the offsets without it, and the gene-wise readers also those of their feature. The gene-wise readers choose
one penalty per gene. The readers' offset of a held-out context is the gene-wise readers' on every gene, with the
growth reader's on the growth genes.

**Arms.** A model's context-level correction is ``proj_ctx(P - A0)``, the weighted mean over a context's drugs of its
prediction minus the anchor. Each arm replaces it before clipping (``fold_arms``):

* removal: ``P - proj_ctx(P - A0)``;
* constant: plus the training contexts' mean offset on the unit's plate (the readers' intercept);
* readers: plus the readers' offset.

Given a calibrated prediction (origin resolution's calibration toward the design-aware baseline, on the
calibration-supported units), 'calibrated' is that prediction and 'calibrated + readers' the readers' replacement on
it. The arms change only a prediction's context-level part: its differences between a context's drugs are kept.
"""
from functools import cache

import numpy as np
import pandas as pd

from ..origin_resolution import tahoe
from . import data, programs

__all__ = ["PENALTIES", "GUT", "ARMS", "CALIBRATED_ARMS", "centre", "offsets", "training_offsets", "constant",
           "centre_scale", "gut_lines", "features", "kernel_ridge", "KernelReader", "GeneReaders", "readers_offsets",
           "fit", "arm_predictions", "fold_arms"]

PENALTIES = np.logspace(-2, 5, 15)
GUT = ('Bowel', 'Esophagus/Stomach')   # the organs of gut lineage in Tahoe-100M's cell-line metadata
ARMS = ('original', 'removal', 'constant', 'readers')
CALIBRATED_ARMS = ('calibrated', 'calibrated + readers')   # the arms given a calibrated prediction


def centre(M, plate):
    """The rows of ``M`` minus their plate's mean (``plate``: each row's plate)."""
    M = M.copy()
    for p in np.unique(plate):
        M[plate == p] -= M[plate == p].mean(axis=0)
    return M


def offsets(change, context, code, weight, average):
    """The training contexts (sorted) and their offsets: per context, the weighted mean over its units of ``change``
    (target minus control-pool mean) minus the drug average (``average``: drugs x genes; ``code``: each unit's drug)."""
    rest = change - average[code]
    contexts = np.unique(context)
    # column-major, so that each gene's units are summed together (pairwise): the floating-point order of the paper's
    # tables
    return contexts, np.stack([np.average(np.asfortranarray(rest[context == c]), axis=0, weights=weight[context == c])
                               for c in contexts])


@cache
def training_offsets(fold):
    """The fold's training contexts (sorted) and their offsets on every gene."""
    d = data.fold_data(fold)
    tr = d['train']
    u = tr['u']
    return offsets(u['truth'] - tr['C'], u['context'], u['p_code'], u['weight'], d['r_double'])


def constant(Y, contexts, plate):
    """Per unit (its plate in ``plate``): the mean offset ``Y`` of the training ``contexts`` on that plate."""
    on = data.plates(contexts)
    mean = {p: np.mean(Y[on == p], axis=0) for p in np.unique(plate)}
    return np.stack([mean[p] for p in plate])


def centre_scale(X, plate, train):
    """X centred per plate on the training rows, then divided by its SD over the training rows."""
    X = X.copy()
    for p in np.unique(plate):
        m = plate == p
        X[m] -= X[m & train].mean(axis=0)
    return X / X[train].std(axis=0)


def gut_lines(metadata):
    """The lines (Cellosaurus identifiers) of gut lineage in a cell-line metadata table."""
    organ = metadata.groupby('Cell_ID_Cellosaur').Organ.first()
    return frozenset(organ[organ.isin(GUT)].index)


@cache
def features(fold):
    """The growth reader's features of every context's input set: the contexts (training contexts first, each side
    sorted), the three growth programs' mean expression, and the same with the gut-lineage indicator as a fourth
    column; each centred and scaled by ``centre_scale``."""
    contexts, train, means = data.all_contexts(data.fold_data(fold)['bags'])
    masks, _ = programs.growth(fold)
    plate = data.plates(contexts)
    X = centre_scale(np.array([[row[m].mean() for m in masks] for row in means]), plate, train)
    gut = gut_lines(pd.read_parquet(tahoe.ORGANS))
    g = centre_scale(np.array([float(line in gut) for line in data.lines(contexts)]), plate, train)
    return contexts, X, np.column_stack([X, g])


def kernel_ridge(K, rows, Y, test, penalties):
    """Kernel ridge without intercept from the kernel rows ``rows`` (targets ``Y``) to the kernel rows ``test``, one
    prediction per penalty."""
    lam, U = np.linalg.eigh(K[np.ix_(rows, rows)])
    lam = np.maximum(lam, 0)   # rounding can leave tiny negative eigenvalues
    UtY = U.T @ Y
    KU = K[np.ix_(test, rows)] @ U
    return np.stack([KU @ (UtY / (lam + a)[:, None]) for a in penalties])


class KernelReader:
    """Kernel ridge from a context's features to its offsets: each plate's mean offset plus the ridge fit of the
    deviations, the penalty chosen by leaving one training line out (both plates), the plate means of the offsets
    recomputed without it."""

    def __init__(self, K, contexts, train, Y):
        """K: the kernel over ``contexts`` (context keys); train: the rows of the training contexts, in the order of
        their offsets ``Y``."""
        self.K, self.plate = K, data.plates(contexts)
        self.train = np.asarray(train)
        self.Y = np.asarray(Y, dtype=np.float64)
        line = data.lines(contexts)[self.train]
        errors = np.zeros(len(PENALTIES))
        for held in np.unique(line):
            out = line == held
            pred = self._fit_predict(~out, self.train[out], PENALTIES)
            errors += np.sum((pred - self.Y[out][None]) ** 2, axis=(1, 2))
        self.penalty = float(PENALTIES[int(np.argmin(errors))])

    def _fit_predict(self, rows, test, penalties):
        """Predictions for the kernel rows ``test`` from the training rows ``rows``, one per penalty."""
        Y, plate = self.Y[rows], self.plate[self.train[rows]]
        means = {p: Y[plate == p].mean(axis=0) for p in np.unique(plate)}
        Yc = Y - np.stack([means[p] for p in plate])
        base = np.stack([means[p] for p in self.plate[test]])
        return base[None] + kernel_ridge(self.K, self.train[rows], Yc, test, penalties)

    def predict(self, test):
        """The offsets of the kernel rows ``test``, from every training context."""
        return self._fit_predict(np.ones(len(self.Y), bool), np.asarray(test), [self.penalty])[0]


class GeneReaders:
    """Per gene, ridge regression of the offset on the gene's own input-set mean (centred per plate on the training
    contexts and divided by the SD of the uncentred means over them, plus 1e-9), the penalty chosen per gene by leaving
    one training line out (both plates), the plate means of the offsets and of the feature recomputed without it."""

    def __init__(self, X, contexts, Y):
        """X: the training contexts' input-set means (contexts x genes); Y: their offsets."""
        self.plate = data.plates(contexts)
        line = data.lines(contexts)
        self.mu = {p: X[self.plate == p].mean(axis=0) for p in np.unique(self.plate)}
        self.sd = X.std(axis=0) + 1e-9
        self.X = (X - np.stack([self.mu[p] for p in self.plate])) / self.sd
        self.Y = Y
        errors = np.zeros((len(PENALTIES), Y.shape[1]))
        for held in np.unique(line):
            out = line == held
            pred = self._fit_predict(~out, self.X[out], self.plate[out])
            errors += ((pred - Y[out][None]) ** 2).sum(axis=1)
        self.penalty = PENALTIES[np.argmin(errors, axis=0)]

    def _fit_predict(self, rows, X, plate, penalty=None):
        """Predictions for standardized means ``X`` on ``plate`` from the training rows ``rows``: one per penalty of
        ``PENALTIES``, or one with each gene's ``penalty``."""
        Xt, Y, on = self.X[rows], self.Y[rows], self.plate[rows]
        means = {p: Y[on == p].mean(axis=0) for p in np.unique(on)}
        xm = {p: Xt[on == p].mean(axis=0) for p in np.unique(on)}
        Xc = Xt - np.stack([xm[p] for p in on])
        Yc = Y - np.stack([means[p] for p in on])
        sxy, sxx = (Xc * Yc).sum(axis=0), (Xc * Xc).sum(axis=0)
        penalties = PENALTIES[:, None] if penalty is None else penalty[None]
        slope = sxy[None] / (sxx[None] + penalties)
        base = np.stack([means[p] for p in plate])
        xc = X - np.stack([xm[p] for p in plate])
        return base[None] + slope[:, None, :] * xc[None]

    def predict(self, X, contexts):
        """The offsets of contexts with input-set means ``X``, from every training context."""
        plate = data.plates(contexts)
        Z = (X - np.stack([self.mu[p] for p in plate])) / self.sd
        return self._fit_predict(np.ones(len(self.Y), bool), Z, plate, self.penalty)[0]


def readers_offsets(Y, contexts, means, K, kernel_contexts, G, test, test_means):
    """The readers' offsets of the contexts ``test`` (input-set means ``test_means``), fitted on the contexts
    ``contexts`` (offsets ``Y``, input-set means ``means``): the gene-wise readers on every gene, the growth reader
    (kernel ``K``, whose rows are ``kernel_contexts``) on the growth genes ``G``."""
    out = GeneReaders(means, contexts, Y).predict(test_means, test)
    row = {c: i for i, c in enumerate(kernel_contexts)}
    growth = KernelReader(K, kernel_contexts, [row[c] for c in contexts], Y[:, G])
    out[:, G] = growth.predict([row[c] for c in test])
    return out


@cache
def fit(fold):
    """The readers fitted on the fold's training lines. Per held-out unit of ``data.fold_data(fold)``, the offset an arm
    puts in place of the context-level correction: 'constant' and 'readers' (units x genes)."""
    d = data.fold_data(fold)
    bags = d['bags']
    contexts, Y = training_offsets(fold)
    held = sorted(bags['heldout'])
    _, G = programs.growth(fold)
    kernel_contexts, _, F = features(fold)
    offset = readers_offsets(Y=Y, contexts=contexts, means=data.input_means(bags['train'], contexts), K=F @ F.T,
                             kernel_contexts=kernel_contexts, G=G, test=held,
                             test_means=data.input_means(bags['heldout'], held))
    row = {c: i for i, c in enumerate(held)}
    return {'constant': constant(Y, contexts, d['plate']), 'readers': offset[[row[c] for c in d['u']['context']]]}


def arm_predictions(P, A, offsets, D, calibrated=None):
    """Each arm's prediction (``ARMS``: name -> units x genes) from a prediction ``P``, the anchor ``A`` and the offsets
    'constant' and 'readers' on the units of ``D``; with ``calibrated``, also ``CALIBRATED_ARMS``."""
    def replaced(X, offset=0.):
        return X - D.project_ctx(X - A) + offset

    out = {'original': P, 'removal': replaced(P), 'constant': replaced(P, offsets['constant']),
           'readers': replaced(P, offsets['readers'])}
    if calibrated is not None:
        out |= {'calibrated': calibrated, 'calibrated + readers': replaced(calibrated, offsets['readers'])}
    return out


def fold_arms(fold, prediction, calibrated=None):
    """``arm_predictions`` of a model's prediction on every held-out unit of the fold or, given a calibrated prediction,
    on the calibration-supported units (``data.calibration_units``), with the fold's anchor and readers (``fit``)."""
    if calibrated is None:
        keep, D = slice(None), data.design(fold)
    else:
        keep, _, D = data.calibration_units(fold)
    d = data.fold_data(fold)
    return arm_predictions(prediction, d['A0'][keep], {k: v[keep] for k, v in fit(fold).items()}, D, calibrated)
