"""Metrics other than squared error, per held-out unit, then per line: the readings of the matched comparisons'
predictions on the metrics the field reports.

A units dictionary ``u`` holds ``truth``, ``weight``, ``context`` and ``p_code`` per held-out unit (as in ``scoring``).
A *reference* ``ref`` (built by a dataset's reproduction code) holds, for the units ``u``: ``C`` the held-out control
vector, ``R`` that vector plus the training lines' mean response, ``ybar`` the scoring-weighted mean truth of each
unit's context, ``d`` the truth minus it, ``gene_weight`` the gene weights of the weighted R², ``index`` and ``n`` the
unit's line and the line count, and, on genetic screens, ``target`` the target gene's column per unit (-1 for none).

Per unit (``unit_values``), predictions clipped at zero: squared error ``mse``; ``weighted_r2`` and
``weighted_r2_contrast``, the adapted weighted R² of the change from the context mean (against the truth's context mean,
or the prediction's own); ``pearson_delta`` and ``spearman_delta``, the correlation of prediction and truth after the
control vector is subtracted; ``direction_top50``, the share of the truth's 50 largest changes that the prediction's 50
largest name with the same sign; ``cosine_pds``, one minus the rank of the true perturbation among its context's
perturbations by cosine similarity, over their count minus one (1: ranked first); ``systema_pearson_delta`` and
``systema_pearson_delta_observed``, the correlation after ``R`` or the context's mean truth is subtracted. On genetic
screens (``target_excluded_values``) the target gene is set to the control value in prediction and truth for every
metric but squared error and the weighted R², whose gene weights leave it out. ``per_line`` sums each metric over the
units of each line, by weight, skipping undefined values; ``point`` divides.
"""
import numpy as np
from scipy.stats import rankdata

from ..natural.score import row_correlation

__all__ = ["context_mean", "effect_weights", "r2_units", "top50", "pds_rows", "unit_values", "target_excluded_values",
           "per_line", "point"]


def context_mean(x, uw, ctx):
    """Scoring-weighted mean profile of each context, broadcast back to its units; the weights are normalised first,
    so a context with one unit returns that unit's profile exactly."""
    out = np.empty_like(x)
    for c in np.unique(ctx):
        idx = np.flatnonzero(ctx == c)
        out[idx] = (uw[idx] / uw[idx].sum()) @ x[idx]
    return out


def effect_weights(d, target=None):
    """|d| min-max scaled over genes, squared, normalised (uniform when all zero); the target gene, where given, gets
    weight 0 and is left out of the range."""
    a = np.abs(d)
    keep = np.ones_like(a, bool)
    if target is not None:
        rows = np.flatnonzero(target >= 0)
        keep[rows, target[rows]] = False
    lo = np.where(keep, a, np.inf).min(1, keepdims=True)
    hi = np.where(keep, a, -np.inf).max(1, keepdims=True)
    span = hi - lo
    m = np.where(span > 0, (a - lo) / np.where(span > 0, span, 1), 0.0) * keep
    w = m ** 2
    tot = w.sum(1, keepdims=True)
    return np.where(tot > 0, w / np.where(tot > 0, tot, 1), keep / keep.sum(1, keepdims=True))


def r2_units(d, dh, w):
    """1 - sum w (d - dh)^2 / sum w (d - dbar)^2 per unit; undefined where the denominator is zero."""
    dbar = (w * d).sum(1, keepdims=True)
    num = (w * (d - dh) ** 2).sum(1)
    den = (w * (d - dbar) ** 2).sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r2 = 1.0 - num / den
    r2[~(den > 0)] = np.nan
    return r2


def top50(a, b):
    """Per row: the share of b's 50 largest nonzero changes that a's 50 largest name, with the same sign."""
    out = np.full(len(a), np.nan)
    for i, (aa, bb) in enumerate(zip(a, b)):
        t = np.argsort(-np.abs(bb), kind='stable')[:50]
        t = t[bb[t] != 0]
        guess = np.argsort(-np.abs(aa), kind='stable')[:50]
        if len(t):
            out[i] = np.mean(np.isin(t, guess) & (np.sign(aa[t]) == np.sign(bb[t])))
    return out


def pds_rows(a, b, contexts, codes, w):
    """Per-row cosine PDS: 1 - the rank of the true perturbation among its context's perturbations, by the cosine
    similarity of the prediction to each perturbation's weighted mean truth, over the count minus one."""
    pds = np.full(len(a), np.nan)
    for context in np.unique(contexts):
        mask = np.flatnonzero(contexts == context)
        pc = np.unique(codes[mask])
        if len(pc) < 2:
            continue
        target = np.stack([np.average(b[mask[codes[mask] == p]], axis=0, weights=w[mask[codes[mask] == p]]) for p in pc])
        norms = np.linalg.norm(target, axis=1)
        for p in pc:
            rows = mask[codes[mask] == p]
            prediction = np.average(a[rows], axis=0, weights=w[rows])
            den = np.linalg.norm(prediction) * norms
            sim = np.divide(target @ prediction, den, out=np.zeros(len(pc)), where=den > 0)
            correct = sim[np.flatnonzero(pc == p)[0]]
            rank = np.sum(sim > correct) + (np.sum(sim == correct) - 1) / 2
            pds[rows] = 1 - rank / (len(pc) - 1)
    return pds


def unit_values(pred, u, ref):
    """Every metric above per unit, the prediction clipped at zero."""
    x, t, w, ctx = np.maximum(pred, 0), u['truth'], u['weight'], u['context']
    a, b = x - ref['C'], t - ref['C']
    return {'mse': np.mean((x - t) ** 2, axis=1),
            'weighted_r2': r2_units(ref['d'], x - ref['ybar'], ref['gene_weight']),
            'weighted_r2_contrast': r2_units(ref['d'], x - context_mean(x, w, ctx), ref['gene_weight']),
            'pearson_delta': row_correlation(a, b),
            'spearman_delta': row_correlation(rankdata(a, axis=1), rankdata(b, axis=1)),
            'direction_top50': top50(a, b),
            'cosine_pds': pds_rows(a, b, ctx, u['p_code'], w),
            'systema_pearson_delta': row_correlation(x - ref['R'], t - ref['R']),
            'systema_pearson_delta_observed': row_correlation(x - ref['ybar'], t - ref['ybar'])}


def target_excluded_values(pred, u, ref):
    """``unit_values`` with each unit's target gene set to the control value in prediction and truth; squared error and
    the weighted R² on the clipped prediction as it is (the R²'s gene weights leave the target out)."""
    rows = np.flatnonzero(ref['target'] >= 0)
    cols = ref['target'][rows]
    x, t = np.maximum(pred, 0).copy(), u['truth'].copy()
    mse = np.mean((x - t) ** 2, axis=1)
    x[rows, cols] = ref['C'][rows, cols]
    t[rows, cols] = ref['C'][rows, cols]
    out = unit_values(x, {**u, 'truth': t}, ref)
    out['mse'] = mse
    clipped = np.maximum(pred, 0)
    out['weighted_r2'] = r2_units(ref['d'], clipped - ref['ybar'], ref['gene_weight'])
    out['weighted_r2_contrast'] = r2_units(ref['d'], clipped - context_mean(clipped, u['weight'], u['context']), ref['gene_weight'])
    return out


def per_line(values, u, ref):
    """metric -> (weighted sum per line, weight per line), undefined unit values left out."""
    out = {}
    for metric, v in values.items():
        good = np.isfinite(v)
        w = np.where(good, u['weight'], 0.)
        out[metric] = (np.bincount(ref['index'], w * np.where(good, v, 0.), ref['n']), np.bincount(ref['index'], w, ref['n']))
    return out


def point(pair):
    """One metric over lines: its weighted sum over lines divided by the weight (a ``per_line`` pair)."""
    return float(pair[0].sum() / pair[1].sum())
