"""The natural task's risk on the clipped scale, and its additive parts.

A *unit* is one context and one perturbation. A context is one cell line on one plate (Tahoe-100M), one cell
identity in one background (GSE306429) or one cell line (Replogle–Nadig); its key is ``"<line or identity>|..."``.
A units dictionary holds, in one row order:

- ``truth``: the scored target per unit (units x genes);
- ``context``: the context keys;
- ``p_code``: the perturbation codes;
- ``weight``: the scoring weights.

The risk (``scsailor.natural.score.risk``) splits a prediction's squared error into the *common* part, each context's
weighted mean error over its perturbations (the context-level part), and the *differential* part, the rest (the
perturbation-difference part). Predictions are clipped at zero before they are scored, as the natural task's scoring
does.
"""
import numpy as np

from ..natural.score import project, risk

__all__ = ["PARTS", "PARTKEY", "project", "risk", "scores", "line_parts", "unit_parts", "gene_contributions", "identities", "retained"]


PARTS = ('total', 'common', 'differential')   # the clipped total and its two additive parts
PARTKEY = {'total': 'total_mse', 'common': 'common_error_mse', 'differential': 'difference_error_mse'}   # their keys in risk


def scores(pred, units, mask=None):
    """The risk of a prediction clipped at zero, on the masked units (all units without a mask)."""
    mask = np.ones(len(units['truth']), dtype=bool) if mask is None else mask
    return risk(np.maximum(pred[mask], 0) - units['truth'][mask], units['context'][mask], units['weight'][mask])


def identities(units):
    """Each unit's cell line or identity: its context key up to the first ``|``."""
    return np.asarray([c.split('|')[0] for c in units['context']])


def line_parts(pred, units):
    """Each line's additive share of the clipped total, and of its common and differential parts."""
    error = np.maximum(pred, 0) - units['truth']
    w = units['weight'] / units['weight'].sum()
    lines = identities(units)
    total = w * np.mean(error ** 2, axis=1)
    common = w * np.mean(project(error, units['context'], units['weight']) ** 2, axis=1)
    out = {'total': total, 'common': common, 'differential': total - common}
    return {part: {line: float(v[lines == line].sum()) for line in np.unique(lines)} for part, v in out.items()}


def unit_parts(pred, units, genes=None):
    """Each unit's additive contribution to the clipped risk, weights normalized over all units. Common is the unit's
    weighted share of its context's squared mean error; differential is its own deviation from that mean. Over any
    union of whole contexts, total = common + differential exactly; for groups that cut contexts (perturbations),
    read the differential contribution. A gene subset divides by all genes, so gene groups add up to the whole."""
    e = np.maximum(pred, 0) - units['truth']
    pe = project(e, units['context'], units['weight'])
    w = units['weight'] / units['weight'].sum()
    if genes is not None:
        e, pe = e[:, genes], pe[:, genes]
    n = units['truth'].shape[1]
    return {'total': w * np.sum(e ** 2, axis=1) / n, 'common': w * np.sum(pe ** 2, axis=1) / n,
            'differential': w * np.sum((e - pe) ** 2, axis=1) / n}


def gene_contributions(pred, units):
    """Per-gene additive contributions to each part of the clipped risk (they sum over genes to the risk)."""
    e = np.maximum(pred, 0) - units['truth']
    pe = project(e, units['context'], units['weight'])
    w = (units['weight'] / units['weight'].sum())[:, None]
    n = e.shape[1]
    return {'total': (w * e ** 2).sum(axis=0) / n, 'common': (w * pe ** 2).sum(axis=0) / n,
            'differential': (w * (e - pe) ** 2).sum(axis=0) / n}


def retained(units, keep):
    """The kept units, with weights recomputed by the datasets' rule: contexts equal, perturbations equal within a
    context, units equal within a context and perturbation (not renormalized)."""
    u = {k: v[keep] for k, v in units.items()}
    w, contexts = np.empty(len(u['context'])), np.unique(u['context'])
    for c in contexts:
        in_c = u['context'] == c
        codes = np.unique(u['p_code'][in_c])
        for p in codes:
            m = in_c & (u['p_code'] == p)
            w[m] = 1 / (len(contexts) * len(codes) * m.sum())
    u['weight'] = w
    return u
