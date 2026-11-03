"""Attribution's model contrasts: the parent decomposed into removable parts, their calibrated amounts, the
recommendation condition, Shapley shares and the alignment of a correction.

The parent is written p = b + r + d: b the anchor's base (the input bag's mean, "no change"), r = a - b the anchor's
transferred response, d = p - a the model's deviation from the anchor. Three removals, each scaled by an amount chosen
on the calibration units:

    M5  the cell-line-level part P d          (amount lam: the cheapest fix)
    M7  the perturbation part (I - P) d       (amount mu: the drug shrink)
    N   the transferred response r            (amount kappa: the response shrink)

The cheap alternatives are the cheapest fix, the uniform shrink toward the anchor and the whole prediction shrunk
toward no change. An action is recommended when, in every seed, it is not the parent and its calibration total is
below the parent's and below every cheap alternative's by more than the margin, in points of the parent's total.
"""
import itertools
import math

import numpy as np

from .comparison import MARGIN
from .policies import G, lowest, predict
from .scoring import line_parts, project, risk, scores

__all__ = ["PLAYERS", "ALTERNATIVES", "compose", "drug_shrink", "nochange_shrink", "removals", "calibrate",
           "calibration_entry", "branch", "runs", "shapley", "line_one", "alignment"]

PLAYERS = ('M5', 'M7', 'N')
ALTERNATIVES = ('cheapest_fix', 'global_parent_shrink', 'nochange_shrink')


def compose(p, a, b, units, lam=1., mu=1., kappa=1.):
    """b + kappa r + lam P d + mu (I - P) d; all ones return p."""
    d = p - a
    pd = project(d, units['context'], units['weight'])
    return b + kappa * (a - b) + lam * pd + mu * (d - pd)


def drug_shrink(p, a, mu, units):
    """M7's removal alone: p - (1 - mu)(I - P)(p - a)."""
    d = p - a
    return p - (1 - mu) * (d - project(d, units['context'], units['weight']))


def nochange_shrink(p, b, kappa):
    """The whole prediction shrunk toward no change, b + kappa (p - b)."""
    return b + kappa * (p - b)


def removals(p, a, b, units, amounts):
    """Every subset of the three removals at their calibrated amounts (keys: frozensets of PLAYERS), plus the cheap
    alternatives 'uniform' and 'nochange_alt'."""
    out = {}
    for k in range(len(PLAYERS) + 1):
        for s in itertools.combinations(PLAYERS, k):
            out[frozenset(s)] = compose(p, a, b, units, lam=amounts['cheapest_fix'] if 'M5' in s else 1.,
                                        mu=amounts['drug_shrink'] if 'M7' in s else 1.,
                                        kappa=amounts['response_shrink'] if 'N' in s else 1.)
    out['uniform'] = predict('global_parent_shrink', amounts['global_parent_shrink'], p, None, a, units)
    out['nochange_alt'] = nochange_shrink(p, b, amounts['nochange_shrink'])
    return out


def calibrate(fn, units):
    """fn(amount) -> prediction; the amount on G with the lowest clipped total, ties to less change (amount 1 first)."""
    return lowest(lambda x: scores(fn(x), units)['total_mse'], reversed(G))


def calibration_entry(p, a, b, units):
    """Calibration amount and clipped total of the parent and of every removal and cheap alternative."""
    out = {'parent': (None, scores(p, units)['total_mse'])}
    out['cheapest_fix'] = calibrate(lambda x: predict('cheapest_fix', x, p, None, a, units), units)
    out['global_parent_shrink'] = calibrate(lambda x: predict('global_parent_shrink', x, p, None, a, units), units)
    out['drug_shrink'] = calibrate(lambda x: compose(p, a, b, units, mu=x), units)
    out['response_shrink'] = calibrate(lambda x: compose(p, a, b, units, kappa=x), units)
    out['nochange_shrink'] = calibrate(lambda x: nochange_shrink(p, b, x), units)
    return {k: {'amount': v[0], 'calibration': v[1]} for k, v in out.items()}


def branch(entries, action, action_calibration=None, margin=MARGIN, alternatives=ALTERNATIVES):
    """The recommendation condition, every seed: the action is not the parent, and it is below the parent and below
    every cheap alternative that differs from it by more than the margin, in points of the parent's calibration total.
    action_calibration: seed -> the action's calibration total where the action is not one of the entries."""
    gaps = {}
    for seed, e in entries.items():
        base = e['parent']['calibration']
        mine = e[action]['calibration'] if action in e else action_calibration[seed]
        gaps[seed] = {ref: 100 * (e[ref]['calibration'] - mine) / base for ref in ('parent',) + tuple(alternatives) if ref != action}
    holds = action != 'parent' and all(g > margin for seed in gaps.values() for g in seed.values())
    return {'recommendation': bool(holds), 'gaps_pts': gaps}


def runs(preds, units, scales):
    """One (per-line parts, scale) pair per seed, as comparison.detect and compare read them."""
    return [(line_parts(p, units), s) for p, s in zip(preds, scales)]


def shapley(value, players):
    """Shapley shares of composing removals: value(frozenset) is the error change of removing that set (zero for the
    empty set). Undetected contrasts are left out of `players` and so carry share zero."""
    players = list(players)
    shares = dict.fromkeys(players, 0.)
    for order in itertools.permutations(players):
        done = frozenset()
        for q in order:
            shares[q] += value(done | {q}) - value(done)
            done = done | {q}
    return {q: v / math.factorial(len(players)) for q, v in shares.items()} if players else {}


def line_one(detections, separation):
    """Line 1 of the headline. detections: 'M5'/'M7' -> comparison.detect against the parent; separation:
    detect of M5's removal against M7's (negative: M5 removes more error), with M7's against M5's as 'reverse'."""
    any_case = any(d['level'] == 'case-level' for d in detections.values())
    if any_case:
        signs = '; '.join(f"{r}: {d['case_counts']['lower']} of {d['case_counts']['of']} unit-seeds lower, total {d['total_pts']:+.2f} pts"
                          for r, d in detections.items())
        units = next(iter(detections.values()))['units']
        return f'Case-level only ({units} outer units): {signs}'
    found = [r for r, d in detections.items() if d['detected']]
    if not found:
        return 'No model contrast detected'
    if len(found) == 1:
        return f'Leading model contrast: {found[0]}'
    if separation['detected']:
        return 'Leading model contrast: M5'
    reverse = separation.get('reverse')
    if reverse is not None and reverse['detected']:
        return 'Leading model contrast: M7'
    return 'Leading model contrasts, not separated by the data: {M5, M7}'


def alignment(pred, anchor, units, unmatchable=None):
    """Per part, on the raw scale: the cosine between the predicted correction (pred - anchor) and the true one
    (truth - anchor); the best scale of the predicted correction; its parallel ratio, the size of its component along
    the true correction relative to the true one (above 1: over-sized along the true direction); and its
    misdirected share, 1 - cosine^2. With ``unmatchable`` (per part, the energy no predictor reading the bag can
    match), the ceiling of the cosine and the cosine over it."""
    base = risk(anchor - units['truth'], units['context'], units['weight'])
    model = risk(pred - units['truth'], units['context'], units['weight'])
    corr = risk(pred - anchor, units['context'], units['weight'])
    out = {}
    for part, key in (('common', 'common_error_mse'), ('differential', 'difference_error_mse')):
        e2, p2 = base[key], corr[key]
        dot = (e2 + p2 - model[key]) / 2
        cos = dot / np.sqrt(e2 * p2) if p2 > 0 else float('nan')
        out[part] = {'size_over_anchor_error': p2 / e2, 'cosine': cos, 'best_scale': dot / p2 if p2 > 0 else float('nan'),
                     'parallel_ratio': dot / e2, 'misdirected_share': 1 - cos * cos}
        if unmatchable is not None:
            u = unmatchable[part]
            ceiling = float(np.sqrt(max(0., 1 - u / e2)))
            out[part] |= {'ceiling': ceiling, 'cosine_over_ceiling': cos / ceiling if ceiling > 0 else float('nan'),
                          'parallel_ratio_matchable': dot / (e2 - u) if e2 > u else float('nan')}
    return out
