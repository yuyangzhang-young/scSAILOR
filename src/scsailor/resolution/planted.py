"""Planted changes: origin resolution read back on changes whose location, slope and return are set by construction.

A planted change is added to a model's predictions on calibration and held-out lines alike. Its direction comes from
the residual, the target minus the original model calibrated toward the design-aware baseline (``net_cost``), so a
planted return is a gain beyond that calibration. Inside one part of the error, the common or the contrast component
(``scoring``), the change has a part along the residual and a part along noise orthogonal to it; with slope s on the
residual and return G it has energy G / s^2, both in % of the original model's total MSE. A null has slope 0 and only
noise.

    slopes    0.1, 0.2, 0.33, 0.5, 0.75
    returns   0.1, 0.3, 1, 3
    nulls     energy 3, 10, 30

The read-back runs origin resolution as on a measured change: the kept fraction and the calibrations of ``net_cost``,
the change's slopes on the original model's need, its cost at full strength and the part that holds it, and the gaps
of every comparison. The right answers are known: the cost sits in the planted part, the slopes return the planted
slope, a null is refused and a large return is recommended.

A *side* is one of ``net_cost``'s calibration or held-out sides; planting fills in its ``R``.
"""
import numpy as np

from . import menu, net_cost
from .scoring import project, scores
from .slopes import inner

__all__ = ["SLOPES", "RETURNS", "NULL_ENERGY", "PARTS", "ENTROPY", "grid", "split", "plant", "planted_side", "read", "read_back"]

SLOPES = (.1, .2, .33, .5, .75)
RETURNS = (.1, .3, 1., 3.)     # % of the original model's total MSE
NULL_ENERGY = (3., 10., 30.)   # % of the original model's total MSE
PARTS = ('common', 'contrast')
ENTROPY = 20261102             # the fixed entropy of every planted change's random stream


def grid():
    """(part, slope, return, energy) of every planted change, in each part the signal grid and then the nulls."""
    for part in PARTS:
        for s in SLOPES:
            for g in RETURNS:
                yield part, s, g, g / s ** 2
        for e in NULL_ENERGY:
            yield part, 0., 0., e


def split(x, u, part):
    """The common component of x (each context's weighted mean over its perturbations) or the contrast (the rest)."""
    c = project(x, u['context'], u['weight'])
    return c if part == 'common' else x - c


def plant(sd, seed, amounts, part, s, energy_pts, rng):
    """The seed's original model plus a change inside one part of the error: slope s on the residual, energy
    ``energy_pts`` (% of the original's total MSE), the rest of it noise orthogonal to the residual. None when slope s
    asks for more energy along the residual than the change has."""
    u, p, w = sd['u'], sd['p'][seed], sd['u']['weight']
    calibrated = menu.two_part(net_cost.design_baseline(sd, amounts['kappa']), p, *amounts['seeds'][str(seed)]['original_design'], u)
    e = split(u['truth'] - calibrated, u, part)
    noise = split(rng.standard_normal(e.shape), u, part)
    ee = inner(e, e, w)
    noise = noise - inner(noise, e, w) / ee * e
    noise = noise / np.sqrt(inner(noise, noise, w))
    energy = energy_pts / 100 * net_cost.total(p, u)
    along = s * energy / np.sqrt(ee)
    if along ** 2 > energy:
        return None
    return p + along * e / np.sqrt(ee) + np.sqrt(energy - along ** 2) * noise


def planted_side(sd, stream, amounts, part, s, energy_pts, side):
    """The side with every seed's planted change as ``R``, or None when one cannot be planted. ``stream`` (one integer
    per model and fold) and ``side`` (0 for calibration, 1 for held-out lines) give each planted change its own random
    stream."""
    out = dict(sd)
    out['R'] = {}
    for seed in sd['p']:
        rng = np.random.default_rng([ENTROPY, stream, seed, PARTS.index(part), int(100 * s), int(10 * energy_pts), side])
        out['R'][seed] = plant(sd, seed, amounts, part, s, energy_pts, rng)
        if out['R'][seed] is None:
            return None
    return out


def read(sd, amounts, kept):
    """On one side, in % of the original model's total MSE and as means over seeds: the change's cost at full strength
    and its common and contrast parts; the change's term, the original model calibrated toward the design-aware baseline
    minus the kept change calibrated the same way; and the second training run's term, the same original minus the
    second training run calibrated the same way."""
    u = sd['u']
    T = net_cost.design_baseline(sd, amounts['kappa'])
    rows = []
    for s in sd['p']:
        p, R, v, k = sd['p'][s], sd['R'][s], amounts['seeds'][str(s)], kept[str(s)]
        base = scores(p, u)
        end = scores(R, u)
        X = p + k['alpha'] * (R - p)
        original = net_cost.total(menu.two_part(T, p, *v['original_design'], u), u)
        action = net_cost.total(menu.two_part(T, X, *k['action_design'], u), u)
        seed = net_cost.total(menu.two_part(T, net_cost.extra_seed(sd, s), *v['second_design'], u), u)
        pts = 100 / base['total_mse']
        rows.append({'cost': pts * (end['total_mse'] - base['total_mse']), 'cost_common': pts * (end['common_error_mse'] - base['common_error_mse']),
                     'cost_contrast': pts * (end['difference_error_mse'] - base['difference_error_mse']),
                     'change_term': pts * (original - action), 'extra_seed_term': pts * (original - seed)})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def read_back(cal, held, amounts, stream, part, s, energy_pts):
    """Origin resolution on one planted change, with the model's amounts (``net_cost.calibrate_original``): None when it
    cannot be planted, else the readings and the gaps on calibration lines (``net_cost.cumulative_gaps``). Readings: the
    kept fraction, mean and smallest over seeds; the change's total, common and contrast slopes on calibration and
    held-out lines; and ``read`` on both."""
    c = planted_side(cal, stream, amounts, part, s, energy_pts, 0)
    h = planted_side(held, stream, amounts, part, s, energy_pts, 1)
    if c is None or h is None:
        return None
    kept = net_cost.calibrate_change(c, amounts)
    alphas = [kept[str(seed)]['alpha'] for seed in c['p']]
    s_cal, s_held = net_cost.change_slopes(c)[0], net_cost.change_slopes(h)[0]
    readings = {'alpha_mean': float(np.mean(alphas)), 'alpha_min': min(alphas),
                's_cal': s_cal['total'], 's_cal_common': s_cal['common'], 's_cal_contrast': s_cal['differential'],
                's_held': s_held['total'], 's_held_common': s_held['common'], 's_held_contrast': s_held['differential'],
                **{f'cal_{k}': v for k, v in read(c, amounts, kept).items()}, **{f'held_{k}': v for k, v in read(h, amounts, kept).items()}}
    return readings, net_cost.cumulative_gaps(c, amounts, kept)
