"""How much of a prediction layer to keep: one slope per layer, read on calibration lines.

A model's prediction is written in three layers around its baseline T (the plate-aware baseline on Tahoe-100M, the
baseline elsewhere): the transferred response T - b, with b the control mean, the context-level correction
P(p - T) and the perturbation-level correction (I - P)(p - T). The slope of a layer is the regression coefficient of
what is needed on the layer, with no intercept, in the risk's weighted inner product. By the algebra of squared error
it is the fraction of the layer that lowers error most, and at full strength a layer lowers error only above one half.

    slope >= 1/2         keep the layer
    1/16 <= slope < 1/2  shrink it to the slope
    0 < slope < 1/16     no usable signal
    slope <= 0           refuse it

A *side* is one chain on one set of units, as a dataset's binding reads it:

- ``u``: the units (``scoring``);
- ``seeds``: seed -> the model's prediction on the units;
- ``T``, ``b``: the baseline and the control mean per unit;
- ``outside``: per unit, the control pool's mean without the input set's cells;
- ``lines``, ``strata``: per unit its line (the calibration unit, and what a partner must differ in) and its design
  stratum (what partners share: a plate, a cell type, a stimulus, or none).

The specificity of the perturbation-level layer compares the slope on a unit's own need with the slope on the needs of
other lines of its stratum at the same perturbation. Calibration reads every side's slopes, and held-out lines only
report the same numbers beside it.
"""
import itertools

import numpy as np

from . import card, menu
from .policies import G
from .scoring import project, scores

__all__ = ["LAYERS", "KEEP", "EMPTY", "S_CUT", "TOLERANCE", "POSITIVE", "pool_without_set", "layers", "inner", "slope", "seed_slopes", "fraction",
           "judge", "assigned", "context_class", "by_line", "specificity", "line_terms", "mock_readings", "ordinary", "with_layer", "gain",
           "seed_difference_slopes", "held_out_columns"]

LAYERS = ('response', 'line', 'perturbation')
KEEP = 1 / 2       # keep at slope >= KEEP
EMPTY = 1 / 16     # no usable signal below it
S_CUT = 1 / 8      # context-specific at S >= S_CUT
TOLERANCE = 1 / 4  # the primary metric's tolerance
POSITIVE = tuple(a for a in G if a > 0)   # the amount grid without zero


def pool_without_set(pool, input_set, pool_cells, set_cells):
    """The control pool's mean without the input set's cells, per unit."""
    return (pool_cells * pool - set_cells * input_set) / (pool_cells - set_cells)


def layers(sd, reference, seed):
    """layer -> (x, y): one seed's prediction layer and the needed correction, both on the raw scale. ``reference`` is
    the control the need is measured from, 'outside' or the input set 'b'; it changes the transferred response's and the
    context-level correction's needs only, because it is constant within a context."""
    u, T, b, base = sd['u'], sd['T'], sd['b'], sd[reference]
    w, context = u['weight'], u['context']
    d = sd['seeds'][seed] - T
    x_line = project(d, context, w)
    need = u['truth'] - T + b - base
    y_line = project(need, context, w)
    return {'response': (T - b, u['truth'] - base), 'line': (x_line, y_line), 'perturbation': (d - x_line, need - y_line)}


def inner(x, y, w):
    return float(np.sum(w * np.mean(x * y, axis=1)))


def slope(x, y, w):
    """The regression coefficient of the need y on the layer x, no intercept, in the risk's weighted inner product."""
    return inner(x, y, w) / inner(x, x, w)


def seed_slopes(sd, reference):
    """layer -> the mean over seeds of the layer's slope."""
    per = [layers(sd, reference, s) for s in sd['seeds']]
    return {name: float(np.mean([slope(*L[name], sd['u']['weight']) for L in per])) for name in LAYERS}


def fraction(s):
    return float(np.clip(s, 0, 1))


def judge(s, keep=KEEP, empty=EMPTY):
    if s <= 0:
        return 'refuse'
    if s < empty:
        return 'no usable signal'
    return 'keep' if s >= keep else 'shrink'


def assigned(s):
    """The fraction the judgment assigns: all, the slope, or none."""
    return {'keep': 1., 'shrink': s}.get(judge(s), 0.)


def context_class(s, S, cut=S_CUT):
    """The perturbation-level judgment on specificity; S is None with a single calibration line."""
    if s < EMPTY:
        return 'empty'
    if S is None:
        return 'not accepted'
    return 'specific' if S >= cut else 'generic'


def by_line(sd, y):
    """(stratum, perturbation) -> line -> the mean need there: contexts of one line first, so a line counts once."""
    groups = {}
    for i, (stratum, q, line) in enumerate(zip(sd['strata'], sd['u']['p_code'], sd['lines'])):
        groups.setdefault((stratum, int(q)), {}).setdefault(line, []).append(i)
    return {k: {line: y[rows].mean(axis=0) for line, rows in v.items()} for k, v in groups.items()}


def specificity(sd, s, partner=None):
    """The perturbation-level layer's specificity, means over seeds. m is the slope on the needs of other lines of the
    same stratum at the same perturbation, averaged over those lines, on the units with such a partner. S = s - m with
    s the layer's slope on all units; S_matched takes s on m's units. ``partner``: the side the partners come from
    (default this side). All None with no partner."""
    seeds = list(sd['seeds'])
    y = layers(sd, 'b', seeds[0])['perturbation'][1]   # the need does not depend on the seed
    source = partner or sd
    needs = by_line(source, layers(source, 'b', seeds[0])['perturbation'][1])
    keep, other = [], []
    for i, (stratum, q, line) in enumerate(zip(sd['strata'], sd['u']['p_code'], sd['lines'])):
        here = needs.get((stratum, int(q)), {})
        others = [v for l, v in here.items() if l != line]
        if others:
            keep.append(i)
            other.append(np.mean(others, axis=0))
    if not keep:
        return {'slope_matched': None, 'm': None, 'S': None, 'S_matched': None}
    y, w, other = y[keep], sd['u']['weight'][keep], np.stack(other)
    xs = [layers(sd, 'b', seed)['perturbation'][0][keep] for seed in seeds]
    matched, m = float(np.mean([slope(x, y, w) for x in xs])), float(np.mean([slope(x, other, w) for x in xs]))
    return {'slope_matched': matched, 'm': m, 'S': s - m, 'S_matched': matched - m}


def line_terms(sd, reference):
    """layer -> line -> (x.y, x.x) per seed: a seed's slope on any set of lines is the ratio of the sums, because the
    projection works per context."""
    lines, w = sd['lines'], sd['u']['weight']
    per = [layers(sd, reference, s) for s in sd['seeds']]
    out = {}
    for name in LAYERS:
        out[name] = {}
        for line in np.unique(lines):
            m = lines == line
            out[name][line] = (np.asarray([inner(L[name][0][m], L[name][1][m], w[m]) for L in per]),
                               np.asarray([inner(L[name][0][m], L[name][0][m], w[m]) for L in per]))
    return out


def mock_readings(terms):
    """Yields (n, hit): every split of the calibration lines into n lines that calibrate and the rest that score."""
    lines = sorted(terms)
    ratio = lambda group: float(np.mean(sum(terms[l][0] for l in group) / sum(terms[l][1] for l in group)))
    for n in range(1, len(lines)):
        for group in itertools.combinations(lines, n):
            rest = [l for l in lines if l not in group]
            yield n, abs(fraction(ratio(group)) - fraction(ratio(rest))) <= TOLERANCE


def ordinary(sd):
    """Ordinary calibration's amounts, means over seeds: (response, line, perturbation), lowest clipped total on the
    amount grid. The transferred response's amount does not depend on the model."""
    u, T, b = sd['u'], sd['T'], sd['b']
    line, perturbation = np.mean([menu.pick_two(T, x, u)[0] for x in sd['seeds'].values()], axis=0)
    response = card.calibrate(lambda y: b + y * (T - b), u)[0]
    return float(response), float(line), float(perturbation)


def with_layer(sd, name, a, seed):
    """One seed's prediction with one layer added at fraction a to its own baseline: no change for the transferred
    response, the baseline for a correction."""
    u, T, b = sd['u'], sd['T'], sd['b']
    if name == 'response':
        return b + a * (T - b)
    d = sd['seeds'][seed] - T
    line = project(d, u['context'], u['weight'])
    return T + a * (line if name == 'line' else d - line)


def gain(sd, name, a):
    """The error the layer removes at fraction a, in % of the layer's baseline's clipped total MSE, mean over seeds
    (positive: lower error)."""
    def points(seed):
        total = lambda x: scores(with_layer(sd, name, x, seed), sd['u'])['total_mse']
        base = total(0.)
        return 100 * (base - total(a)) / base
    seeds = list(sd['seeds'])
    return float(np.mean([points(s) for s in (seeds[:1] if name == 'response' else seeds)]))


def seed_difference_slopes(sd):
    """The difference of two seeds as if it were a layer: rows of (layer, seeds, slope). It should carry no signal."""
    u, seeds = sd['u'], list(sd['seeds'])
    need = layers(sd, 'outside', seeds[0])
    rows = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        d = sd['seeds'][seeds[i]] - sd['seeds'][seeds[j]]
        line = project(d, u['context'], u['weight'])
        for name, x, y in (('line', line, need['line'][1]), ('perturbation', d - line, need['perturbation'][1])):
            rows.append({'layer': name, 'seeds': f'{seeds[i]}-{seeds[j]}', 'slope': slope(x, y, u['weight'])})
    return rows


def held_out_columns(sd, name, a, chain, specific):
    """One layer's held-out columns on one chain: its slopes, the best fraction, and the gains at full strength, at 1/8,
    at the calibrated fraction ``a`` and at the grid's best amount."""
    s = seed_slopes(sd, 'outside')[name]
    grid = [gain(sd, name, x) for x in POSITIVE]
    out = {f'heldout_slope_{chain}': s, f'best_{chain}': fraction(s), f'heldout_slope_bag_{chain}': seed_slopes(sd, 'b')[name],
           f'gain_raw_{chain}': gain(sd, name, 1.), f'gain_eighth_{chain}': gain(sd, name, 1 / 8),
           f'gain_fraction_{chain}': gain(sd, name, a) if a > 0 else 0., f'gain_best_grid_{chain}': max(grid)}
    if name == 'perturbation':
        out |= {f'heldout_{k}_{chain}': v for k, v in specific.items()}
    return out
