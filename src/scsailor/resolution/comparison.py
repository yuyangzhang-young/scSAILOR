"""Comparing two predictions on held-out units: the paired line bootstrap, its call, and the unit-level sign-flip test.

Both tests read *runs*: one pair (per-line parts, scale) per seed, where the per-line parts are
``scoring.line_parts`` of a prediction and the scale is the reference's clipped total (the parent's, usually).
Differences are reported in points (percent) of the scale.

- ``compare``: seeds averaged within a line, lines resampled (``bootstrap``), a call with the 0.2-point margin, and
  the call's rate of "different" when the per-line differences are centred and sign-flipped (its null rate).
- ``detect``: the card's test. Each held-out line is one unit; a contrast is detected when the exact one-sided
  sign-flip p over the units is at most ``ALPHA``, the summed difference is below minus the margin and every seed
  agrees in sign. With too few units for that p (the design floor), the result is case-level: signs per unit and seed.
"""
import itertools

import numpy as np

from .scoring import PARTS

__all__ = ["MARGIN", "ALPHA", "RESAMPLES", "bootstrap", "call", "null_rate", "seed_rule", "compare", "design_floor", "sign_patterns",
           "signflip", "detect"]

MARGIN = .2          # points of the scale: a difference smaller than this is not called "different"
ALPHA = .025         # one side of the comparison call's 95% percentile interval
EXACT_UP_TO = 14     # sign patterns enumerated exactly up to 2^14 units; drawn beyond
SIGNFLIP_DRAWS = 10000
NULL_DRAWS = 1000    # sign flips of the null rate
NULL_SHIFTS = (0., -.2, -.5, -1.)   # planted shifts of the null rate, in points of the scale
RESAMPLES = 2000     # line resamples of the bootstrap
BOOTSTRAP_SEED = 20261011


def bootstrap(differences, scale):
    """Estimate and 95% interval, in percent of ``scale``, of a summed per-line difference.

    ``differences`` maps each line to its summed difference (seeds averaged within a line). The unit is the held-out
    cell line, with all of its contexts kept together; intervals are conditional on the fitted models. Every call
    draws the same line resamples for a given number of lines, so an interval does not depend on what else was
    computed before it.
    """
    lines = sorted(differences)
    values = np.asarray([differences[line] for line in lines])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(lines), size=(RESAMPLES, len(lines)))
    sums = values[draws].sum(axis=1)
    return {"estimate_pct": 100 * float(values.sum()) / scale,
            "interval_pct": [100 * float(np.percentile(sums, 2.5)) / scale, 100 * float(np.percentile(sums, 97.5)) / scale]}


def call(result):
    """"different", "no different" or "inconclusive" for a bootstrap result."""
    low, high = result['interval_pct']
    if (high < 0 or low > 0) and abs(result['estimate_pct']) > MARGIN:
        return 'different'
    if low >= -MARGIN and high <= MARGIN:
        return 'no different'
    return 'inconclusive'


def wilson(k, n, z=1.96):
    """Wilson score interval of a proportion k / n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def seed_rule(flags):
    """'holds' when every seed agrees, 'partly' when some do, 'no' when none does."""
    return 'holds' if all(flags) else 'partly' if any(flags) else 'no'


def null_rate(mean_diff, scale):
    """The call's rate of "different" on these per-line differences centred and sign-flipped, with planted shifts."""
    names = sorted(mean_diff)
    values = np.asarray([mean_diff[n] for n in names])
    centred = values - values.mean()
    signs = np.random.default_rng(20261015).choice((-1., 1.), size=(NULL_DRAWS, len(names)))
    table = {}
    for shift in NULL_SHIFTS:
        results = [bootstrap(dict(zip(names, centred * sign + shift * scale / (100 * len(names)))), scale) for sign in signs]
        hits = sum(call(r) == 'different' and (shift == 0 or r['estimate_pct'] < 0) for r in results)
        table[str(shift)] = {'different_rate': hits / NULL_DRAWS, 'interval_95': wilson(hits, NULL_DRAWS)}
    return table


def compare(runs_x, runs_y):
    """Paired line bootstrap of x minus y per part (seeds averaged within a line), with the call and null rate."""
    result = {}
    scale = float(np.mean([t for _, t in runs_x]))
    for part in PARTS:
        names = sorted(runs_x[0][0][part])
        mean = {n: float(np.mean([x[0][part][n] - y[0][part][n] for x, y in zip(runs_x, runs_y)])) for n in names}
        r = bootstrap(mean, scale)
        if part == 'total':
            r['call'] = call(r)
            r['null_different_rate'] = null_rate(mean, scale)['0.0']['different_rate']
        result[part] = r
    return result


def design_floor(count):
    """Smallest one-sided p-value of the exact sign-flip test over `count` biological units."""
    return 0.5 ** count


def sign_patterns(n, seed=20261019):
    """Sign patterns over n units: all of them up to 2^14 (the observed all-plus pattern included), else the observed
    pattern plus SIGNFLIP_DRAWS random ones."""
    if n <= EXACT_UP_TO:
        return np.asarray(list(itertools.product((1., -1.), repeat=n)))
    return np.vstack([np.ones(n), np.random.default_rng(seed).choice((1., -1.), size=(SIGNFLIP_DRAWS, n))])


def signflip(values, seed=20261019):
    """One-sided exact sign-flip p-value that the mean of `values` is below zero (all patterns up to 2^14)."""
    values = np.asarray(values, dtype=np.float64)
    observed = values.mean()
    return float(np.mean((sign_patterns(len(values), seed) * values).mean(axis=1) <= observed + 1e-15))


def detect(x_runs, y_runs, part='total', with_bootstrap=True):
    """x minus y, per held-out unit averaged over seeds, in points of each seed's scale. Detected when the design
    floor allows a test, the one-sided sign-flip p is at most ALPHA, the total (the sum over units) is below -MARGIN
    and the seeds agree in sign; below the floor the result is case-level (signs per unit and seed). The bootstrap
    call and its null rate are reported beside it."""
    names = sorted(x_runs[0][0][part])
    per = np.asarray([[100 * (x[0][part][n] - y[part][n]) / s for x, (y, s) in zip(x_runs, y_runs)] for n in names])
    seed_totals = per.sum(axis=0)
    rule = seed_rule([v < 0 for v in seed_totals])
    unit_means = per.mean(axis=1)
    out = {'units': len(names), 'floor': design_floor(len(names)), 'total_pts': float(unit_means.sum()),
           'per_unit_pts': {n: per[i].tolist() for i, n in enumerate(names)}, 'seed_totals_pts': seed_totals.tolist(),
           'seed_rule': rule, 'seed_rule_floor': 0.125}
    if with_bootstrap and part == 'total' and len(names) > 1:
        result = compare(x_runs, y_runs)['total']
        out['bootstrap_call'] = {k: result[k] for k in ('estimate_pct', 'interval_pct', 'call', 'null_different_rate')}
    if out['floor'] > ALPHA:
        out |= {'level': 'case-level', 'detected': None,
                'case_counts': {'lower': int(np.sum(per < 0)), 'of': int(per.size)}}
        return out
    p = signflip(unit_means)
    out |= {'level': 'unit-level', 'p': p,
            'detected': bool(p <= ALPHA and unit_means.sum() < -MARGIN and rule == 'holds')}
    return out
