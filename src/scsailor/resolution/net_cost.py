"""Matched comparisons: how much of a prediction's improvement required the restoring change.

Write a for the anchor, b for the control mean, p for the original model and R for the restoring change at full
strength. The change is kept at the fraction alpha, its slope on calibration lines truncated to 0-1 (``slopes``), which
gives X = p + alpha (R - p). The design-aware baseline is b + kappa (A - b), with A the perturbation mean within the
screen's design stratum and kappa the slope of the target on A - b, truncated to 0-1, on calibration units: the screen's
design defines it and no model enters it.

Three sets of references widen one gate. Each reference is calibrated on calibration lines, and the sets accumulate:

    set 1  the field's references: no change, the baseline, the original model as trained, the context-level shrink, the
           uniform shrink and the shrink toward no change. The kept change is calibrated toward the baseline.
    set 2  the same treatment and the screen's design: the design-aware baseline, and the original model with the same
           two-amount calibration toward it. From here the kept change is calibrated toward the design-aware baseline.
    set 3  the same budget: an extra seed, the original averaged with the next seed's original under the same calibration.

The gate recommends a change above a margin against every reference in every seed, with some of the change kept. Held-out
lines report the same gaps as continuous differences and take no part in any choice.

The accounting splits the distance between the baseline and the kept change calibrated toward the design-aware baseline
in two orders. Forward: baseline, design-aware baseline, original model's correction, change, with the extra seed apart.
Reverse: baseline, change, original model's correction, design-aware baseline. The split between the first two terms
depends on the order and is not read; the change's term is small in both.

A *side* is one chain on one set of units, as a dataset's binding reads it:

- ``u``: the units (``scoring``);
- ``p``, ``R``: seed -> the original model and the change at full strength on the units (``R``: None without a change);
- on the calibration side and the held-out side of the full chain, ``a`` (the anchor, averaged over seeds), ``b`` (the
  control mean per unit) and ``A`` (the design stratum's mean).
"""
import numpy as np

from . import card, menu
from .policies import predict
from .scoring import PARTKEY, project, scores
from .slopes import fraction, inner, slope

__all__ = ["MARGINS", "part_slopes", "change_slopes", "extra_seed_slopes", "price", "partner", "extra_seed", "design_baseline", "calibrate_design", "total",
           "calibrate_original", "calibrate_change", "references", "actions", "cumulative_gaps", "recommended", "ledger", "costs",
           "reference_points", "recovery_at", "planted_reading", "stratum_shift"]

MARGINS = (0., .2, .5)   # points of the original model's total


def part_slopes(x, y, u):
    """total, common and contrast slope of the need y on x, with x's common share of energy."""
    w, context = u['weight'], u['context']
    px, py = project(x, context, w), project(y, context, w)
    energy = inner(x, x, w)
    return {'total': inner(x, y, w) / energy, 'common': slope(px, py, w), 'differential': slope(x - px, y - py, w),
            'common_energy': inner(px, px, w) / energy}


def change_slopes(sd):
    """Mean over seeds of the change's slopes on the original model's need, and the seeds' total slopes."""
    per = [part_slopes(sd['R'][s] - sd['p'][s], sd['u']['truth'] - sd['p'][s], sd['u']) for s in sd['p']]
    return {k: float(np.mean([r[k] for r in per])) for k in per[0]}, [r['total'] for r in per]


def extra_seed_slopes(sd):
    """The extra seed as a change, every ordered pair: the slope of i's need on j minus i. Two models' slopes toward each
    other add to exactly 1, so an extra seed sits at one half."""
    out = []
    for i in sd['p']:
        for j in sd['p']:
            if i != j:
                out.append({'seed': i, 'second': j, **part_slopes(sd['p'][j] - sd['p'][i], sd['u']['truth'] - sd['p'][i], sd['u'])})
    return out


def price(s):
    """Points of cost at full strength per point the best size returns; blank without a return."""
    return (1 - 2 * s) / (s * s) if s > 0 else None


def partner(sd, seed):
    """The next seed of the side, the last one's being the first."""
    seeds = list(sd['p'])
    return seeds[(seeds.index(seed) + 1) % len(seeds)]


def extra_seed(sd, seed):
    """The original model averaged with the next seed's."""
    return (sd['p'][seed] + sd['p'][partner(sd, seed)]) / 2


def design_baseline(sd, kappa):
    return sd['b'] + kappa * (sd['A'] - sd['b'])


def calibrate_design(sd):
    """kappa: the slope of truth - b on A - b, truncated to 0-1, on calibration units; no model enters."""
    u = sd['u']
    return fraction(slope(sd['A'] - sd['b'], u['truth'] - sd['b'], u['weight']))


def total(x, u):
    return scores(x, u)['total_mse']


def calibrate_original(sd):
    """The model-level amounts: kappa, and per seed the original model's and the extra seed's same calibration toward
    the design-aware baseline, and the original's field calibration and cheap alternatives."""
    u, a, b = sd['u'], sd['a'], sd['b']
    kappa = calibrate_design(sd)
    T = design_baseline(sd, kappa)
    out = {'kappa': kappa, 'seeds': {}}
    for s in sd['p']:
        p = sd['p'][s]
        entry = card.calibration_entry(p, a, b, u)
        out['seeds'][str(s)] = {'cheap': {k: entry[k]['amount'] for k in card.ALTERNATIVES},
                                'original_field': menu.pick_two(a, p, u)[0], 'original_design': menu.pick_two(T, p, u)[0],
                                'second_design': menu.pick_two(T, extra_seed(sd, s), u)[0]}
    return out


def calibrate_change(sd, model_amounts):
    """Per seed the kept fraction alpha (the change's slope truncated to 0-1) and the kept change's calibrations: toward
    the baseline, toward the design-aware baseline, and, for the reverse accounting, the change alone on the baseline."""
    u, a = sd['u'], sd['a']
    T = design_baseline(sd, model_amounts['kappa'])
    out = {}
    for s in sd['p']:
        p, delta = sd['p'][s], sd['R'][s] - sd['p'][s]
        alpha = fraction(slope(delta, u['truth'] - p, u['weight']))
        X = p + alpha * delta
        out[str(s)] = {'alpha': alpha, 'action_field': menu.pick_two(a, X, u)[0], 'action_design': menu.pick_two(T, X, u)[0],
                       'change_alone': menu.pick_two(a, a + alpha * delta, u)[0]}
    return out


def references(sd, s, amounts):
    """set -> reference -> prediction for seed s, with the model-level amounts; sets accumulate."""
    u, a, b, p = sd['u'], sd['a'], sd['b'], sd['p'][s]
    m = amounts['seeds'][str(s)]
    T = design_baseline(sd, amounts['kappa'])
    field = {'no change': b, 'baseline': a, 'original': p,
             'cell-line shrink': predict('cheapest_fix', m['cheap']['cheapest_fix'], p, None, a, u),
             'uniform shrink': predict('global_parent_shrink', m['cheap']['global_parent_shrink'], p, None, a, u),
             'shrink toward no change': card.nochange_shrink(p, b, m['cheap']['nochange_shrink'])}
    design = {'design baseline': T, 'original, same calibration': menu.two_part(T, p, *m['original_design'], u)}
    seed = {'second seed, same calibration': menu.two_part(T, extra_seed(sd, s), *m['second_design'], u)}
    return {1: field, 2: design, 3: seed}


def actions(sd, s, kept, kappa):
    """The kept change at each set: toward the baseline at set 1, toward the design-aware baseline after it."""
    p = sd['p'][s]
    X = p + kept['alpha'] * (sd['R'][s] - p)
    field = menu.two_part(sd['a'], X, *kept['action_field'], sd['u'])
    design = menu.two_part(design_baseline(sd, kappa), X, *kept['action_design'], sd['u'])
    return {1: field, 2: design, 3: design}


def cumulative_gaps(sd, amounts, kept):
    """stage -> seed -> reference -> 100 (reference - action) / original on clipped totals, against that stage's action,
    every reference up to the stage. 'variant': the original with the action's two-amount field calibration, printed
    beside stage 1 and not part of the gate."""
    u, out = sd['u'], {}
    for s in sd['p']:
        base = total(sd['p'][s], u)
        refs, acts = references(sd, s, amounts), actions(sd, s, kept[str(s)], amounts['kappa'])
        for stage in (1, 2, 3):
            action = total(acts[stage], u)
            for level in range(1, stage + 1):
                for name, x in refs[level].items():
                    out.setdefault(stage, {}).setdefault(s, {})[name] = 100 * (total(x, u) - action) / base
        same = menu.two_part(sd['a'], sd['p'][s], *amounts['seeds'][str(s)]['original_field'], u)
        out.setdefault('variant', {}).setdefault(s, {})['original, field two-amount calibration'] = 100 * (total(same, u) - total(acts[1], u)) / base
    return out


def recommended(gaps, alphas, margin):
    """Above the margin against every reference, every seed, with some of the change kept in every seed."""
    return min(alphas) > 0 and all(min(v.values()) > margin for v in gaps.values())


def ledger(sd, amounts, kept):
    """The accounting in points of the original model's total, mean over seeds, in both orders; both end at the kept
    change calibrated toward the design-aware baseline."""
    u, a = sd['u'], sd['a']
    rows = []
    for s in sd['p']:
        k = kept[str(s)]
        refs, acts = references(sd, s, amounts), actions(sd, s, k, amounts['kappa'])
        p, delta = sd['p'][s], sd['R'][s] - sd['p'][s]
        base, L0, L1 = total(p, u), total(a, u), total(refs[2]['design baseline'], u)
        L2, L3 = total(refs[2]['original, same calibration'], u), total(acts[3], u)
        seed = total(refs[3]['second seed, same calibration'], u)
        R1, R2 = total(menu.two_part(a, a + k['alpha'] * delta, *k['change_alone'], u), u), total(acts[1], u)
        pts = lambda x: 100 * x / base
        rows.append({'baseline_above_original': pts(L0 - base), 'gap': pts(L0 - L3), 'design': pts(L0 - L1), 'correction': pts(L1 - L2), 'change': pts(L2 - L3),
                     'second_seed': pts(L2 - seed), 'reverse_change': pts(L0 - R1), 'reverse_correction': pts(R1 - R2), 'reverse_design': pts(R2 - L3)})
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def costs(sd):
    """Mean over seeds of the change at full strength minus the original model, in points of its total, per part; and the
    realized price: the clipped cost over what the best held-out size returns."""
    u, rows = sd['u'], []
    for s in sd['p']:
        p, R = sd['p'][s], sd['R'][s]
        base, end = scores(p, u), scores(R, u)
        best = fraction(slope(R - p, u['truth'] - p, u['weight']))
        back = base['total_mse'] - total(p + best * (R - p), u)
        cost = end['total_mse'] - base['total_mse']
        rows.append({**{f'cost_{k}_pts': 100 * (end[v] - base[v]) / base['total_mse'] for k, v in PARTKEY.items()},
                     'return_pts': 100 * back / base['total_mse'], 'realized_price': cost / back if back > 0 else float('nan')})
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    out['cost_seeds'] = ' / '.join(f"{r['cost_total_pts']:+.2f}" for r in rows)
    out['seeds_above_zero'] = sum(r['cost_total_pts'] > 0 for r in rows)
    return out


def reference_points(sd, amounts):
    """The references without a model and the original model on a side, points of the original's total, mean over seeds."""
    u, T = sd['u'], design_baseline(sd, amounts['kappa'])
    rows = []
    for s in sd['p']:
        p, v = sd['p'][s], amounts['seeds'][str(s)]
        base = total(p, u)
        pts = lambda x: 100 * (total(x, u) - base) / base
        rows.append({'no_change': pts(sd['b']), 'baseline': pts(sd['a']), 'stratum_mean': pts(sd['A']), 'design_baseline': pts(T),
                     'original_field_calibrated': pts(menu.two_part(sd['a'], p, *v['original_field'], u)),
                     'original_design_calibrated': pts(menu.two_part(T, p, *v['original_design'], u)),
                     'second_seed_design_calibrated': pts(menu.two_part(T, extra_seed(sd, s), *v['second_design'], u))})
    return {'units': len(u['truth']), 'kappa': amounts['kappa'], **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}}


def recovery_at(q, rho, a):
    """Recovery of the change kept at fraction a: Q(a) = a (Q + rho) - a^2 rho, with the original model reading nothing."""
    return a * (q + rho) - a * a * rho


def planted_reading(support, target):
    """(recovery, predicted over true half-difference energy) of a planted state in a run's saved evaluation."""
    b = support['bits'][str(target)]
    return b['Q'], b['predicted_halfcontrast_energy'] / b['truth_halfcontrast_energy']


def stratum_shift(u, strata, train, train_bags, keep, train_strata):
    """Per unit r_stratum - r: the training response within the unit's design stratum (``strata``) minus over all strata,
    from the kept training units (``keep``, with their ``train_strata``), weighted; a perturbation absent from a stratum
    keeps r. ``train_bags``: context -> (input set's mean, pool mean)."""
    pool = np.stack([train_bags[c][1] for c in train['context']])
    change, w, code = train['truth'] - pool, train['weight'], train['p_code']
    r, within = {}, {}
    for q in np.unique(code[keep]):
        m = keep & (code == q)
        r[int(q)] = np.average(change[m], axis=0, weights=w[m])
        for s in np.unique(train_strata[m]):
            n = m & (train_strata == s)
            within[(s, int(q))] = np.average(change[n], axis=0, weights=w[n])
    zero = np.zeros(change.shape[1])
    return np.stack([within.get((s, int(q)), r.get(int(q), zero)) - r.get(int(q), zero) for s, q in zip(strata, u['p_code'])])
