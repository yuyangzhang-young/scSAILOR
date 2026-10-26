"""What to keep of a parent and its repair: the policies, their amounts chosen on calibration units, the diagnosis
labels and the selection rules.

Write p for the parent's prediction, a for its anchor (the input bag's mean plus the model's transferred response), R
for the repair at full strength, r = R - p and P for the per-context common projection (the cell-line-level part).
Every amount is chosen on the masked calibration units by the lowest clipped total over one grid ``G``, with
clipping after combination and ties to less change:

    parent                 p
    full_repair            R
    global_parent_shrink   a + lam (p - a)              (the uniform shrink)
    cheapest_fix           p - (1 - lam) P(p - a)       (the cell-line-level shrink)
    size_fix               p + alpha r
    selective_repair       p + lam P(r) + (r - P(r)), or p ("no repair", amount None)
    always_selective       R - (1 - lam) P(R - a)
    protect_common         R - P(R - p)
    <policy>_at_parent_lam the policy's prediction X, then X - (1 - lam_p) P(X - a) at the parent's cell-line amount

``select`` is the current selection rule (ordinary validation over a menu); ``diagnose`` is the earlier
decision-table rule, kept because saved selections were made with it.
"""
from statistics import mean

import numpy as np

from .comparison import MARGIN, seed_rule
from .scoring import project, scores

__all__ = ["G", "FINE", "MARGIN", "FIXED", "REPAIRS", "lowest", "predict", "choose", "labels", "case_labels", "diagnose",
           "default_menu", "select"]

G = (0., .125, .25, 1 / 3, .5, .75, 1.)                 # the amount grid
FINE = tuple(np.round(np.arange(0, 1.0001, .025), 4))   # the size curve of the labels
FIXED = ('parent', 'full_repair', 'global_parent_shrink', 'cheapest_fix', 'size_fix', 'selective_repair', 'always_selective')
REPAIRS = ('full_repair', 'size_fix', 'protect_common')


def lowest(total, options):
    """The option with the lowest total(option), the earliest on a tie (callers order the options so that a tie goes to
    less change), and that total."""
    best = min([(total(x), i, x) for i, x in enumerate(options)], key=lambda v: (v[0], v[1]))
    return best[2], best[0]


def line_part(values, units):
    return project(values, units['context'], units['weight'])


def at_parent_lam(prediction, a, lam, units):
    return prediction - (1 - lam) * line_part(prediction - a, units)


def predict(policy, x, p, R, a, units):
    """The prediction of one policy at its amount x (None: no amount, or "no repair")."""
    if policy.endswith('_at_parent_lam'):
        inner, lam = policy[:-len('_at_parent_lam')], x[1]
        return at_parent_lam(predict(inner, x[0], p, R, a, units), a, lam, units)
    if policy == 'parent' or (policy == 'selective_repair' and x is None):
        return p
    if policy == 'full_repair':
        return R
    if policy == 'global_parent_shrink':
        return a + x * (p - a)
    if policy == 'cheapest_fix':
        return at_parent_lam(p, a, x, units)
    if policy == 'size_fix':
        return p + x * (R - p)
    if policy == 'selective_repair':
        line = line_part(R - p, units)
        return p + x * line + (R - p - line)
    if policy == 'always_selective':
        return at_parent_lam(R, a, x, units)
    if policy == 'protect_common':
        return R - line_part(R - p, units)
    raise ValueError(policy)


def choose(policy, p, R, a, units, mask):
    """The policy's amount chosen on the masked rows (ties to less change), and its clipped total there."""
    total = lambda x: scores(predict(policy, x, p, R, a, units), units, mask)['total_mse']
    if policy in ('parent', 'full_repair', 'protect_common'):
        return None, total(None)
    return lowest(total, [None] + list(G) if policy == 'selective_repair' else list(G) if policy == 'size_fix' else list(reversed(G)))


def labels(p, R, units, mask):
    """The diagnosis on the masked rows: is the full-strength repair better than the parent; does its cost sit at the
    cell-line level (d1a); is the common part damaged at every amount (d1b); is the repair too large (d1c)."""
    base, full = scores(p, units, mask), scores(R, units, mask)
    total = base['total_mse']
    increase = 100 * (full['total_mse'] - total) / total
    common = 100 * (full['common_error_mse'] - base['common_error_mse']) / total
    curve = {alpha: scores(p + alpha * (R - p), units, mask) for alpha in FINE}
    optimum = min(FINE, key=lambda alpha: (curve[alpha]['total_mse'], alpha))
    gain = 100 * (total - curve[optimum]['total_mse']) / total
    return {'repair_better_than_parent': bool(increase < 0), 'full_strength_pct': increase, 'common_pct': common,
            'd1a': 'no full-strength cost' if increase <= 0 else 'line-level' if common / increase > .5
            else 'drug-level' if common / increase < .5 else 'mixed',
            'd1b_common_damaged': all(curve[alpha]['common_error_mse'] > base['common_error_mse'] for alpha in FINE if alpha > 0),
            'd1c': 'no usable direction' if gain < MARGIN else 'about right' if optimum == 1. else 'too large',
            'd1c_optimum': optimum, 'd1c_gain_pct': gain}


def case_labels(per_seed):
    """Case labels: one label when every seed agrees, otherwise "mixed"; d1b and the route by the seed rule."""
    one = lambda values: values[0] if len(set(values)) == 1 else 'mixed'
    return {'d1a': one([s['d1a'] for s in per_seed]), 'd1c': one([s['d1c'] for s in per_seed]),
            'd1b_common_damaged': seed_rule([s['d1b_common_damaged'] for s in per_seed]),
            'route': 'step 4' if all(s['repair_better_than_parent'] for s in per_seed) else 'step 2'}


def diagnose(seeds, synthetic, can_protect_common):
    """The decision table as a rule, on calibration only.

    ``seeds``: seed -> {'labels', 'calibration'}; ``synthetic``: the repair's synthetic reading ('recovered',
    'partial' or 'not recovered'); ``can_protect_common``: whether the model has a separate common branch to protect
    (the summary-statistics model). Candidates: the parent and the cheapest fix always. Repair candidates, none if the
    synthetic reading is "not recovered": the full repair when the route is step 4 or the size is about right; the
    size fix when the size is too large; protection of the common part when it is damaged in all seeds. Every repair
    candidate also enters shrunk at the parent's lambda. The cheapest fix replaces the parent as default when it is
    below it in all seeds; a repair candidate replaces the default when it is below it in all seeds (the lowest mean
    calibration change among such candidates)."""
    case = case_labels([s['labels'] for s in seeds.values()])
    repairs = []
    if synthetic != 'not recovered':
        if case['route'] == 'step 4' or case['d1c'] == 'about right':
            repairs.append('full_repair')
        if case['d1c'] == 'too large':
            repairs.append('size_fix')
        if can_protect_common and case['d1b_common_damaged'] == 'holds':
            repairs.append('protect_common')
    candidates = [form for t in repairs for form in (t, f'{t}_at_parent_lam')]
    cal = lambda t: {seed: s['calibration'][t] for seed, s in seeds.items()}
    beats = lambda t, u: all(cal(t)[k] < cal(u)[k] for k in seeds)
    default = 'cheapest_fix' if beats('cheapest_fix', 'parent') else 'parent'
    winners = [t for t in candidates if beats(t, default)]
    action = min(winners, key=lambda t: np.mean([cal(t)[k] / cal('parent')[k] for k in seeds])) if winners else default
    case.update(synthetic=synthetic, candidates=candidates, default=default, action=action,
                cause_distinguished=case['d1a'] != 'mixed')
    return case


def default_menu(can_protect_common):
    """The actions the current rule chooses from when a model has an IL repair; protecting the common part only where
    the model has a separate common branch (the summary-statistics model)."""
    actions = ['parent', 'cheapest_fix', 'full_repair', 'full_repair_at_parent_lam',
               'size_fix', 'size_fix_at_parent_lam']
    if can_protect_common:
        actions += ['protect_common', 'protect_common_at_parent_lam']
    return actions


def select(entries, actions):
    """The current rule over a menu of actions (the first two are the parent and the cheapest fix): ordinary
    validation picks the lowest mean parent-normalized calibration total; the gated consensus is reported beside it.
    ``entries``: seed -> {'calibration': action -> total}."""
    cal = {a: [e['calibration'][a] / e['calibration']['parent'] for e in entries.values()]
           for a in actions}
    average = {a: mean(v) for a, v in cal.items()}
    beats = lambda a, b: all(x < y for x, y in zip(cal[a], cal[b]))
    default = 'cheapest_fix' if beats('cheapest_fix', 'parent') else 'parent'
    winners = [a for a in actions[2:] if beats(a, default)]
    return {'menu': actions, 'default': default, 'mean_normalized_calibration': average,
            'beats_default_all_seeds': {a: beats(a, default) for a in actions},
            'ungated_consensus': min(winners, key=average.get) if winners else default,
            'ordinary_validation': min(actions, key=average.get)}
