"""The single-model repair menu: one trained model, one seed, repaired from its own predictions.

X is the seed's parent p, or its information-localization repair R = p + alpha (R_IL - p) at origin resolution's
calibrated size alpha (where a cell with alpha > 0 is handed on). T is the target the corrections shrink toward: the
plate-aware anchor A where the screen measures every line on the same two replicate plates with pooled wells
(Tahoe-100M's plates 6 and 14), the anchor a elsewhere.

    shrink_<X>     T + lam (X - T)                                  (shrink_parent only with a plate-aware target:
                                                                    elsewhere it is the uniform shrink, a cheap
                                                                    alternative)
    two_part_<X>   T + lam_c P(X - T) + lam_d (X - T - P(X - T))    the context-level and perturbation parts apart
    corrected_by_<m>   for a published model: the two-amount shrink toward T of p + beta (q - p), q the same seed's
                   parent of in-house architecture m; beta chosen first (ties to the smaller), then the two amounts.

A = a + r_plate[plate] - r, with r and r_plate the training lines' drug averages over both plates and per plate
(``plate_effects``); each context's plate is read from its input bag by the paired plate direction of the training
lines (``plates_from_bags``), since the task's inputs carry no plate label. ``plates`` names the two plates, as the
second field of the context keys (``"<line>|<plate>"``), in the order of the plate direction. With a plate-aware
target the cheap alternatives add A itself, the uniform shrink toward A and the context-level shrink toward A. Every
amount is chosen on the calibration units of the nested chain: one amount by ``card.calibrate``, two by the lowest
clipped total over G x G, ties to the larger.
"""
import numpy as np

from . import card
from .policies import G, lowest
from .scoring import project, scores

__all__ = ["MENU", "PLATE_ALTERNATIVES", "plate_effects", "plates_from_bags", "plate_aware_shift", "side", "available", "two_part", "pick_two",
           "pick_beta", "predict", "add_items", "normalized", "choose"]

MENU = ('shrink_parent', 'two_part_parent', 'shrink_repair', 'two_part_repair')
PLATE_ALTERNATIVES = ('plate_baseline', 'shrink_parent', 'cell_line_shrink_A')   # cheap alternatives with a plate-aware target


def plate_effects(change, line, plate, code, w, lines, n, *, plates):
    """r[drug] and r_plate[plate][drug]: the weighted mean over the training units of ``lines`` of each unit's change
    (its target minus its context's control mean), over both plates or per plate; a drug absent on a plate keeps r.
    ``line``, ``plate``, ``code`` and ``w`` are per unit; ``n`` is the number of drugs."""
    keep = np.isin(line, lines)
    mean = lambda m: np.average(change[m], axis=0, weights=w[m])
    r = np.stack([mean(keep & (code == q)) for q in range(n)])
    r_plate = {pl: np.stack([mean(keep & (code == q) & (plate == pl)) if np.any(keep & (code == q) & (plate == pl)) else r[q]
                             for q in range(n)]) for pl in plates}
    return r, r_plate


def plates_from_bags(train_bags, lines, bags, contexts, *, plates):
    """Each context's plate read from its input bag: the first plate when (b - mu) . d > 0, with d the training lines'
    mean first-minus-second plate bag difference and mu their mean bag. Bags map a context to (bag mean, pool mean)."""
    first, second = plates
    pairs = [(train_bags[f'{l}|{first}'][0], train_bags[f'{l}|{second}'][0]) for l in lines]
    d = np.mean([x1 - x2 for x1, x2 in pairs], axis=0)
    mu = np.mean([x for pair in pairs for x in pair], axis=0)
    return {c: first if (bags[c][0] - mu) @ d > 0 else second for c in contexts}


def plate_aware_shift(u, r, r_plate, train_bags, lines, bags, *, plates):
    """Per unit r_plate[plate] - r, the plate-aware target's shift from the anchor, with each context's plate read from
    its bag, and the agreement of the read plates with the contexts' labels."""
    contexts = sorted(set(u['context']))
    plate_of = plates_from_bags(train_bags, lines, bags, contexts, plates=plates)
    check = {'contexts': len(contexts), 'agree_with_label': sum(plate_of[c] == c.split('|')[1] for c in contexts)}
    return np.stack([r_plate[plate_of[c]][q] - r[q] for c, q in zip(u['context'], u['p_code'])]), check


def side(u, p, a, plate_shift=None, repair_end=None, repair_alpha=None, options=None, plate_check=None):
    """The menu's inputs on one side (calibration units with the nested chain, or outer units with the full chain).

    p, a: seed -> the parent's prediction and anchor on the units ``u``; plate_shift: per unit r_plate - r, giving
    the plate-aware target A = a + plate_shift; repair_end(seed): the IL cell's end prediction, read only when the
    repair is used; repair_alpha: seed -> origin resolution's calibrated size (the repair base is dropped when every
    alpha is 0, since it would repeat the parent's items); options: in-house model -> seed -> its parent's prediction
    (a published model's corrections)."""
    sd = {'u': u, 'p': p, 'a': a, 'A': None, 'T': a, 'bases': {}, 'option': dict(options or {}), 'plate_check': plate_check}
    if plate_shift is not None:
        sd['A'] = sd['T'] = {s: a[s] + plate_shift for s in p}
    if repair_end is not None and all(repair_alpha.get(str(s)) is not None for s in p) and any(repair_alpha[str(s)] > 0 for s in p):
        sd['bases']['repair'] = {s: p[s] + repair_alpha[str(s)] * (repair_end(s) - p[s]) for s in p}
    return sd


def available(sd):
    """The menu's items on this side, and a published model's options."""
    items = [i for i in MENU if (i.endswith('_parent') or 'repair' in sd['bases']) and not (i == 'shrink_parent' and sd['A'] is None)]
    return items, [f'corrected_by_{m}' for m in sd['option']]


def two_part(T, X, lam_c, lam_d, u):
    d = X - T
    pd = project(d, u['context'], u['weight'])
    return T + lam_c * pd + lam_d * (d - pd)


def pick_two(T, X, u):
    """(lam_c, lam_d) with the lowest clipped calibration total over G x G; ties to the larger amounts."""
    (lam_c, lam_d), total = lowest(lambda x: scores(two_part(T, X, x[0], x[1], u), u)['total_mse'],
                                   [(c, d) for c in reversed(G) for d in reversed(G)])
    return [float(lam_c), float(lam_d)], total


def pick_beta(p, q, u):
    """The blend amount: lowest clipped total of p + x (q - p) over G, ties to the smaller x."""
    return lowest(lambda x: scores(p + x * (q - p), u)['total_mse'], G)[0]


def predict(name, amount, s, sd):
    """Item `name` at its amount for seed s on a side (the dictionary of side())."""
    u, T, p = sd['u'], sd['T'][s], sd['p'][s]
    if name == 'plate_baseline':
        return sd['A'][s]
    if name == 'cell_line_shrink_A':
        return two_part(sd['A'][s], p, amount, 1., u)
    if name.startswith('corrected_by_'):
        beta, lam_c, lam_d = amount
        q = sd['option'][name[len('corrected_by_'):]][s]
        return two_part(T, p + beta * (q - p), lam_c, lam_d, u)
    form, base = ('shrink', name[len('shrink_'):]) if name.startswith('shrink_') else ('two_part', name[len('two_part_'):])
    X = p if base == 'parent' else sd['bases']['repair'][s]
    return T + amount * (X - T) if form == 'shrink' else two_part(T, X, amount[0], amount[1], u)


def add_items(entries, sd):
    """Adds the menu's items, the options and, with a plate-aware target, the added cheap alternatives to the
    calibration entries (seed -> item -> {'amount', 'calibration'})."""
    u = sd['u']
    items, options = available(sd)
    for key, e in entries.items():
        s = int(key)
        T, p = sd['T'][s], sd['p'][s]
        for name in items:
            X = p if name.endswith('_parent') else sd['bases']['repair'][s]
            amount, total = card.calibrate(lambda y: T + y * (X - T), u) if name.startswith('shrink_') else pick_two(T, X, u)
            e[name] = {'amount': amount, 'calibration': total}
        for name in options:
            q = sd['option'][name[len('corrected_by_'):]][s]
            beta = pick_beta(p, q, u)
            (lam_c, lam_d), total = pick_two(T, p + beta * (q - p), u)
            e[name] = {'amount': [beta, lam_c, lam_d], 'calibration': total}
        if sd['A'] is not None:
            e['plate_baseline'] = {'amount': None, 'calibration': scores(sd['A'][s], u)['total_mse']}
            lam, total = card.calibrate(lambda y: two_part(sd['A'][s], p, y, 1., u), u)
            e['cell_line_shrink_A'] = {'amount': lam, 'calibration': total}


def normalized(entries, names):
    """Each item's calibration total over the parent's, averaged over seeds (items present in every seed)."""
    return {n: float(np.mean([e[n]['calibration'] / e['parent']['calibration'] for e in entries.values()]))
            for n in names if all(n in e for e in entries.values())}


def choose(entries, previous, previous_calibration, sd):
    """The full-menu rule over the earlier action and the menu's items: the lowest mean parent-normalized calibration
    total; the earlier action is kept on a tie. Returns the action, the menu's normalized totals and the option's best
    item (or None) with its normalized total."""
    items, options = available(sd)
    totals = {previous: float(np.mean([previous_calibration[k] / e['parent']['calibration'] for k, e in entries.items()])),
              **normalized(entries, items)}
    action = min(totals, key=lambda k: (totals[k], k != previous))
    option_totals = normalized(entries, options)
    option = min(option_totals, key=option_totals.get) if option_totals else None
    return action, totals, option, option_totals
