"""The field's baselines and decision methods beside origin resolution's cheap alternatives and its action, on every
screen, model and fold where origin resolution has readings; saved predictions, cards and selections only.

Two stages, run in this order, writing into ``--out``:

- ``parse-replay``: ``PARSE_MENU.csv``, origin resolution on Parse PBMC, which has no cards: per group and model the
  repair selections' rule's action and then the full menu's, chosen on the calibration donors (the nested repair chain,
  Parse's repair at its end); each item's value per seed on the calibration donors and on the held-out donors (the
  real chain), in points of the original model's total MSE; ``action``, the chosen item (on the whole set of
  calibration donors);
- ``table``: ``BASELINES_SIDE_BY_SIDE.csv``, one row per screen, fold, model, change, predictor and seed, and a mean row
  per predictor with its margins against origin resolution's action (the minimum over seeds of the row minus the
  action; positive where the action is lower in every seed). Every value is the row minus the original model, in
  points of the original model's total MSE on the same units: held-out lines (the full chain) and calibration lines
  (the nested chain, the units and metric of origin resolution's calibration). Reads ``CONTINUED_X1.csv``
  (``il_cells``) and ``PARSE_MENU.csv`` from ``--out``.

    python -m reproduce.origin_resolution.baselines --stage table --out DIR
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from scsailor.resolution import card, grids, menu, policies
from scsailor.resolution.policies import G
from scsailor.resolution.stages import bag_of
from scsailor.resolution.scoring import scores
from . import ROOT, GENETIC_READINGS, SEEDS, cards, gse306429, il_cells, jiang, parse, read_csv, replogle, tahoe, write_csv

__all__ = ["CASES", "Bank", "replay", "parse_replay_stage", "table_stage", "main"]

PROGRAM = tahoe.PROGRAM / 'results'   # Tahoe's cards; with a change in the menu, under il-cell-<model>-<change>/
NESTED_F1 = ROOT / 'nested-parents-gse306429-replogle-nadig/cards'   # GSE306429's and Replogle-Nadig fold 1's cards
GENETIC_CARDS = {'replogle': GENETIC_READINGS / 'replogle-nadig-fold-2/nested',
                 'jiang': GENETIC_READINGS / 'jiang/nested'}   # the other genetic folds' cards
TAHOE_CELLS = ROOT / 'continued-training-readings/il-cells'   # PER_SEED_<fold>.csv (il_cells)
BINDINGS = {'tahoe': tahoe, 'gse306429': gse306429, 'replogle': replogle, 'jiang': jiang}
MODELS = ('summary', 'deepsets')
CASES = ([('tahoe', f, m) for f in ('F1', 'F2', 'F3') for m in ('summary', 'deepsets', 'attention', 'official_state')]
         + [('gse306429', 'F1', m) for m in MODELS] + [('replogle', f, m) for f in ('F1', 'F2') for m in MODELS]
         + [('jiang', f, m) for f in ('F4', 'F5') for m in MODELS] + [('parse', g, m) for g in parse.FOLDS for m in MODELS])
CHEAP = (('cheapest_fix', 'cell-line-level shrink'), ('global_parent_shrink', 'uniform shrink'),
         ('nochange_shrink', 'shrink toward no change'), ('shrink_parent', 'shrink toward the plate-aware baseline'),
         ('cell_line_shrink_A', 'cell-line shrink toward the plate-aware baseline'))
slug = lambda cell: cell.replace('/', '-')
pts = lambda x, base: 100 * (x - base) / base
T = lambda x, u: scores(x, u)['total_mse']


# ---------------------------------------------------------------- Parse PBMC: origin resolution replayed
class Bank:
    """One seed's units and predictions on one side, scored as the replay scored them: the weighted mean over units of
    each unit's mean squared error of the clipped prediction."""

    def __init__(self, u, p, a, b, R):
        self.u, self.p, self.a, self.b, self.R, self.rows = u, p, a, b, R, {}

    def prediction(self, name, amount=None, alpha=None):
        if name == 'nochange_shrink':
            return card.nochange_shrink(self.p, self.b, amount)
        if name in menu.MENU:
            bases = {'repair': {0: self.p + alpha * (self.R - self.p)}} if name.endswith('_repair') else {}
            side = {'u': self.u, 'p': {0: self.p}, 'a': {0: self.a}, 'T': {0: self.a}, 'A': None, 'bases': bases, 'option': {}}
            return menu.predict(name, amount, 0, side)
        return policies.predict(name, amount, self.p, self.R, self.a, self.u)

    def risk(self, name, amount=None, alpha=None, direct=False):
        if direct:
            return scores(self.prediction(name, amount, alpha), self.u)['total_mse']
        key = (name, json.dumps(amount), alpha)
        if key not in self.rows:
            error = np.maximum(self.prediction(name, amount, alpha), 0) - self.u['truth']
            self.rows[key] = np.mean(error * error, axis=1)
        return float(np.average(self.rows[key], weights=self.u['weight']))

    def pick(self, name, alpha=None, options=None):
        """The amount with the lowest total, ties to the earlier option; totals within 1e-13 of the lowest are scored
        again by the natural task's risk."""
        if options is None:
            options = list(itertools.product(reversed(G), repeat=2)) if name.startswith('two_part_') else list(reversed(G))
        values = [self.risk(name, x, alpha) for x in options]
        best = min(values)
        close = [i for i, v in enumerate(values) if abs(v - best) <= 1e-13 * max(abs(best), 1e-12)]
        if len(close) > 1:
            for i in close:
                values[i] = self.risk(name, options[i], alpha, direct=True)
        i = min(range(len(options)), key=lambda j: (values[j], j))
        return {'amount': options[i], 'calibration': values[i]}


def earlier(bank):
    """The repair selections' rule's entries: every action of its menu (``policies.default_menu``) at its calibrated
    amount."""
    e = {'amount': {}, 'calibration': {}}
    for name in policies.default_menu(True) + ['global_parent_shrink']:
        if name.endswith('_at_parent_lam'):
            amount = (e['amount'][name[:-len('_at_parent_lam')]], e['amount']['cheapest_fix'])
            total = bank.risk(name, amount)
        elif name in ('parent', 'full_repair', 'protect_common'):
            amount, total = None, bank.risk(name)
        else:
            chosen = bank.pick(name, options=list(G) if name == 'size_fix' else None)
            amount, total = chosen['amount'], chosen['calibration']
        e['amount'][name], e['calibration'][name] = amount, total
    return e


def replay(group, model):
    """One Parse PBMC case's menu rows: the repair selections' rule's diagnosis and the size fix's amount on the
    calibration donors, then the full menu over its action."""
    case = parse.case(model, group)
    banks = {}
    for which, chain, side, bags in (('calibration', 'nested', 'train', 'train'), ('outer', 'full', 'outer', 'heldout')):
        u = case['units'][which]
        b = bag_of(case['bags'][bags], u)[0]
        banks[which] = {s: Bank(u, case['load'](s, chain, side, u), case['load'](s, chain, side, u, 'anchor_raw'), b,
                                case['load'](s, f'cell_{chain}', side, u)) for s in SEEDS}
    old = {str(s): earlier(bank) for s, bank in banks['calibration'].items()}
    for s, bank in banks['calibration'].items():
        old[str(s)]['labels'] = policies.labels(bank.p, bank.R, bank.u, None)
    previous = policies.diagnose(old, 'recovered', model == 'summary')['action']
    alpha = {k: e['amount']['size_fix'] for k, e in old.items()}
    sd = {'A': None, 'bases': {'repair': {}} if any(x > 0 for x in alpha.values()) else {}, 'option': {}}
    items, _ = menu.available(sd)
    entries = {}
    for s, bank in banks['calibration'].items():
        k = str(s)
        entries[k] = {'parent': {'amount': None, 'calibration': bank.risk('parent')},
                      **{name: bank.pick(name) for name in card.ALTERNATIVES}, **{name: bank.pick(name, alpha[k]) for name in items}}
    action, normalized, _, _ = menu.choose(entries, previous, {k: e['calibration'][previous] for k, e in old.items()}, sd)
    for k, e in entries.items():
        e.setdefault(previous, {'amount': old[k]['amount'][previous], 'calibration': old[k]['calibration'][previous]})
    names = list(dict.fromkeys(('parent',) + card.ALTERNATIVES + tuple(normalized)))
    rows = []
    for k, e in entries.items():
        bank = banks['outer'][int(k)]
        held = {name: bank.risk(name, e[name]['amount'], alpha[k]) for name in names}
        for name in names[1:]:
            rows.append({'card': f'Parse_{group}_{model}', 'seed': k, 'policy': name, 'amount': json.dumps(e[name]['amount']),
                         'alpha': alpha[k], 'cal_pts': 100 * (e[name]['calibration'] / e['parent']['calibration'] - 1),
                         'outer_pts': 100 * (held[name] / held['parent'] - 1), 'action': action})
    return rows


def parse_replay_stage(out):
    write_csv(Path(out) / 'PARSE_MENU.csv', [r for group in parse.FOLDS for model in MODELS for r in replay(group, model)])


# ---------------------------------------------------------------- origin resolution's results and the changes' readings
def or_result(screen, fold, model, cell=None):
    """(CALIBRATION, ATTRIBUTION_RUN) of origin resolution for the model (cell None: its own result) or with the
    change in its menu; None where no such result exists."""
    if screen == 'tahoe':
        if model in tahoe.F23_READINGS and fold != 'F1':
            folder = tahoe.F23_READINGS[model] / ('nested' if cell is None else f'il-cell-{model}-{slug(cell)}')
        elif cell is None or cell == tahoe.CELL.get(model):
            folder = PROGRAM
        else:
            folder = PROGRAM / f'il-cell-{model}-{slug(cell)}'
    elif cell is None and (screen == 'gse306429' or (screen, fold) == ('replogle', 'F1')):
        folder = NESTED_F1
    else:
        folder = GENETIC_CARDS[screen] if cell is None else GENETIC_READINGS / f'il-cell-{model}-{slug(cell)}'
    files = [folder / f'{kind}_{screen}_{fold}_{model}.json' for kind in ('CALIBRATION', 'ATTRIBUTION_RUN')]
    return tuple(json.loads(f.read_text()) for f in files) if all(f.is_file() for f in files) else None


def or_numbers(result):
    """Per seed: the action's calibration and held-out values against the original model, and every cheap
    alternative's."""
    cal, run = result
    v = run['verification']
    action = cal['current_action']
    seeds = [str(s) for s in SEEDS]
    base = {s: cal['entries'][s]['parent']['calibration'] for s in seeds}
    held_action = v['parent']['seed_totals_pts'] if 'parent' in v else [0.] * len(seeds)
    out = {'action': action, 'action_cal': [pts(cal['action_calibration'][s], base[s]) for s in seeds],
           'action_held': list(held_action), 'amounts': [cal['action_amounts'][s] for s in seeds],
           'alpha': [cal['menu_inputs']['repair_alpha'][s] for s in seeds], 'alternatives': {}}
    for name, _ in CHEAP:
        if name in cal['entries'][seeds[0]]:
            held = held_action if name == action else [a - b for a, b in zip(held_action, v[name]['seed_totals_pts'])]
            out['alternatives'][name] = ([pts(cal['entries'][s][name]['calibration'], base[s]) for s in seeds], held)
    return out


def parse_action(rows):
    """Parse's action and cheap alternatives from the replay's menu rows of the case."""
    by = {}
    for r in rows:
        by.setdefault(r['policy'], {})[int(r['seed'])] = r
    name = rows[0]['action']
    get = lambda pol, col: [0. if pol == 'parent' else float(by[pol][s][col]) for s in SEEDS]
    alternatives = {pol: (get(pol, 'cal_pts'), get(pol, 'outer_pts')) for pol in card.ALTERNATIVES if pol != name}
    return {'action': name, 'action_cal': get(name, 'cal_pts'), 'action_held': get(name, 'outer_pts'),
            'amounts': [json.loads(by[name][s]['amount']) if name in by else None for s in SEEDS],
            'alpha': [float(by[name][s]['alpha']) if name in by and by[name][s]['alpha'] else None for s in SEEDS],
            'alternatives': alternatives}


def change_tables(screen, fold, model):
    """(per-seed rows by change, selections by change) of the restoring changes, or ({}, {})."""
    il = tahoe.IL_NAME.get(model, model)
    if screen == 'tahoe':
        table, selection = TAHOE_CELLS / f'PER_SEED_{fold}.csv', lambda cell: tahoe.selection_path(model, fold, cell, None)
    elif screen in ('replogle', 'jiang'):
        folder = GENETIC_READINGS / 'il-cells'
        table = folder / f'PER_SEED_{screen}_{fold}.csv'
        selection = lambda cell: folder / f'SELECTIONS_{screen}_{fold}_{model}_{slug(cell)}.json'
    else:
        return {}, {}
    if not table.is_file():
        return {}, {}
    per = {}
    for r in read_csv(table):
        if r['model'] == il:
            per.setdefault(r['cell'], {})[int(r['seed'])] = r
    return per, {c: json.loads(selection(c).read_text()) for c in per if selection(c).is_file()}


def continued_x1(out, fold, model):
    """Per seed, continued training minus the original model on the held-out lines."""
    name = il_cells.MODEL_NAME[tahoe.IL_NAME.get(model, model)]
    found = {int(r['seed']): float(r['continued_total_pts']) for r in read_csv(Path(out) / 'CONTINUED_X1.csv') if (r['fold'], r['model']) == (fold, name)}
    return [found[s] for s in SEEDS]


def sides(screen, fold, model):
    """Per side ('calibration': the nested chain on the calibration units; 'outer': the full chain on the held-out
    units) the menu's inputs and the no-change prediction b; the changes' prediction files (change -> seed -> chain ->
    file); the training units, which only Parse PBMC's pick by seen lines reads."""
    if screen == 'parse':
        case = parse.case(model, fold)
        S = {}
        for w, chain, side in (('calibration', 'nested', 'train'), ('outer', 'full', 'outer')):
            u = case['units'][w]
            S[w] = menu.side(u, {s: case['load'](s, chain, side, u) for s in SEEDS}, {s: case['load'](s, chain, side, u, 'anchor_raw') for s in SEEDS})
        b = {w: bag_of(case['bags']['train' if w == 'calibration' else 'heldout'], S[w]['u'])[0] for w in S}
        return S, b, lambda cell: case['grids'], parse.fold_training_units(fold)
    binding = BINDINGS[screen]
    case = binding.case(model, fold, seeds=SEEDS)
    no_change_kept = {str(s): None for s in SEEDS}   # the menu without a change's base
    S = {w: cards.menu_side(case, w, no_change_kept, SEEDS) for w in ('calibration', 'outer')}
    b = {w: bag_of(case['bags']['train' if w == 'calibration' else 'heldout'], S[w]['u'])[0] for w in S}
    grid_paths = lambda cell: (tahoe.grid_paths(model, fold, SEEDS, cell) if screen == 'tahoe' else binding.grid_paths(model, SEEDS, fold, cell))
    return S, b, grid_paths, None


# ---------------------------------------------------------------- rows
def emit(rows, key, order, group, row, cal=None, held=None, status='computed', source='', note='', or_cols=None, action=None):
    """One predictor: a row per seed and a mean row with the margins of origin resolution's action against it
    (minimum over seeds of row minus action; positive: the action is lower in every seed)."""
    screen, fold, model, change = key
    common = {'screen': screen, 'fold': fold, 'model': model, 'change': change, 'order': order, 'group': group, 'row': row,
              'status': status, 'source': source, 'note': note, **(or_cols or {})}
    for i, s in enumerate(SEEDS):
        rows.append(common | {'seed': s, 'calibration_minus_original_pts': '' if cal is None or cal[i] is None else cal[i],
                              'heldout_minus_original_pts': '' if held is None or held[i] is None else held[i]})
    mean = lambda x: '' if x is None or any(v is None for v in x) else float(np.mean(x))
    margin = lambda x, y: '' if x is None or y is None or any(v is None for v in x) else min(a - b for a, b in zip(x, y))
    rows.append(common | {'seed': 'mean', 'calibration_minus_original_pts': mean(cal), 'heldout_minus_original_pts': mean(held),
                          'action_margin_calibration_pts': margin(cal, action and action['action_cal']),
                          'action_margin_heldout_pts': margin(held, action and action['action_held'])})


def or_columns(numbers, label):
    if numbers is None:
        return {'or_result': label, 'or_action': 'no origin-resolution result with this change in the menu'}
    return {'or_result': label, 'or_action': numbers['action']}


def build_case(screen, fold, model, out, parse_menu, rows):
    key = (screen, fold, model, '')
    S, b, grid_paths, train = sides(screen, fold, model)
    action = parse_action(parse_menu[f'Parse_{fold}_{model}']) if screen == 'parse' else or_numbers(or_result(screen, fold, model))
    tot = {w: {} for w in S}   # totals per side and seed
    for w in S:
        u, p, a = S[w]['u'], S[w]['p'], S[w]['a']
        tot[w]['parent'] = [T(p[s], u) for s in SEEDS]
        tot[w]['anchor'] = [T(a[s], u) for s in SEEDS]
        tot[w]['nochange'] = [T(b[w], u)] * len(SEEDS)
        if S[w]['A'] is not None:
            tot[w]['plate'] = [T(S[w]['A'][s], u) for s in SEEDS]
        tot[w]['extra'] = [T((p[s] + p[SEEDS[(i + 1) % 3]]) / 2, u) for i, s in enumerate(SEEDS)]
    amounts = {}
    for name, mix in (('two', lambda p, i: (p[SEEDS[i]] + p[SEEDS[(i + 1) % 3]]) / 2), ('three', lambda p, i: np.mean([p[x] for x in SEEDS], axis=0))):
        amounts[name] = []
        for w in S:
            tot[w][name] = []
        for i, s in enumerate(SEEDS):
            (lc, ld), _ = menu.pick_two(S['calibration']['T'][s], mix(S['calibration']['p'], i), S['calibration']['u'])
            amounts[name].append((lc, ld))
            for w in S:
                tot[w][name].append(T(menu.two_part(S[w]['T'][s], mix(S[w]['p'], i), lc, ld, S[w]['u']), S[w]['u']))
    rel = lambda w, name: [pts(x, y) for x, y in zip(tot[w][name], tot[w]['parent'])]
    own = or_columns(action, "origin resolution's result for the model" + (
        f" (with {tahoe.CELL[model]} in its menu)" if screen == 'tahoe' and model in tahoe.CELL else ''))
    # field baselines
    emit(rows, key, 1, 'field baseline', 'no change (control mean b)', rel('calibration', 'nochange'), rel('outer', 'nochange'), or_cols=own, action=action)
    emit(rows, key, 2, 'field baseline', 'perturbation mean (anchor a = b + r)', rel('calibration', 'anchor'), rel('outer', 'anchor'), or_cols=own, action=action)
    if 'plate' in tot['outer']:
        emit(rows, key, 3, 'field baseline', 'plate-aware baseline (A = a + r_plate - r)', rel('calibration', 'plate'), rel('outer', 'plate'), or_cols=own, action=action)
    emit(rows, key, 4, 'seeds', 'extra seed at full strength (average with the next seed)', rel('calibration', 'extra'), rel('outer', 'extra'), or_cols=own, action=action)
    emit(rows, key, 5, 'seeds', 'two seeds, same calibration (two-amount shrink toward ' + ('A' if 'plate' in tot['outer'] else 'a') + ')',
         rel('calibration', 'two'), rel('outer', 'two'), note='amounts (cell-line, rest) per seed: ' + '; '.join(f'{x:g}, {y:g}' for x, y in amounts['two']),
         or_cols=own, action=action)
    emit(rows, key, 6, 'seeds', 'three-seed average, same shrink (reference)', rel('calibration', 'three'), rel('outer', 'three'),
         note='amounts per seed: ' + '; '.join(f'{x:g}, {y:g}' for x, y in amounts['three']), or_cols=own, action=action)
    per, selections = change_tables(screen, fold, model)
    first = next(iter(per.values()), None)
    if screen == 'tahoe':
        emit(rows, key, 7, 'training', 'continued training, matched (whole network, input/a/x1)', None, continued_x1(out, fold, model), status='copied',
             source='continued-training-readings/CONTINUED_X1.csv', note='full chain only; no calibration value', or_cols=own, action=action)
        emit(rows, key, 8, 'training', 'training longer (own schedule; built models 40k->80k, State 5k->10k)', None,
             [float(first[s]['longer_total_pts']) for s in SEEDS] if first else None,
             status='copied' if first else 'not computed: no saved training-longer grid for this model and fold',
             source='PER_SEED longer_* (training longer)', note='no nested-chain run: no calibration value',
             or_cols=own, action=action)
    elif screen in ('replogle', 'jiang'):
        emit(rows, key, 7, 'training', 'continued training, matched (whole network, input/a/x1)', None,
             [float(first[s]['continuation_total_pts']) for s in SEEDS] if first else None, status='copied',
             source='cards-genetic-screens PER_SEED continuation_*', note='full chain only; no calibration value', or_cols=own, action=action)
        emit(rows, key, 8, 'training', 'training longer (own schedule)', status='not computed: the genetic parents kept no weights past 40k')
    else:
        emit(rows, key, 7, 'training', 'continued training, matched (whole network, input/a/x1)', status='not computed: no real-task runs on this screen')
        emit(rows, key, 8, 'training', 'training longer (own schedule)', status='not computed: no saved later weights on this screen')
    emit(rows, key, 9, 'published models', "published models' raw scores", status='not recomputed: see Fig. 2')
    # field decision methods and the changes
    changes = parse_change(S, grid_paths, train) if screen == 'parse' else per
    pick_by_seen_lines(rows, key, screen, changes, first, own, action)
    for cell in changes:
        ckey = (screen, fold, model, cell)
        cres = None if screen == 'parse' else or_result(screen, fold, model, cell)
        cact = action if screen == 'parse' else or_numbers(cres) if cres else None
        ccols = or_columns(cact, "origin resolution's result with this change in its menu" if screen != 'parse' else
                           "origin resolution's result (the repair in its menu; calibration-support replay)")
        paths = grid_paths(cell)
        half = {}
        for w, chain, side in (('calibration', 'nested', 'train'), ('outer', 'full', 'outer')):
            u = S[w]['u']
            half[w] = []
            for s in SEEDS:
                p, R = grids.load(paths[s][chain], side, u), grids.load(paths[s][f'cell_{chain}'], side, u)
                half[w].append(pts(T(p + .5 * (R - p), u), T(p, u)))
        if screen == 'parse':
            full_cal, full_held = changes[cell]['full_cal'], changes[cell]['full_held']
        else:
            sel = selections[cell]['entries']
            base = lambda s: sel[str(s)]['calibration']['parent']
            full_cal = [pts(sel[str(s)]['calibration']['full_repair'], base(s)) for s in SEEDS]
            full_held = [float(per[cell][s]['cell_total_pts']) for s in SEEDS]
        emit(rows, ckey, 10, 'field decision method', 'the change at full strength', full_cal, full_held,
             status='copied' if screen != 'parse' else 'computed', or_cols=ccols, action=cact)
        emit(rows, ckey, 11, 'field decision method', 'fixed half, no calibration (original + 0.5 (change - original))', half['calibration'], half['outer'],
             or_cols=ccols, action=cact)
        if screen != 'parse':
            sel = selections[cell]
            current = sel['current_action']
            emit(rows, ckey, 12, 'field decision method', f"ordinary calibration over the change's menu (lowest calibration error, no margin): {current}",
                 [pts(sel['entries'][str(s)]['calibration'][current], sel['entries'][str(s)]['calibration']['parent']) for s in SEEDS],
                 [float(per[cell][s]['current_action_total_pts']) for s in SEEDS], status='copied', source='SELECTIONS, PER_SEED current_action_*',
                 or_cols=ccols, action=cact)
            emit(rows, ckey, 13, 'field decision method', 'the change at its calibrated lambda (shrunk at the original model\'s cell-line amount)',
                 [pts(sel['entries'][str(s)]['calibration']['full_repair_at_parent_lam'], sel['entries'][str(s)]['calibration']['parent']) for s in SEEDS],
                 [float(per[cell][s]['cell_at_lam_total_pts']) for s in SEEDS], status='copied', source='SELECTIONS, PER_SEED cell_at_lam_*',
                 or_cols=ccols, action=cact)
        if cact is not None:
            emit(rows, ckey, 40, "origin resolution's action", f"{cact['action']}; kept fraction alpha per seed: "
                 + ', '.join('-' if x is None else f'{x:g}' for x in cact['alpha']), cact['action_cal'], cact['action_held'],
                 status='copied', or_cols=ccols, action=cact)
    # origin resolution's cheap alternatives and its action for the model
    for i, (name, label) in enumerate(CHEAP):
        if name in action['alternatives']:
            cal_v, held_v = action['alternatives'][name]
            emit(rows, key, 20 + i, "origin resolution's cheap alternative", label, cal_v, held_v, status='copied', or_cols=own, action=action)
    emit(rows, key, 30, "origin resolution's action", f"{action['action']}; amounts per seed: "
         + '; '.join(json.dumps(a) for a in action['amounts']), action['action_cal'], action['action_held'], status='copied',
         or_cols=own, action=action)
    print(screen, fold, model, flush=True)


def pick_by_seen_lines(rows, key, screen, changes, first, own, action):
    """Per seed, the lowest training-line error at full strength among the original model, each change and training
    longer (where it exists); its held-out value."""
    if not changes:
        emit(rows, key, 14, 'field decision method', 'pick by seen lines (lowest training-line error)', [0.] * 3, [0.] * 3,
             note='only the original model is a candidate', or_cols=own, action=action)
        return
    picks, held = [], []
    for s in SEEDS:
        cand = {'original model': (0., 0.)}
        for cell, r in changes.items():
            cand[cell] = ((float(r[s]['cell_training_total_pts']), float(r[s]['cell_total_pts'])) if screen != 'parse'
                          else (r['train'][SEEDS.index(s)], r['full_held'][SEEDS.index(s)]))
        if screen == 'tahoe' and first and first[s]['longer_training_total_pts'] not in ('', None):
            cand['training longer'] = (float(first[s]['longer_training_total_pts']), float(first[s]['longer_total_pts']))
        name = min(cand, key=lambda k: cand[k][0])
        picks.append(f'{s}: {name}')
        held.append(cand[name][1])
    emit(rows, key, 14, 'field decision method', 'pick by seen lines (lowest training-line error, full strength)', None, held,
         note='picked ' + '; '.join(picks) + ('' if screen == 'tahoe' else '; training longer not available on this screen'),
         or_cols=own, action=action)


def parse_change(S, grid_paths, train):
    """Parse's repair (the menu's change): full strength on both sides and on the training lines (its real chain)."""
    paths = grid_paths(None)
    full_cal, full_held, seen = [], [], []
    for s in SEEDS:
        for w, chain, side, values in (('calibration', 'nested', 'train', full_cal), ('outer', 'full', 'outer', full_held)):
            u = S[w]['u']
            values.append(pts(T(grids.load(paths[s][f'cell_{chain}'], side, u), u), T(grids.load(paths[s][chain], side, u), u)))
        seen.append(pts(T(grids.load(paths[s]['cell_full'], 'train', train), train), T(grids.load(paths[s]['full'], 'train', train), train)))
    return {'the repair (step 8000)': {'full_cal': full_cal, 'full_held': full_held, 'train': seen}}


def table_stage(out):
    parse_menu = {}
    for r in read_csv(Path(out) / 'PARSE_MENU.csv'):
        parse_menu.setdefault(r['card'], []).append(r)
    rows = []
    for case in CASES:
        build_case(*case, out, parse_menu, rows)
    write_csv(Path(out) / 'BASELINES_SIDE_BY_SIDE.csv', rows)


STAGES = {'parse-replay': parse_replay_stage, 'table': table_stage}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The field's baselines beside origin resolution's alternatives and action.")
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    STAGES[args.stage](args.out)


if __name__ == '__main__':
    main()
