"""Recovery and cost of one cell that information localization hands on, from loaded predictions.

A cell is IL's map recipe behind a cut (``<cut>/<arm>/x<rate multiple>``), handed on for one or more planted targets.
Its real-task runs give, per seed, the parent trained on real labels and the cell's end on the same lines (the parent
on every training line; the nested parent's runs give the calibration entries). Over the seeds:

  Q1 Did the planted target recover?  IL's synthetic reading: setting 1 (synthetic parent) and setting 2 (natural parent,
     the bridge), every seed (``recovery``, ``q1_text``).
  Q2 Did real error on new lines improve?  The cell's end against its own parent, pooled over seeds (paired line
     bootstrap, margin 0.2 points) with the seed rule beside it (``change``); training lines beside it.
  Q3 Where does the change lie?  Every comparison split into the cell-line-level (context-common) and drug-difference
     parts, pooled and per line; beside it the diagnosis of the full-strength change on the calibration lines against
     the same reading on the held-out lines.
  Q4 Does it beat continued training and simple calibration?  (a) continued training of the parent on its own recipe
     for as long as a cell run (site and recipe together); (b) the parent's cell-line-level shrink and the uniform
     shrink, chosen on the calibration lines; (c) the cell and continued training, both shrunk at that amount, with a
     sensitivity favouring continued training (its best held-out amount); (d) the cell shrunk at that amount against
     simple calibration.

Credit: improved with the target recovered in none of the evaluable map runs -> the intervention's own effect;
recovered in all -> not separated between reuse of the cue and the recipe; otherwise undetermined, or no gain to credit.
"""
import numpy as np

from .comparison import compare, seed_rule
from .policies import G, case_labels, labels, predict
from .scoring import PARTKEY, line_parts, scores

__all__ = ["AVERAGED", "SETTINGS", "q1_text", "recovery", "change", "credit", "fmt", "text", "diagnosis", "protocol", "score"]

AVERAGED = ('prepr', 'cpa')   # read out as the final-quarter weight mean
RECIPE = {'prepr': 'Adam, constant, weight decay 1e-5', 'cpa': "Adam, constant, CPA's own weight decay"}
DEFAULT_RECIPE = 'fresh AdamW, 1,000-update warm-up, cosine to a tenth'
SETTINGS = (('synthetic', 'setting 1 (synthetic parent)'), ('natural', 'setting 2 (natural parent, bridge)'))
CALIBRATED = {'cheapest_fix', 'uniform_shrink', 'cell_at_lam', 'current_action', 'continuation_at_lam'}
COMPARED = {'continuation', 'continuation_at_lam'}
PAIRS = {'Q2 cell - parent': ('cell', 'parent'),
         'Q4a cell - continued training': ('cell', 'continuation'),
         'Q4b cell - simple calibration': ('cell', 'cheapest_fix'),
         'Q4b cell - uniform shrink': ('cell', 'uniform_shrink'),
         'Q4c cell at lambda - continued training at lambda': ('cell_at_lam', 'continuation_at_lam'),
         'Q4d cell at lambda - simple calibration': ('cell_at_lam', 'cheapest_fix'),
         'current action - parent': ('current_action', 'parent'),
         'current action - simple calibration': ('current_action', 'cheapest_fix'),
         'current action - uniform shrink': ('current_action', 'uniform_shrink'),
         'current action - continued training at lambda': ('current_action', 'continuation_at_lam')}


def q1_text(found, seeds):
    """IL's outcome per setting and seed (``found``: (half, seed) -> the map row)."""
    def one(r):
        if r is None:
            return 'not in table'
        if r['outcome'] == 'diverged':
            return 'diverged (excluded)'
        return f"{r['outcome']} {float(r['target_Q']):.2f} [{float(r['target_low']):.2f}, {float(r['target_high']):.2f}]"
    return '; '.join(name + ': ' + ', '.join(f'{seed} {one(found.get((half, seed)))}' for seed in seeds)
                     for half, name in SETTINGS)


def recovery(found, seeds):
    """One reading over both settings and every seed; a diverged map run is excluded, as in IL's table."""
    if any((half, seed) not in found for half, _ in SETTINGS for seed in seeds):
        return 'incomplete'
    outcomes = [found[(half, seed)]['outcome'] for half, _ in SETTINGS for seed in seeds]
    outcomes = [o for o in outcomes if o != 'diverged']
    if not outcomes:
        return 'incomplete'
    if all(o.startswith('recovered') for o in outcomes):
        return 'recovered'
    if all(o == 'not recovered' for o in outcomes):
        return 'not recovered'
    return 'partial or mixed'


def change(comparison, below):
    """Q2's reading from the pooled call and the seeds below the parent."""
    t, rule = comparison['total'], seed_rule(below)
    if t['call'] == 'different' and t['estimate_pct'] < 0 and rule == 'holds':
        return 'improved'
    if t['call'] == 'different' and t['estimate_pct'] > 0 and rule == 'no':
        return 'worse'
    if t['call'] == 'no different':
        return 'no change'
    return 'unresolved'


def credit(improvement, recovered, model):
    if improvement == 'unresolved':
        return 'undetermined: the real-error change is unresolved'
    if improvement != 'improved':
        return 'no gain to credit'
    averaged = ' (with the weight-averaging read-out; Q4a is matched in read-out)' if model in AVERAGED else ''
    if recovered == 'not recovered':
        return ("the intervention's own effect" + averaged + ': real error improved while the target recovered in none of '
                'the evaluable map runs of settings 1 and 2')
    if recovered == 'recovered':
        return ('not separated' + averaged + ': the target recovered and real error improved; reuse of its cue is not '
                'separated from the recipe')
    return f'undetermined: target recovery {recovered}'


def fmt(r):
    return '{e:+.2f} [{lo:+.2f}, {hi:+.2f}]'.format(e=r['estimate_pct'], lo=r['interval_pct'][0], hi=r['interval_pct'][1])


def text(comparison, below):
    t = comparison['total']
    return (f"total {fmt(t)} {t['call']}, null {t['null_different_rate']:.3f}; common {fmt(comparison['common'])}; "
            f"differential {fmt(comparison['differential'])}; seeds below: {seed_rule(below)}")


def diagnosis(per_seed):
    c = case_labels(per_seed)
    return f"cost {c['d1a']}, size {c['d1c']}, common part damaged at every amount: {c['d1b_common_damaged']}"


def protocol(model, record, continued, paired):
    """The cell's and the comparator's protocols, from the first seed's run record on every training line."""
    cell = (f"cell: IL's map recipe behind the cut ({RECIPE.get(model, DEFAULT_RECIPE)}), {record['updates']:,} updates at "
            f"x{float(record['row']['rate_multiple']):g} the parent's rate (peak {record.get('peak_lr')}), trained "
            f"{', '.join(record.get('trained_prefixes') or ['not recorded'])}, read out "
            + ('as the final-quarter weight mean' if model in AVERAGED else 'at the last update'))
    note = ('; the parent is one checkpoint, so Q2, Q4b and Q4d include the averaging gain'
            + (', and Q4a and Q4c are matched in read-out' if paired else '') if model in AVERAGED else '')
    continued = continued if paired else f'not available ({continued})'
    return f'{cell}; continued training: {continued}; Q4a compares site and recipe together{note}'


def score(fold, model, cell, record, held, train, seed_inputs, seeds, saved, missing, continued, paired):
    """Q2-Q4, the current action and the diagnosis check of one cell with complete runs on every training line.
    ``seed_inputs(seed)``: the seed's predictions, ``p``, ``a`` (anchor) and ``R`` (the cell's end) on the
    held-out units ``held``, ``continuation`` there where ``paired``, and under ``train`` functions that read the
    parent's, the cell's and (where paired) continued training's predictions on the training units ``train``.
    ``saved``: the cell's saved selection, or None (``missing`` says why); ``continued``: what continued training is,
    or why it is missing. Returns the answers, Q2's reading, and the per-seed and per-line rows."""
    seed_rows, line_rows = [], []
    action = saved['current_action'] if saved else None
    parts, cont_lam, outer_labels = {}, {x: [] for x in G}, []
    for seed in seeds:
        x = seed_inputs(seed)
        p, a, R = x['p'], x['a'], x['R']
        preds = {'parent': p, 'cell': R}
        if saved:
            amount = saved['entries'][str(seed)]['amount']
            lam = amount['cheapest_fix']
            chosen = amount[action]
            preds |= {'cheapest_fix': predict('cheapest_fix', lam, p, R, a, held),
                      'uniform_shrink': predict('global_parent_shrink', amount['global_parent_shrink'], p, R, a, held),
                      'cell_at_lam': predict('full_repair_at_parent_lam', (None, lam), p, R, a, held),
                      'current_action': predict(action, tuple(chosen) if isinstance(chosen, list) else chosen, p, R, a, held)}
        base = scores(p, held)
        if paired:
            cont = x['continuation']
            preds['continuation'] = cont
            if saved:
                preds['continuation_at_lam'] = predict('cheapest_fix', lam, cont, R, a, held)
                for y in G:
                    cont_lam[y].append((line_parts(predict('cheapest_fix', y, cont, R, a, held), held), base['total_mse']))
        s = {k: scores(v, held) for k, v in preds.items()}
        tp = scores(x['train']['parent'](), train)
        tt = {k: scores(read(), train) for k, read in x['train'].items() if k != 'parent'}
        outer_labels.append(labels(p, R, held, np.ones(len(held['truth']), bool)))
        seed_rows.append({'fold': fold, 'model': model, 'cell': cell, 'seed': seed,
                          'lambda': lam if saved else '', 'current_action': action or '',
                          **{f'{k}_{part}_pts': 100 * (s[k][key] - base[key]) / base['total_mse']
                             for k in preds if k != 'parent' for part, key in PARTKEY.items()},
                          **{f'{k}_training_{part}_pts': 100 * (t[key] - tp[key]) / tp['total_mse']
                             for k, t in tt.items() for part, key in PARTKEY.items()}})
        for k, v in preds.items():
            parts.setdefault(k, []).append((line_parts(v, held), base['total_mse']))
        own, parent = parts['cell'][-1][0], parts['parent'][-1][0]
        continued_parts = parts['continuation'][-1][0] if paired else None
        line_rows += [{'fold': fold, 'model': model, 'cell': cell, 'seed': seed, 'line': line, 'part': part,
                       'cell_minus_parent_pts': 100 * (own[part][line] - parent[part][line]) / base['total_mse'],
                       'cell_minus_continuation_pts': (100 * (own[part][line] - continued_parts[part][line]) / base['total_mse']
                                                       if paired else '')}
                      for part in PARTKEY for line in own[part]]
    totals = lambda runs_: [sum(x[0]['total'].values()) for x in runs_]
    below = lambda x, y: [u < v for u, v in zip(totals(x), totals(y))]

    def unavailable(x, y):
        needs = {x, y} - set(parts)
        return 'not available: ' + '; '.join(([continued] if needs & COMPARED and not paired else [])
                                             + ([missing] if needs & CALIBRATED and not saved else []))

    q2, q2_below = compare(parts['cell'], parts['parent']), below(parts['cell'], parts['parent'])
    improvement = change(q2, q2_below)
    mine = seed_rows
    calibration = diagnosis([saved['entries'][str(s)]['labels'] for s in seeds]) if saved else f'not available: {missing}'
    outer = diagnosis(outer_labels)
    agree = (case_labels([saved['entries'][str(s)]['labels'] for s in seeds])['d1a'] == case_labels(outer_labels)['d1a']
             if saved else None)
    card = {'status': 'complete' if saved else f'partly complete: {missing}',
            'protocols': protocol(model, record, continued, paired),
            'Q2_real_error': f'{improvement}; ' + text(q2, q2_below) + '; by seed '
                             + ' / '.join(f"{x['cell_total_pts']:+.2f}" for x in mine),
            'Q2_training_lines': ' ; '.join(k + ' ' + ' / '.join(
                f"{x[f'{k}_training_total_pts']:+.2f} (common {x[f'{k}_training_common_pts']:+.2f}, "
                f"differential {x[f'{k}_training_differential_pts']:+.2f})" for x in mine) for k in ('cell', 'continuation')
                if f'{k}_training_total_pts' in mine[0]),
            'Q3_where': f"common {fmt(q2['common'])}, differential {fmt(q2['differential'])}; by seed "
                        + ' / '.join(f"common {x['cell_common_pts']:+.2f}, differential {x['cell_differential_pts']:+.2f}" for x in mine),
            'Q3_diagnosis_calibration_vs_outer': f'calibration lines: {calibration} | outer lines: {outer}'
                                                 + ('' if agree is None else ' | cost reading ' + ('agrees' if agree else 'differs')),
            **{name: text(compare(parts[x], parts[y]), below(parts[x], parts[y])) if x in parts and y in parts
               else unavailable(x, y) for name, (x, y) in PAIRS.items() if not name.startswith('Q2')},
            'Q4 another site at the same dose': 'not available: IL runs only the carried cells in setting 3',
            'lambda_by_seed': ' / '.join(f'{x["lambda"]:g}' for x in mine) if saved else f'not available: {missing}',
            'current_action': action or f'not available: {missing}'}
    if paired and saved:
        best = min(G, key=lambda x: np.mean(totals(cont_lam[x])))
        card['Q4c sensitivity favouring continued training'] = (
            f'continued training at its best outer lambda {best:g}: '
            + text(compare(parts['cell_at_lam'], cont_lam[best]), below(parts['cell_at_lam'], cont_lam[best])))
    return card, improvement, seed_rows, line_rows
