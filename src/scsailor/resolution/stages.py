"""The attribution card's stages over one dataset's case: what the card computes, without reading or writing files.

A *case* is one model on one fold of a dataset, as a dataset's binding builds it:

- ``units``: ``'fit'``, ``'calibration'`` (the training units of the lines the nested parent did not see),
  ``'outer'`` (held-out units, full parent) and ``'outer_nested'`` (held-out units eligible for the nested parent),
  each a units dictionary (``scoring``);
- ``load(seed, chain, side, units, member='raw')``: a seed's prediction on ``units`` (``side`` ``'train'`` or
  ``'outer'``); ``chain`` is ``'nested'`` (the parent fitted without the calibration lines), ``'full'`` (the parent
  on every training line) or ``'cell_full'`` (the end of information localization's repair on the full parent);
  ``member`` is ``'raw'`` or ``'anchor_raw'`` (the model's anchor: the input bag's mean plus its transferred response);
- ``bags``: side (``'train'``, ``'heldout'``) -> context -> (the input bag's mean, the control pool's mean);
- ``noise``: unit set -> each unit's cell-sampling variance of the scored target (D1), or None where not computed;
- ``meta``: ``coincident`` (factors that coincide with identity by design, or None), ``type_variable`` and
  ``composition`` (notes the rows and readings print), and ``target_note`` (what the headline adds when a verified
  recommendation shrinks toward a design-aware target, or None);
- ``gene_variance``: the per-gene cell variance of the training units' scored targets (their treated cells), averaged
  over units, which spreads the cell-sampling floor over genes; or None;
- ``static_rows``: the dataset's documented origin rows (``static``), those whose evidence is fold 1's
  (``fold1_only``) and what the O1 row says of the seed-permutation statistic (``seed_permutation``);
- ``replicates`` (where each held-out line is measured on two replicates, one context per line and replicate, and
  each unit has one twin): per held-out unit its ``line``, its replicate ``half`` (0 or 1) and its ``twin`` in the
  other replicate (-1 where missing), with a ``label`` naming the replicates and the reference; or None;
- ``fold``.

Stages:

- ``selection``: an action chosen once, without repair candidates, for a model with no earlier selection (official
  State): the cell-line-level shrink when it is below the parent in every seed, else the parent;
- ``calibration``: every removal and cheap alternative calibrated on the calibration units, the menu's items added
  and the full-menu rule applied, the recommendation condition decided, and, where perturbation groups are declared,
  the group to be tested chosen. Nothing here reads held-out truth;
- ``attribution``: the parent, the action and every removal on the held-out units; verification against the cheap
  alternatives; the contrasts M5, M7 and N by the unit-level test and their Shapley shares; the headline; one row
  per origin; the "where", the references and the three readings of the nested parent.

``repair_entry`` and ``parent_entry`` are the per-seed calibration entries of the earlier selections (a model with a
repair, such as an IL cell's; a parent-only menu), which the card reads as its saved selection.
"""
import numpy as np

from . import attribution, card, comparison, menu, origins, policies
from .references import reliability
from .scoring import PARTKEY, PARTS, gene_contributions, identities, scores, unit_parts

__all__ = ["PARTKEY", "LABEL", "SELECTION_RULE", "bag_of", "net", "choose_group", "repair_alpha", "selection",
           "calibration", "calibration_message", "action_prediction", "gene_groups", "four_part_note", "composition",
           "attribution_stage", "repair_entry", "parent_entry", "parent_selection"]

LABEL = {'total': 'total', 'common': 'common', 'differential': 'drug difference'}
# the wording of the saved records
SELECTION_RULE = "round 12's default without repair candidates: the cell-line-level shrink if below the parent in every seed"
GROUP_RULE = 'largest drug-difference excess of the nested parent on the calibration units, seed mean'


def bag_of(bags, units):
    """Per-unit base b (the input bag's mean, the anchor's base at L1 and above) and pool mean C, from one side's
    bags."""
    return (np.stack([bags[c][0] for c in units['context']]), np.stack([bags[c][1] for c in units['context']]))


def net(parts, floor):
    """Per-unit contributions net of their floor where one exists (tests read the net excess)."""
    return {k: parts[k] - floor[k] for k in parts} if floor is not None else parts


def choose_group(cal_parts, cal, cal_floor, mapping):
    """Choose, then test: the group with the largest drug-difference excess of the nested parent's
    seed-mean contributions on the calibration units, net of their floor where it exists."""
    excess = attribution.group_excess(net(cal_parts, cal_floor), cal, mapping)
    return max(excess, key=excess.get), excess


def repair_alpha(saved):
    """Seed -> origin resolution's calibrated size of the IL repair in a saved selection (its size-fix amount), or
    None where the selection has no repair."""
    return {k: saved['entries'][k]['amount'].get('size_fix') for k in saved['entries']}


def selection(case, seeds):
    """The action of a model with no earlier selection (official State), chosen once before its calibration stage:
    the earlier decision-table rule without repair candidates (``policies.diagnose``), the cell-line-level shrink when
    its calibration total is below the parent's in every seed, else the parent. Returns the per-seed entries and the
    action."""
    cal = case['units']['calibration']
    b, _ = bag_of(case['bags']['train'], cal)
    entries = {}
    for seed in seeds:
        e = card.calibration_entry(case['load'](seed, 'nested', 'train', cal), case['load'](seed, 'nested', 'train', cal, 'anchor_raw'), b, cal)
        entries[str(seed)] = {'amount': {k: v['amount'] for k, v in e.items()}, 'calibration': {k: v['calibration'] for k, v in e.items()}}
    below = all(e['calibration']['cheapest_fix'] < e['calibration']['parent'] for e in entries.values())
    return entries, 'cheapest_fix' if below else 'parent'


def calibration(case, saved, seeds, side, mapping):
    """The calibration stage on the calibration units. ``saved``: the saved
    selection (``current_action`` and per seed its ``amount`` and ``calibration``); ``side``: the menu's calibration
    side (``menu.side``, with ``repair_alpha(saved)``); ``mapping``: the declared perturbation grouping (p_code ->
    group), or None. Returns the calibration record without the dataset's labels and files. The recomputed parent
    calibration total must equal the saved one."""
    action = saved['current_action']
    cal = case['units']['calibration']
    b, _ = bag_of(case['bags']['train'], cal)
    entries, action_cal, cal_parts = {}, {}, []
    for seed in seeds:
        p, a = case['load'](seed, 'nested', 'train', cal), case['load'](seed, 'nested', 'train', cal, 'anchor_raw')
        entry = card.calibration_entry(p, a, b, cal)
        old = saved['entries'][str(seed)]['calibration']
        if not np.isclose(entry['parent']['calibration'], old['parent'], rtol=1e-9, atol=0):
            raise ValueError(f"seed {seed}: parent calibration {entry['parent']['calibration']} differs from the "
                             f"saved {old['parent']}: units or grid differ from the saved selection")
        entries[str(seed)] = entry
        action_cal[str(seed)] = old[action]
        cal_parts.append(unit_parts(p, cal))
    # the menu's added items and the full-menu rule over them and the earlier action; a published model's option
    # (corrected by one in-house model) is chosen and branched beside it
    alpha = repair_alpha(saved)
    menu.add_items(entries, side)
    previous = action
    action, normalized, option, option_normalized = menu.choose(entries, previous, action_cal, side)
    if action != previous:
        action_cal = {s: entries[s][action]['calibration'] for s in entries}
    alternatives = card.ALTERNATIVES + (menu.PLATE_ALTERNATIVES if side['A'] is not None else ())
    decision = card.branch(entries, action, action_cal, alternatives=alternatives)
    option_record = None
    if option is not None:
        option_record = {'action': option, 'normalized_calibration': option_normalized,
                         'amounts': {s: entries[s][option]['amount'] for s in entries},
                         'branch': card.branch(entries, option, {s: entries[s][option]['calibration'] for s in entries}, alternatives=alternatives)}
    group = None
    if mapping is not None:
        parts = {part: np.mean([x[part] for x in cal_parts], axis=0) for part in PARTS}
        floor = attribution.floor_parts(cal, case['noise']['calibration']) if case['noise'].get('calibration') is not None else None
        chosen, excess = choose_group(parts, cal, floor, mapping)
        group = {'chosen': chosen, 'basis': 'net of the floor' if floor is not None else 'raw', 'rule': GROUP_RULE,
                 'calibration_excess': excess}
    return {'outer_truth_read': False, 'outer_units_built': False,
            'current_action': action, 'previous_action': previous, 'menu_normalized_calibration': normalized,
            'cheap_alternatives': list(alternatives),
            'menu_inputs': {'repair_alpha': alpha, 'items_available': menu.available(side), 'plate_from_bag': side['plate_check']},
            'option': option_record,
            'action_amounts': ({s: saved['entries'][s]['amount'][action] for s in saved['entries']} if action == previous
                               else {s: entries[s][action]['amount'] for s in entries}),
            'action_calibration': action_cal, 'entries': entries, 'branch': decision, 'perturbation_group': group,
            'calibration_units': len(cal['truth']), 'margin_pts': card.MARGIN, 'grid_G': list(policies.G)}


def calibration_message(record, seeds):
    """The calibration stage's printed summary after the dataset's labels: action, amounts, recommendation, group."""
    entries, group = record['entries'], record['perturbation_group']
    return (f"action {record['current_action']}; amounts "
            + ', '.join(f"{k} {[entries[str(s)][k]['amount'] for s in seeds]}" for k in ('cheapest_fix', 'drug_shrink', 'response_shrink', 'nochange_shrink', 'global_parent_shrink'))
            + f"; recommendation {record['branch']['recommendation']}; perturbation group {group['chosen'] if group else 'none declared'}")


def action_prediction(case, saved, seed, p, a, units):
    """The saved action on the held-out units for one seed, where it is not a menu item (menu.predict serves those).
    A repair action reads the IL cell's end on the full parent."""
    action, amount = saved['current_action'], saved['action_amounts'][str(seed)]
    if action in menu.MENU:
        raise ValueError(f'{action} is a menu item: predict it with menu.predict on menu.side(...)')
    R = None
    if action not in ('parent', 'cheapest_fix', 'global_parent_shrink'):
        R = case['load'](seed, 'cell_full', 'outer', units)
    return policies.predict(action, tuple(amount) if isinstance(amount, list) else amount, p, R, a, units)


def gene_groups(arrays):
    """Gene groups from training data only, the same rule on every dataset: quintiles of the training control
    mean, of the training control-cell variance, and of the across-line variability of the training response.
    ``arrays``: ``c_train`` and ``context_keys`` (model constants), ``target``, ``parent`` and ``codes`` (training
    units), ``variance`` (per training context, its control cells' gene variances) and ``pools`` (context index ->
    control pool mean)."""
    c_train, keys = arrays['c_train'], arrays['context_keys']
    target, parent, codes = arrays['target'], arrays['parent'], arrays['codes']
    var, pools = arrays['variance'], arrays['pools']
    response = target - np.stack([pools[int(q)] for q in parent])
    lines = np.asarray([keys[q].split('|')[0] for q in parent])
    spread = []
    for code in np.unique(codes):
        m = codes == code
        per_line = [response[m & (lines == l)].mean(axis=0) for l in np.unique(lines[m])]
        if len(per_line) > 1:
            spread.append(np.var(np.stack(per_line), axis=0, ddof=1))
    quintile = lambda v: np.searchsorted(np.quantile(v, [.2, .4, .6, .8]), v, side='right')
    return {'control mean quintile': quintile(c_train), 'control variance quintile': quintile(np.mean(var, axis=0)),
            'across-line response variability quintile': quintile(np.mean(spread, axis=0))}


def four_part_note(units):
    """Where the design makes a part of the four-part split zero by structure: not evidence of absence."""
    ident = identities(units)
    notes = []
    if len(np.unique(ident)) == 1:
        notes.append('one identity: the identity x drug part is zero by structure')
    if len(set(zip(ident, units['p_code']))) == len(ident):
        notes.append('one unit per identity and perturbation: the within-identity residual is zero by structure')
    return '; '.join(notes) + ('; not identifiable here, so a zero is not evidence of absence' if notes else '')


def composition(units):
    """The identity mix of a reading: each identity's share of the scoring weight."""
    ident = identities(units)
    w = units['weight'] / units['weight'].sum()
    return ', '.join(f'{l} {w[ident == l].sum():.2f}' for l in np.unique(ident))


def attribution_stage(case, saved, seeds, side, evidence, genes, mapping):
    """The attribution stage on the held-out units.

    ``saved``: the calibration record; ``side``: the menu's outer side (``menu.side`` with the saved repair sizes);
    ``evidence(case, held, mean_parts, pool, parent_part, preds)``: the dataset's evidence for the computed rows, a
    dictionary with ``d2`` and ``d7`` (each a row's ``status`` and ``evidence``), optionally ``d4``, ``d8``, ``t1``,
    ``t7`` and ``d2_by_context``, and its ``reference_rows`` and ``where_rows``; ``genes``: ``gene_groups`` of the
    training data; ``mapping``: the declared perturbation grouping, or None.

    Returns the headline, origin rows, "where" rows, reference rows and readings (the card's five tables) and the run
    record's contents without the dataset's labels and files."""
    held = case['units']['outer']
    b, C = bag_of(case['bags']['heldout'], held)
    amounts = {s: {k: v['amount'] for k, v in saved['entries'][str(s)].items()} for s in seeds}
    preds = {k: [] for k in ('parent', 'action', 'anchor', 'nochange', 'uniform', 'nochange_alt')}
    action_name = saved['current_action']
    option = saved.get('option')
    removal, scales = {}, []
    for seed in seeds:
        p, a = side['p'][seed], side['a'][seed]   # the full parent and its anchor on the held-out units
        rem = card.removals(p, a, b, held, amounts[seed])
        for key, v in rem.items():
            removal.setdefault(key, []).append(v)
        preds['parent'].append(p)
        preds['action'].append(menu.predict(action_name, saved['action_amounts'][str(seed)], seed, side) if action_name in menu.MENU
                               else action_prediction(case, saved, seed, p, a, held))
        if side['A'] is not None:   # the added cheap alternatives with a plate-aware target, at their calibrated amounts
            for ref in menu.PLATE_ALTERNATIVES:
                preds.setdefault(ref, []).append(menu.predict(ref, saved['entries'][str(seed)][ref]['amount'], seed, side))
        if option:
            preds.setdefault('option', []).append(menu.predict(option['action'], option['amounts'][str(seed)], seed, side))
        preds['anchor'].append(a)
        preds['nochange'].append(b)
        preds['uniform'].append(rem['uniform'])
        preds['nochange_alt'].append(rem['nochange_alt'])
        scales.append(scores(p, held)['total_mse'])
    for player in card.PLAYERS:
        preds[player] = removal[frozenset({player})]
    R = {k: card.runs(v, held, scales) for k, v in preds.items()}
    seeds_mean = lambda f: float(np.mean([f(i) for i in range(len(seeds))]))
    part_mean = lambda xs, key: seeds_mean(lambda i: scores(xs[i], held)[key])

    # verification of the action
    verification = {}
    references = ((('parent', 'parent'), ('cheapest_fix', 'M5'), ('global_parent_shrink', 'uniform'), ('nochange_shrink', 'nochange_alt'))
                  + tuple((ref, ref) for ref in menu.PLATE_ALTERNATIVES if ref in preds))
    for ref, key in references:
        if ref != saved['current_action']:
            verification[ref] = comparison.detect(R['action'], R[key])
    recommended = saved['branch']['recommendation']
    verified = recommended and all(v['seed_rule'] == 'holds' for v in verification.values())
    option_verification = None
    if option:   # the published model corrected by one in-house model: the same references, and the card's own action
        option_verification = {ref: comparison.detect(R['option'], R[key]) for ref, key in references + (('action', 'action'),)
                               if ref != option['action']}

    # contrasts, sharing and the headline
    detections = {r: comparison.detect(R[r], R['parent']) for r in card.PLAYERS}
    separation = comparison.detect(R['M5'], R['M7'])
    separation['reverse'] = comparison.detect(R['M7'], R['M5'])
    case_level = any(d['level'] == 'case-level' for d in detections.values())
    players = list(card.PLAYERS) if case_level else [r for r in card.PLAYERS if detections[r]['detected']]
    joint = removal[frozenset(players)]
    shares = {}
    for part, key in PARTKEY.items():
        def value(s, key=key):
            if not s:
                return 0.
            return seeds_mean(lambda i: 100 * (scores(removal[frozenset(s)][i], held)[key] - scores(preds['parent'][i], held)[key]) / scales[i])
        shares[part] = card.shapley(value, players)
    parent_part = {part: part_mean(preds['parent'], key) for part, key in PARTKEY.items()}
    line2_parts = {}
    for part, key in PARTKEY.items():   # line 2 per part: the joint removal against the lower reference of that part
        ref_value = {name: part_mean(preds[name], key) for name in ('anchor', 'nochange')}
        best = min(ref_value, key=ref_value.get)
        line2_parts[part] = {'parent': parent_part[part], 'joint removal': part_mean(joint, key), **ref_value, 'best reference': best,
                             'joint minus best (pts of the parent total)': seeds_mean(lambda i, key=key, best=best: 100 * (
                                 scores(joint[i], held)[key] - scores(preds[best][i], held)[key]) / scales[i])}
    n = detections['N']
    line0 = (f"Repair recommendation verified: {saved['current_action']}, below the parent and every cheap alternative in every seed"
             if verified else ('Recommendation not confirmed on the outer units' if recommended else 'No recommendation: attribution card'))
    if verified and side['A'] is not None and saved['current_action'] in menu.MENU:
        line0 += case['meta']['target_note']
    line4 = None
    if option:
        option_ok = option['branch']['recommendation'] and all(v['seed_rule'] == 'holds' for k, v in option_verification.items() if k != 'action')
        line4 = (f"Option for a published model: {option['action']}, {'verified' if option_ok else 'not verified'}; "
                 f"{option_verification['parent']['total_pts']:+.2f} pts against the parent, "
                 f"{option_verification['action']['total_pts']:+.2f} against the card's action")
    line1 = card.line_one({k: detections[k] for k in ('M5', 'M7')}, separation)
    line2 = (f"parent {1e3 * parent_part['total']:.3f}e-3; Shapley shares ({'case-level, all three' if case_level else 'detected only'}): "
             + ', '.join(f"{q} {shares['total'].get(q, 0.):+.2f}" for q in card.PLAYERS)
             + f" pts; joint removal at calibrated amounts ({', '.join(players) if players else 'none'}) {1e3 * line2_parts['total']['joint removal']:.3f}e-3; "
             + 'against the lower reference of each part, in points of the parent total: '
             + ', '.join(f"{LABEL[part]} {v['joint minus best (pts of the parent total)']:+.2f} ({v['best reference']})" for part, v in line2_parts.items())
             + f"; transferred response N {n['total_pts']:+.2f} pts ({n['level']}{', detected' if n.get('detected') else ''}; candidates T7, T1, D6, not separated)")

    # references on the outer units
    floor = attribution.floor_parts(held, case['noise']['outer']) if case['noise'].get('outer') is not None else None
    gap = attribution.bag_gap(held, b, C)
    ref_rows, references = [], {}
    ref_rows.append({'reference': 'parent (seed mean)', **parent_part})
    ref_rows.append({'reference': f"joint removal at calibrated amounts ({', '.join(players) if players else 'none'}; seed mean)",
                     **{part: v['joint removal'] for part, v in line2_parts.items()}})
    for name in ('anchor', 'nochange'):
        references[name] = {part: line2_parts[part][name] / parent_part[part] for part in PARTKEY}
        ref_rows.append({'reference': name, **{part: line2_parts[part][name] for part in PARTKEY}})
        ref_rows.append({'reference': f'joint removal minus {name} (pts of the parent total)', **{part: seeds_mean(
            lambda i, key=key, name=name: 100 * (scores(joint[i], held)[key] - scores(preds[name][i], held)[key]) / scales[i]) for part, key in PARTKEY.items()}})
    s = scores(attribution.oracle_context_mean(held), held)
    ref_rows.append({'reference': 'oracle context mean (transductive)', **{part: s[key] for part, key in PARTKEY.items()}})
    references['T2'] = {part: float(gap[part].sum()) / parent_part[part] for part in PARTKEY}
    ref_rows.append({'reference': 'T2 input-sampling (bag base against pool)', **{part: float(v.sum()) for part, v in gap.items()}})
    if floor is not None:
        references['D1'] = {part: float(floor[part].sum()) / parent_part[part] for part in PARTKEY}
        ref_rows.append({'reference': 'D1 cell-sampling floor', **{part: float(v.sum()) for part, v in floor.items()}})

    # which kind: alignment with its references, full parent on the outer units. Where each held-out line is measured on
    # two replicates, the square root of the true correction's split-half reliability between them is the cosine that a
    # predictor blind to the replicate can reach with one replicate's target. It is a
    # reference, not a ceiling: a model that reads the replicate from its input (Tahoe's plate from the bag) can exceed
    # it. The input bag's ceiling of the first cards stands beside it.
    unmatch = {'common': float(gap['common'].sum()), 'differential': float(floor['differential'].sum()) if floor is not None else 0.}
    align = [card.alignment(preds['parent'][i], preds['anchor'][i], held, unmatch) for i in range(len(seeds))]
    mean_al = lambda part, k: float(np.mean([x[part][k] for x in align]))
    rep = case.get('replicates')
    if rep is not None:
        ceilings = [reliability(held['truth'] - preds['anchor'][i], held, rep['line'], rep['half'], rep['twin']) for i in range(len(seeds))]
        reproducible = lambda part: float(np.mean([c[part]['sqrt_R'] for c in ceilings]))
        of_it = lambda part: float(np.mean([x[part]['cosine'] / c[part]['sqrt_R'] for x, c in zip(align, ceilings)]))
    against = lambda part: (f"against the target's reproducibility {rep['label']} of {reproducible(part):.3f} ({of_it(part):.3f} of it) "
                            f"and the input bag's ceiling of {mean_al(part, 'ceiling'):.3f} ({mean_al(part, 'cosine_over_ceiling'):.3f} of it)"
                            if rep is not None else
                            f"against a ceiling of {mean_al(part, 'ceiling'):.3f} ({mean_al(part, 'cosine_over_ceiling'):.3f} of it)")
    kind = {r: (f"{part}: cosine {mean_al(part, 'cosine'):.3f} {against(part)}; parallel ratio {mean_al(part, 'parallel_ratio'):.3f} "
                f"({mean_al(part, 'parallel_ratio_matchable'):.3f} of the matchable part); best scale {mean_al(part, 'best_scale'):.3f}; "
                f"misdirected share {mean_al(part, 'misdirected_share'):.3f}" + ('' if floor is not None or part == 'common' else ' (no D1 floor: ceiling 1)'))
            for r, part in (('M5', 'common'), ('M7', 'differential'))}

    # rows and line 3
    parent_parts = [unit_parts(preds['parent'][i], held) for i in range(len(seeds))]
    mean_parts = {part: np.mean([x[part] for x in parent_parts], axis=0) for part in PARTS}
    ev = {'detections': detections, 'kind': kind, 'shares': shares, 'references': references,
          'units': detections['M5']['units'], 'seed_spread_pct': 100 * (max(scales) - min(scales)) / float(np.mean(scales)),
          'fold': case['fold'], **evidence(case, held, mean_parts, C, parent_part, preds)}
    ref_rows += ev['reference_rows']
    rows_static = case['static_rows']
    ev['rows'] = origins.rows(ev, rows_static['static'], rows_static['fold1_only'], rows_static['seed_permutation'])
    line3 = origins.line_three(ev, case['meta']['coincident'])

    # the "where"
    where_rows = []
    ident = identities(held)
    structure = four_part_note(held)
    for name in ('parent', 'action', 'M5', 'M7', 'N', 'anchor', 'nochange'):
        for i, seed in enumerate(seeds):
            parts = unit_parts(preds[name][i], held)
            for grouping, keys in (('identity', ident), ('context', held['context'])):
                where_rows += [{'seed': seed, 'grouping': grouping, 'predictor': name, **r} for r in attribution.where(parts, held, keys, scales[i], floor)]
            if name in ('parent', 'action'):
                where_rows += [{'seed': seed, 'grouping': 'four parts', 'predictor': name, 'key': k, 'part': 'energy',
                                'contribution_pts': 100 * v / scales[i], 'structure': structure} for k, v in attribution.four_parts(preds[name][i], held).items()]
    where_rows += ev['where_rows']
    if 'd2_by_context' in ev:
        where_rows += [{'grouping': 'D2 per context (treated, mean-gene common energy)', 'key': k, 'value': v if v is not None else 'undefined'}
                       for k, v in ev['d2_by_context'].items()]
    gfloor = None
    if floor is not None and case.get('gene_variance') is not None:   # per-gene share of the floor (approximation, stated)
        g = case['gene_variance']
        gfloor = attribution.gene_floor(floor, g / g.sum())
    for gname, labels in genes.items():
        for name in ('parent', 'action', 'anchor', 'nochange'):
            for i, seed in enumerate(seeds):
                vec = gene_contributions(preds[name][i], held)
                where_rows += [{'seed': seed, 'grouping': f'gene: {gname}', 'predictor': name,
                                'floor': 'approximate per-gene floor' if gfloor is not None else 'no per-gene floor (raw excess only)', **r}
                               for r in attribution.gene_where(vec, labels, scales[i], gfloor)]
    group = saved['perturbation_group']
    if group is not None:
        labels = np.asarray([mapping[int(c)] for c in held['p_code']])
        drugs = {g: len(np.unique(held['p_code'][labels == g])) for g in np.unique(labels)}
        for name in ('parent', 'anchor', 'nochange'):
            for i, seed in enumerate(seeds):
                where_rows += [{'seed': seed, 'grouping': 'perturbation group (descriptive)', 'predictor': name, 'drugs': drugs[r['key']], **r}
                               for r in attribution.where(unit_parts(preds[name][i], held), held, labels, scales[i], floor)]
        test = attribution.perturbation_group_test(net(mean_parts, floor), held, mapping, chosen=group['chosen'])[0]
        where_rows.append({'grouping': 'perturbation group: chosen on calibration, tested on outer', 'predictor': 'parent (seed mean)',
                           'key': test['group'], 'part': test['part'], 'selection': group['rule'], 'excess_basis': group['basis'],
                           'excess_pts': 100 * test['excess'] / float(np.mean(scales)),
                           **{k: v for k, v in test.items() if k not in ('group', 'part', 'excess')}})
    else:
        where_rows.append({'grouping': 'perturbation group', 'key': 'none declared'})

    # three readings of the nested parent, each with its floor, ceilings and composition
    readings = []
    note = ('the readings differ in composition (identity mix in the composition column'
            + (f"; {case['meta']['composition']}" if case['meta'].get('composition') else '')
            + '), so a gap between them is not read as specific to the held-out units')
    for i, seed in enumerate(seeds):
        for reading, u, side_name, noise in (('fit', case['units']['fit'], 'train', case['noise'].get('fit')),
                                             ('calibration', case['units']['calibration'], 'train', case['noise'].get('calibration')),
                                             ('outer', case['units']['outer_nested'], 'outer', case['noise'].get('outer_nested'))):
            p, a = case['load'](seed, 'nested', side_name, u), case['load'](seed, 'nested', side_name, u, 'anchor_raw')
            bb, CC = bag_of(case['bags']['heldout' if side_name == 'outer' else 'train'], u)
            fl = attribution.floor_parts(u, noise) if noise is not None else None
            un = {'common': float(attribution.bag_gap(u, bb, CC)['common'].sum()), 'differential': float(fl['differential'].sum()) if fl is not None else 0.}
            s = scores(p, u)
            al = card.alignment(p, a, u, un)
            readings.append({'seed': seed, 'reading': reading, 'units': len(u['truth']), **{part: s[key] for part, key in PARTKEY.items()},
                             **({f'floor_{part}': float(v.sum()) for part, v in fl.items()} if fl is not None else {'floor': 'not computed'}),
                             **{f'{part}_{k}': v for part, d in al.items() for k, v in d.items()},
                             'composition': composition(u), 'note': note})
        readings.append({'seed': seed, 'reading': 'full parent, outer', 'units': len(held['truth']), 'total': scales[i],
                         **{f'{part}_{k}': v for part, d in align[i].items() for k, v in d.items()}, 'composition': composition(held)})

    headline = [{'line': 0, 'text': line0}, {'line': 1, 'text': line1}, {'line': 2, 'text': line2}, {'line': 3, 'text': line3}]
    if line4:
        headline.append({'line': 4, 'text': line4})
    record = {'verification': verification, 'option': {'record': option, 'verification': option_verification} if option else None,
              'detections': detections, 'separation': separation, 'shares_pts': shares,
              'players': players, 'line2_per_part': line2_parts, 'headline': [h['text'] for h in headline]}
    return {'CARD': headline, 'ROWS': ev['rows'], 'WHERE': where_rows, 'REFERENCES': ref_rows, 'READINGS': readings}, record


def repair_entry(p, R, a, units, mask):
    """One seed's selection entry for a parent p and its repair R (anchor a) on the masked calibration units: the
    diagnosis labels, every policy's amount and total, and each repair shrunk at the parent's cell-line amount."""
    entry = {'labels': policies.labels(p, R, units, mask), 'amount': {}, 'calibration': {}}
    for policy in policies.FIXED + ('protect_common',):
        entry['amount'][policy], entry['calibration'][policy] = policies.choose(policy, p, R, a, units, mask)
    for t in policies.REPAIRS:
        x = (entry['amount'][t], entry['amount']['cheapest_fix'])
        entry['amount'][f'{t}_at_parent_lam'] = x
        entry['calibration'][f'{t}_at_parent_lam'] = scores(policies.predict(f'{t}_at_parent_lam', x, p, R, a, units), units, mask)['total_mse']
    return entry


def parent_entry(p, a, units, mask):
    """One seed's selection entry without a repair: the parent, the cell-line-level shrink and the uniform shrink,
    each with its amount and calibration total."""
    entry = {'amount': {}, 'calibration': {}}
    for policy in ('parent', 'cheapest_fix', 'global_parent_shrink'):
        entry['amount'][policy], entry['calibration'][policy] = policies.choose(policy, p, None, a, units, mask)
    return entry


def parent_selection(entries, seeds, actions):
    """The current rule over the parent-only menu ``actions`` (ordinary validation), and the recommendation branch:
    the action is not the parent and is below the parent and the uniform shrink in every seed."""
    choice = policies.select(entries, actions)
    action = choice['ordinary_validation']
    value = lambda t: [entries[str(s)]['calibration'][t] for s in seeds]
    below = lambda x, y: all(u < v for u, v in zip(value(x), value(y)))
    branch = {'action_is_parent': action == 'parent', 'below_parent_every_seed': below(action, 'parent'),
              'below_uniform_shrink_every_seed': below(action, 'global_parent_shrink')}
    branch['recommendation'] = not branch['action_is_parent'] and branch['below_parent_every_seed'] \
        and branch['below_uniform_shrink_every_seed']
    return choice, action, branch
