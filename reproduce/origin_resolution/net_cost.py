"""The matched comparisons of every restoring change and intervention, from saved grids.

The changes are the restoring changes information localization names: Tahoe-100M's four models on folds 1-3, and the
summary-statistics and DeepSets models on Replogle-Nadig folds 1-2 and Jiang et al. folds 4-5. An intervention that recovers nothing is a
'control'. Other screens have no nested chain for a change, so they read the model-level columns only.

Nine stages, writing into ``--out``; the first four run in this order:

- ``calibration``: from fit and calibration lines; builds no held-out unit. ``AMOUNTS.json`` and the
  ``*_CALIBRATION.csv`` tables, and ``FORMULA_CHECK.csv`` (the recovery formula on the two measured planted points);
- ``heldout``: the fixed amounts on held-out lines: ``CARDS_HELDOUT.csv``, ``GATE_HELDOUT.csv``, ``MODELS_HELDOUT.csv``,
  ``SEED_SLOPES_HELDOUT.csv`` and ``RECOVERY_HELDOUT.csv``;
- ``tables``: every summary from those tables, so another margin needs no rerun: ``CARDS.csv``, ``SEED_BAND.csv``,
  ``SLOPE_SUMMARY.csv``, ``LADDER.csv``, ``LEDGER.csv``, ``RECOVERY_CARDS.csv``, ``RECOVERY_SUMMARY.csv``,
  ``RESTORED_LIST.csv`` and ``DESIGN_BASELINE.csv``;
- ``rule``: the recommendation rule (``net_cost.recommend``) per restoring change counted as recovered and per
  comparison, from ``CARDS.csv``, ``GATE_CALIBRATION.csv`` and ``GATE_HELDOUT.csv``: ``RECOMMENDATIONS.csv``. The
  verdict and the worst gaps on calibration lines; the same worst gaps on held-out lines, which take no part in it. A
  model's fixed intervention is the change of kind 'control' on the same screen and fold;
- ``common-share``: where the added error sits, from ``CARDS.csv`` and ``RESTORED_LIST.csv`` and the full chain's cost
  readings: ``COMMON_SHARE.csv``, the common part's share of each restoring change's cost, mean over seeds;
- ``light-cases``: ``OTHER_SCREENS_LIGHT_CASES.csv``, the light version's readings on the screens other than
  Tahoe-100M as information localization names them (``LIGHT_APPLIED``, and the light columns of the extended menu's
  cases for the summary-statistics and DeepSets models on Replogle-Nadig, seed 31002, window 0.02); ``tables``
  reads the saved copy (``LIGHT``);
- ``plate-free-seed`` (Tahoe-100M): ``PLATE_FREE_SEED.csv``, the original model and the second training run each
  calibrated toward the plain baseline with the same two-amount rule, amounts chosen on calibration lines; their
  difference on calibration and held-out lines, in points of the original model's total;
- ``plate-free-cards``: ``PLATE_FREE_CARDS.csv``, the gate's points without the plate, from ``PLATE_FREE_SEED.csv``,
  ``CARDS.csv`` and ``GATE_*.csv``;
- ``uniform-vs-two-part``: ``UNIFORM_VS_TWO_PART.csv``, the original model's two-part calibration against a uniform
  shrink toward the same target (the plain and the design-aware baseline) on every model and fold, amounts from the
  nested chain's calibration lines, read on held-out lines, per seed and as the seed mean with a line-resampled
  interval.

Files keep their original names and columns; 'card' there means one change on one model and fold, 'second seed' the
extra seed, 'set' a step of the matched comparisons.

    python -m reproduce.origin_resolution.net_cost --stage calibration --out DIR
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scsailor.resolution import card, il_cells, menu, net_cost
from scsailor.resolution.policies import predict
from scsailor.resolution.stages import bag_of
from scsailor.resolution.scoring import PARTS, line_parts, scores
from scsailor.resolution.comparison import bootstrap, pool_seeds
from scsailor.resolution.slopes import fraction
from . import ROOT, SEEDS, at, genetic_cells, read_csv, runs, tahoe, write_csv
from .slopes import BINDINGS, BUILT, CASES, load_case

__all__ = ["all_changes", "key", "level", "counted", "levels", "read_side", "calibration_stage", "heldout_stage", "tables_stage", "saved_gaps",
           "rule_stage", "common_share_stage", "light_cases_stage",
           "plate_free_seed_stage", "plate_free_cards_stage",
           "uniform_vs_two_part_stage", "MARGINS", "recommended", "main"]

# The saved tables' ``recommended_set<stage>_m<margin>`` columns and ``LADDER.csv``: per comparison set, whether a change
# is above the margin against every reference in every seed with some of it kept in every seed. They are columns of the
# saved tables, not the paper's recommendation, which is ``net_cost.recommend`` (the ``rule`` stage).
MARGINS = (0., .2, .5)   # % of the original model's total MSE


def recommended(gaps, alphas, margin):
    """Above the margin against every reference, every seed, with some of the change kept in every seed."""
    return min(alphas) > 0 and all(min(v.values()) > margin for v in gaps.values())


MODELS = ('summary', 'deepsets', 'attention', 'official_state')
PLANTED = {'common mean': 0, 'common variance': 5, 'statistic-related perturbation identity': 1, 'selective mean': 1}   # planted state per target
JIANG_MAP = ROOT / 'information-localization-light/jiang'   # information localization's light map runs
TAG = {'summary': 'su', 'deepsets': 'ds'}
NATURAL_AMOUNTS = ROOT / 'information-localization/state-correction-amounts/results/NATURAL_AMOUNTS.csv'
ROOT2 = 2 ** .5 - 1
LINES = (.15, .20, .25, .30, .35)   # recovery lines of the summary
COST_ONLY = ((ROOT / 'cost-readings-state/COST.csv', 'official_state', 'official State'),
             (ROOT / 'cost-readings-attention-model-jiang/COST.csv', 'attention', 'attention'),
             (ROOT / 'cost-readings-attention-model/COST.csv', 'attention', 'attention'),
             (ROOT / 'cost-readings-built-models/summary/COST.csv', 'summary', 'summary-statistics'),
             (ROOT / 'cost-readings-built-models/deepsets/COST.csv', 'deepsets', 'DeepSets'))   # full chain only
LIGHT = ROOT / 'light-version-cases/OTHER_SCREENS_LIGHT_CASES.csv'
LIGHT_APPLIED = [ROOT / 'light-version-cases/applications' / name for name in
                 ('JIANG_PARSE_LIGHT_CASES.csv', 'GSE_LIGHT_CASES.csv', 'OFFICIAL_STATE_LIGHT_CASES.csv',
                  'ATTENTION_PARSE_REPLOGLE_LIGHT_CASES.csv', 'GSE_ATTENTION_LIGHT_CASES.csv', 'JIANG_ATTENTION_LIGHT_CASES.csv')]
LIGHT_ARCHIVE = ROOT / 'light-version-cases/cases-extended-menu/CASES.csv'


def key(c):
    return f"{c['screen']}|{c['fold']}|{c['model']}|{c['cell']}"


def level(target):
    return 'selective' if target in ('statistic-related perturbation identity', 'selective mean') else 'shared'


def tahoe_changes():
    for fold in ('F1', 'F2', 'F3'):
        for model in MODELS:
            il = tahoe.IL_NAME.get(model, model)
            map_dir = runs.MODELS[fold][il]['map']
            for cell, rows in runs.carried_cells(map_dir, il):
                control = all(r['kind'].startswith('diagnostic') for r in rows)
                found = {r['target']: runs.synthetic_card(map_dir, il, cell, r['suffix']) for r in rows}
                yield {'screen': 'tahoe', 'fold': fold, 'model': model, 'cell': cell, 'kind': 'control' if control else 'restoring',
                       'targets': sorted(found), 'reading': {t: il_cells.recovery(f, SEEDS) for t, f in found.items()}, 'found': found}


def genetic_changes():
    for dataset, folds in (('jiang', ('F4', 'F5')), ('replogle', ('F1', 'F2'))):
        for fold in folds:
            for model in BUILT:
                for cell, (columns, recovered) in genetic_cells.handed_on(dataset, fold, model).items():
                    control = 'diagnostic intervention' in columns['handed_on_by']
                    found = {}
                    if dataset == 'replogle':
                        rows = dict(runs.carried_cells(genetic_cells.MAPS / fold, model))[cell]
                        found = {r['target']: runs.synthetic_card(genetic_cells.MAPS / fold, model, cell, r['suffix']) for r in rows}
                    yield {'screen': dataset, 'fold': fold, 'model': model, 'cell': cell, 'kind': 'control' if control else 'restoring',
                           'targets': sorted(recovered), 'reading': recovered, 'found': found}


def all_changes():
    return list(tahoe_changes()) + list(genetic_changes())


def counted(c):
    """A restoring change counts as recovered unless information localization's reading of every target is partial."""
    return c['kind'] == 'restoring' and not all(v in ('partial or mixed', 'not recovered') for v in c['reading'].values())


def levels(c):
    found = {level(t) for t in c['targets']}
    return 'both' if len(found) == 2 else found.pop()


def read_side(screen, case, which, cell):
    """One side (``net_cost``): 'fit' and 'calibration' on the nested chain, 'held-out' on the full chain, 'held-out
    nested' on the nested chain. Builds held-out units only for the last two, which only the held-out stage asks for."""
    unit_set, chain, grid, bank = {'fit': ('fit', 'nested', 'train', 'train'), 'calibration': ('calibration', 'nested', 'train', 'train'),
                                   'held-out': ('outer', 'full', 'outer', 'heldout'),
                                   'held-out nested': ('outer_nested', 'nested', 'outer', 'heldout')}[which]
    u = case['units'][unit_set]
    sd = {'u': u, 'p': {s: case['load'](s, chain, grid, u) for s in SEEDS},
          'R': {s: case['load'](s, f'cell_{chain}', grid, u) for s in SEEDS} if cell else None}
    if which in ('calibration', 'held-out'):
        sd['a'] = np.mean([case['load'](s, chain, grid, u, 'anchor_raw') for s in SEEDS], axis=0)
        sd['b'] = bag_of(case['bags'][bank], u)[0]
        sd['A'] = sd['a'] + BINDINGS[screen].design_shift(case, u, chain == 'nested', bank)
    return sd


def evaluation_supports(c, target, seed):
    """(end supports, parent supports) of the natural half's map run for this target, or None. Jiang's light runs are named
    by model, half, cut, arm, rate and seed."""
    if c['screen'] == 'jiang':
        cut, arm, rate = c['cell'].split('/')
        run = JIANG_MAP / c['fold'] / 'map' / f"j{c['fold']}_{TAG[c['model']]}_nat_{cut}_{arm}_{rate}_{seed}"
    else:
        row = c['found'].get(target, {}).get(('natural', seed))
        run = at(row['run']) if row else None
    if run is None or not (run / 'EVAL.json').is_file():
        return None
    return json.loads((run / 'EVAL.json').read_text())['after']['supports'], json.loads((run / 'EVAL_PARENT.json').read_text())['supports']


def planted_runs(c):
    """(target, planted state, seed, end supports, parent supports) of every target and seed with a saved evaluation."""
    for target in c['targets']:
        planted = PLANTED.get(target)
        for seed in SEEDS:
            read = evaluation_supports(c, target, seed) if planted is not None else None
            if read is not None:
                yield target, planted, seed, *read


def recovery_rows(c, kept):
    rows = []
    for target, planted, seed, end, parent in planted_runs(c):
        q, rho = net_cost.planted_reading(end['validation'], planted)
        k = kept[str(seed)]
        part = 1 if level(target) == 'selective' else 0   # the statistic's layer in the two amounts
        field, design = k['alpha'] * k['action_field'][part], k['alpha'] * k['action_design'][part]
        rows.append({'card': key(c), 'target': target, 'level': level(target), 'seed': seed, 'Q_full': q, 'rho': rho,
                     'Q_parent': net_cost.planted_reading(parent['validation'], planted)[0],
                     'alpha': k['alpha'], 'delivered_field': field, 'delivered_design': design,
                     'Q_at_alpha': net_cost.recovery_at(q, rho, k['alpha']),
                     'Q_at_field': net_cost.recovery_at(q, rho, field), 'Q_at_design': net_cost.recovery_at(q, rho, design)})
    return rows


def formula_check():
    """The planted statistic through a corrector at the natural task's amounts: measured against the formula at beta x
    lam_d."""
    rows = []
    for r in read_csv(NATURAL_AMOUNTS):
        if 'repaired' in r['predictor']:
            run = ROOT / f"information-localization-maps-fold-1/map/map-summary-synthetic-{r['seed']}-projections-e-x30/EVAL.json"
            q, rho = net_cost.planted_reading(json.loads(run.read_text())['after']['supports']['heldout'], 1)
            a = float(r['beta']) * float(r['lam_d'])
            rows.append({'seed': r['seed'], 'a': a, 'Q_full': q, 'rho': rho, 'formula': net_cost.recovery_at(q, rho, a),
                         'measured': float(r['Q_bit1'])})
    return rows


def calibration_stage(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    amounts, cards, gates, seeds, models, recovery = {}, [], [], [], [], []
    for screen, fold, model in CASES:
        sd = read_side(screen, load_case(screen, fold, model, False), 'calibration', None)
        m = net_cost.calibrate_original(sd)
        amounts[f'{screen}|{fold}|{model}'] = m
        models.append({'screen': screen, 'fold': fold, 'model': model, 'units': len(sd['u']['truth']), 'kappa': m['kappa'],
                       'lam_c_original': float(np.mean([v['original_design'][0] for v in m['seeds'].values()])),
                       'lam_d_original': float(np.mean([v['original_design'][1] for v in m['seeds'].values()]))})
        seeds += [{'screen': screen, 'fold': fold, 'model': model, 'side': 'calibration', **r} for r in net_cost.extra_seed_slopes(sd)]
        print(screen, fold, model, flush=True)
    for c in all_changes():
        case = load_case(c['screen'], c['fold'], c['model'], False, c['cell'])
        cal, fit = (read_side(c['screen'], case, w, c['cell']) for w in ('calibration', 'fit'))
        m = net_cost.calibrate_original(cal)
        kept = net_cost.calibrate_change(cal, m)
        amounts[key(c)] = {'model': m, 'kept': kept}
        s_cal, per_seed = net_cost.change_slopes(cal)
        s_fit, _ = net_cost.change_slopes(fit)
        gaps = net_cost.cumulative_gaps(cal, m, kept)
        alphas = [kept[str(s)]['alpha'] for s in SEEDS]
        row = {'card': key(c), 'screen': c['screen'], 'fold': c['fold'], 'model': c['model'], 'cell': c['cell'], 'kind': c['kind'],
               'counted': counted(c),
               'levels': levels(c) if c['kind'] == 'restoring' else '', 'targets': '; '.join(c['targets']),
               'reading': '; '.join(f'{k}: {v}' for k, v in c['reading'].items()),
               'calibration_lines': len(BINDINGS[c['screen']].calibration_lines(c['fold'])), **{f's_fit_{k}': v for k, v in s_fit.items()},
               **{f's_cal_{k}': v for k, v in s_cal.items()},
               **{f's_cal_{s}': v for s, v in zip(SEEDS, per_seed)}, 'alpha_min': min(alphas), 'alpha_mean': float(np.mean(alphas)),
               'price_cal': net_cost.price(s_cal['total']), 'kappa': m['kappa']}
        for stage in (1, 2, 3):
            worst = [min(v.values()) for v in gaps[stage].values()]
            row[f'min_gap_set{stage}'] = min(worst)
            row[f'tightest_set{stage}'] = min(((n, g) for v in gaps[stage].values() for n, g in v.items()), key=lambda x: x[1])[0]
            for margin in MARGINS:
                row[f'recommended_set{stage}_m{margin:g}'] = recommended(gaps[stage], alphas, margin)
        worst = [min(min(gaps[1][s].values()), gaps['variant'][s]['original, field two-amount calibration']) for s in SEEDS]
        row['min_gap_set1_with_field_calibration'] = min(worst)
        cards.append(row)
        for stage, by_seed in gaps.items():
            gates += [{'card': key(c), 'stage': stage, 'seed': s, 'reference': n, 'gap_pts': g} for s, v in by_seed.items() for n, g in v.items()]
        recovery += recovery_rows(c, kept)
        print(key(c), flush=True)
    with (out / 'AMOUNTS.json').open('x') as handle:
        handle.write(json.dumps(amounts, indent=1) + '\n')
    for name, rows in (('CARDS_CALIBRATION.csv', cards), ('GATE_CALIBRATION.csv', gates), ('SEED_SLOPES_CALIBRATION.csv', seeds),
                       ('MODELS_CALIBRATION.csv', models),
                       ('RECOVERY_CALIBRATION.csv', recovery), ('FORMULA_CHECK.csv', formula_check())):
        write_csv(out / name, rows)


def heldout_stage(out):
    out = Path(out)
    if (out / 'CARDS_HELDOUT.csv').exists():
        raise FileExistsError('held-out tables exist; this stage runs once')
    amounts = json.loads((out / 'AMOUNTS.json').read_text())
    models, seeds, cards, gates, recovery = [], [], [], [], []
    for screen, fold, model in CASES:
        sd = read_side(screen, load_case(screen, fold, model, True), 'held-out', None)
        models.append({'screen': screen, 'fold': fold, 'model': model, **net_cost.reference_points(sd, amounts[f'{screen}|{fold}|{model}'])})
        seeds += [{'screen': screen, 'fold': fold, 'model': model, 'side': 'held-out', **r} for r in net_cost.extra_seed_slopes(sd)]
        print(screen, fold, model, flush=True)
    for c in all_changes():
        case = load_case(c['screen'], c['fold'], c['model'], True, c['cell'])
        held = read_side(c['screen'], case, 'held-out', c['cell'])
        nested = read_side(c['screen'], case, 'held-out nested', c['cell'])
        saved = amounts[key(c)]
        m, kept = saved['model'], saved['kept']
        s, per_seed = net_cost.change_slopes(held)
        s_nested, _ = net_cost.change_slopes(nested)
        gaps = net_cost.cumulative_gaps(held, m, kept)
        row = {'card': key(c), **{f's_held_{k}': v for k, v in s.items()}, **{f's_held_{i}': v for i, v in zip(SEEDS, per_seed)},
               **{f's_held_nested_{k}': v for k, v in s_nested.items()}, 'kept_held': fraction(s['total']), 'price_held': net_cost.price(s['total']),
               **net_cost.costs(held), **net_cost.ledger(held, m, kept)}
        for stage in (1, 2, 3):
            row[f'min_gap_set{stage}'] = min(min(v.values()) for v in gaps[stage].values())
        cards.append(row)
        for stage, by_seed in gaps.items():
            gates += [{'card': key(c), 'stage': stage, 'seed': i, 'reference': n, 'gap_pts': g} for i, v in by_seed.items() for n, g in v.items()]
        for target, planted, i, end, _ in planted_runs(c):
            q, rho = net_cost.planted_reading(end['heldout'], planted)
            a = fraction(per_seed[SEEDS.index(i)])
            recovery.append({'card': key(c), 'target': target, 'seed': i, 'Q_full': q, 'rho': rho, 'kept_held': a,
                             'Q_at_kept_held': net_cost.recovery_at(q, rho, a)})
        print(key(c), flush=True)
    for name, rows in (('CARDS_HELDOUT', cards), ('GATE_HELDOUT', gates), ('MODELS_HELDOUT', models), ('SEED_SLOPES_HELDOUT', seeds),
                       ('RECOVERY_HELDOUT', recovery)):
        write_csv(out / f'{name}.csv', rows)


def num(rows, column):
    return np.asarray([float(r[column]) for r in rows if r[column] not in ('', None, 'nan')])


def groups(cards):
    """(name, kind, cards): the reporting groups; a change restoring both levels sits in shared and in selective."""
    tahoe = [c for c in cards if c['screen'] == 'tahoe']
    genetic = [c for c in cards if c['screen'] != 'tahoe']
    for label, part in (('Tahoe', tahoe), ('Replogle-Nadig', [c for c in cards if c['screen'] == 'replogle']),
                        ('Jiang', [c for c in cards if c['screen'] == 'jiang']), ('genetic screens', genetic)):
        restoring = [c for c in part if c['counted'] == 'True']
        yield label, 'restoring', restoring
        yield label, 'shared', [c for c in restoring if c['levels'] in ('shared', 'both')]
        yield label, 'selective', [c for c in restoring if c['levels'] in ('selective', 'both')]
        yield label, 'both levels', [c for c in restoring if c['levels'] == 'both']
        yield label, 'no recovery', [c for c in part if c['kind'] == 'control']
    for model in MODELS:
        yield f'Tahoe {model}', 'restoring', [c for c in tahoe if c['model'] == model and c['counted'] == 'True']


def seed_band(seeds):
    rows = []
    for side in sorted({r['side'] for r in seeds}):
        for screen in ('tahoe', 'gse306429', 'jiang', 'replogle', 'parse'):
            for part in PARTS:
                v = num([r for r in seeds if r['side'] == side and r['screen'] == screen], part)
                if len(v):
                    rows.append({'side': side, 'screen': screen, 'part': part, 'ordered_pairs': len(v), 'mean': v.mean(),
                                 'p05': np.percentile(v, 5), 'p95': np.percentile(v, 95), 'min': v.min(), 'max': v.max()})
    return rows


def slope_summary(cards, lo, hi):
    rows = []
    for label, kind, part in groups(cards):
        if not part:
            continue
        row = {'group': label, 'kind': kind, 'cards': len(part)}
        for side in ('fit', 'cal', 'held'):
            for name in PARTS:
                v = num(part, f's_{side}_{name}')
                row |= {f's_{side}_{name}_min': v.min(), f's_{side}_{name}_median': np.median(v), f's_{side}_{name}_max': v.max()}
        for side in ('cal', 'held'):
            s = num(part, f's_{side}_total')
            row |= {f'{side}_below_root2': int((s < ROOT2).sum()), f'{side}_root2_to_half': int(((s >= ROOT2) & (s < .5)).sum()),
                    f'{side}_half_or_more': int((s >= .5).sum()),
                    f'{side}_below_band': int((s < lo).sum()), f'{side}_in_band': int(((s >= lo) & (s <= hi)).sum()),
                    f'{side}_above_band': int((s > hi).sum())}
            prices = np.asarray([np.inf if x <= 0 else net_cost.price(x) for x in s])
            row[f'{side}_median_price'] = float(np.median(prices))
        real = num(part, 'realized_price')
        row['realized_median_price'] = float(np.median(real)) if len(real) else ''
        a, b = num(part, 's_cal_total'), num(part, 's_held_total')
        row |= {'spearman_cal_held': float(spearmanr(a, b).statistic) if len(a) > 2 else '',
                'within_eighth': int((np.abs(a - b) <= .125).sum()), 'median_cost_pts': float(np.median(num(part, 'cost_total_pts'))),
                'cost_above_zero': int((num(part, 'cost_total_pts') > 0).sum())}
        rows.append(row)
    return rows


def ladder(cards):
    rows = []
    for label, kind, part in groups(cards):
        if not part:
            continue
        for margin in MARGINS:
            row = {'group': label, 'kind': kind, 'cards': len(part), 'margin': margin}
            for stage in (1, 2, 3):
                row[f'recommended_set{stage}'] = sum(c[f'recommended_set{stage}_m{margin:g}'] == 'True' for c in part)
            row['set1_with_field_calibration'] = sum(float(c['min_gap_set1_with_field_calibration']) > margin and float(c['alpha_min']) > 0 for c in part)
            rows.append(row)
    return rows


def ledger(cards):
    columns = ('baseline_above_original', 'gap', 'design', 'correction', 'change', 'second_seed', 'reverse_change', 'reverse_correction',
               'reverse_design')
    rows = []
    for label, kind, part in groups(cards):
        if not part:
            continue
        mean = {k: float(np.mean(num(part, k))) for k in columns}
        shares = num(part, 'design') / num(part, 'gap')
        rows.append({'group': label, 'kind': kind, 'cards': len(part), **mean, 'design_share': mean['design'] / mean['gap'],
                     'reverse_design_share': mean['reverse_design'] / mean['gap'],
                     'design_share_card_median': float(np.median(shares)),
                     'change_above_second_seed': int((num(part, 'change') > num(part, 'second_seed')).sum())})
    return rows


def recovery_tables(cal, held, cards):
    by_card = {c['card']: c for c in cards}
    pairs = {(r['card'], r['target']) for r in cal if r['card'] in by_card}
    rows = []
    for card, target in sorted(pairs):
        a = [r for r in cal if (r['card'], r['target']) == (card, target)]
        h = [r for r in held if (r['card'], r['target']) == (card, target)]
        c = by_card[card]
        row = {'card': card, 'kind': c['kind'], 'counted': c['counted'], 'target': target, 'level': a[0]['level'], 'seeds': len(a)}
        for k in ('Q_full', 'rho', 'Q_parent', 'alpha', 'delivered_field', 'delivered_design', 'Q_at_alpha', 'Q_at_field', 'Q_at_design'):
            row[k] = float(np.mean(num(a, k)))
        for k in ('Q_full', 'rho', 'kept_held', 'Q_at_kept_held'):
            row[f'heldout_{k}'] = float(np.mean(num(h, k))) if h else ''
        rows.append(row)
    out = []
    tahoe = [r for r in rows if r['counted'] == 'True' and r['card'].startswith('tahoe')]
    for name, part in (('selective', [r for r in tahoe if r['level'] == 'selective']), ('shared', [r for r in tahoe if r['level'] == 'shared']),
                       ('shared, common mean only', [r for r in tahoe if r['target'] == 'common mean'])):
        if part:
            base = {'level': name, 'readings': len(part)}
            for k in ('Q_full', 'Q_at_alpha', 'Q_at_field', 'Q_at_design'):
                v = num(part, k)
                out.append({**base, 'read': k, 'min': v.min(), 'median': np.median(v), 'max': v.max(),
                            **{f'below_{x:g}': int((v < x).sum()) for x in LINES}})
    return rows, out


def restored(cards):
    """Every change with its cost, those of the full chain's cost readings included (``cost_readings``), whose
    recovery is quoted from the light readings."""
    rows = [{'screen': c['screen'], 'fold': c['fold'], 'model': c['model'], 'cell': c['cell'], 'kind': c['kind'], 'counted': c['counted'],
             'levels': c['levels'],
             'targets': c['targets'], 'reading': c['reading'], 'cost_pts': float(c['cost_total_pts']), 'cost_seeds': c['cost_seeds'],
             'seeds_above_zero': c['seeds_above_zero']} for c in cards]
    cases = read_csv(LIGHT)
    for path, model, name in COST_ONLY:
        found = {}
        for r in read_csv(path):
            if r['cell'] != runs.COMPARATOR:
                found.setdefault((r['dataset'], r['fold'], r['cell']), []).append(float(r['cost_total_pts']))
        for (dataset, fold, cell), v in found.items():
            names = sorted({r['target_name'] for r in cases if (r['dataset'], r['fold'], r['cell'], r['model'], r['kind']) == (dataset,
                                                            fold, cell, name, 'qualified repair')})
            kinds = {level(n) for n in names}
            rows.append({'screen': dataset, 'fold': fold, 'model': model, 'cell': cell, 'kind': 'restoring', 'counted': 'True',
                         'levels': 'both' if len(kinds) == 2 else min(kinds),
                         'targets': '; '.join(names), 'reading': 'light reading (quoted)', 'cost_pts': float(np.mean(v)),
                         'cost_seeds': ' / '.join(f'{x:+.2f}' for x in v),
                         'seeds_above_zero': sum(x > 0 for x in v)})
    for r in rows:
        r |= {'both_levels': r['levels'] == 'both', 'raised': r['cost_pts'] > 0}
    return rows


def design_table(models_cal, models_held):
    cal = {(r['screen'], r['fold'], r['model']): r for r in models_cal}   # every model and fold
    return [{**r, 'lam_c_original': cal[(r['screen'], r['fold'], r['model'])]['lam_c_original'],
             'lam_d_original': cal[(r['screen'], r['fold'], r['model'])]['lam_d_original'],
             'lowest_no_model': min(('no change', float(r['no_change'])), ('baseline', float(r['baseline'])),
                                    ('design baseline', float(r['design_baseline'])), key=lambda x: x[1])[0],
             'design_beats_original': float(r['design_baseline']) < 0} for r in models_held]


def tables_stage(out):
    out = Path(out)
    write = lambda path, rows: write_csv(path, rows, mode='w')   # derived: another margin, or new cost readings, rebuilds them
    held = {r['card']: r for r in read_csv(out / 'CARDS_HELDOUT.csv')}
    rows = [c | {('held_' + k if k.startswith('min_gap') else k): v for k,
                 v in held[c['card']].items() if k != 'card'} for c in read_csv(out / 'CARDS_CALIBRATION.csv')]
    write(out / 'CARDS.csv', rows)
    band = seed_band(read_csv(out / 'SEED_SLOPES_CALIBRATION.csv') + read_csv(out / 'SEED_SLOPES_HELDOUT.csv'))
    write(out / 'SEED_BAND.csv', band)
    t = next(r for r in band if (r['side'], r['screen'], r['part']) == ('calibration', 'tahoe', 'total'))
    write(out / 'SLOPE_SUMMARY.csv', slope_summary(rows, t['p05'], t['p95']))
    write(out / 'LADDER.csv', ladder(rows))
    write(out / 'LEDGER.csv', ledger(rows))
    per_card, reading = recovery_tables(read_csv(out / 'RECOVERY_CALIBRATION.csv'), read_csv(out / 'RECOVERY_HELDOUT.csv'), rows)
    write(out / 'RECOVERY_CARDS.csv', per_card)
    write(out / 'RECOVERY_SUMMARY.csv', reading)
    write(out / 'RESTORED_LIST.csv', restored(rows))
    write(out / 'DESIGN_BASELINE.csv', design_table(read_csv(out / 'MODELS_CALIBRATION.csv'), read_csv(out / 'MODELS_HELDOUT.csv')))


def saved_gaps(path):
    """card -> stage -> seed -> reference -> gap, as ``net_cost.cumulative_gaps`` gives them, from a saved GATE table."""
    out = {}
    for r in read_csv(path):
        stage = r['stage'] if r['stage'] == 'variant' else int(r['stage'])
        out.setdefault(r['card'], {}).setdefault(stage, {}).setdefault(int(r['seed']), {})[r['reference']] = float(r['gap_pts'])
    return out


def rule_stage(out):
    out = Path(out)
    cards = read_csv(out / 'CARDS.csv')
    sides = {'calibration': saved_gaps(out / 'GATE_CALIBRATION.csv'), 'heldout': saved_gaps(out / 'GATE_HELDOUT.csv')}
    controls = {}
    for c in cards:
        if c['kind'] == 'control':
            model = (c['screen'], c['fold'], c['model'])
            if model in controls:
                raise ValueError(f'{model} has two fixed interventions: {controls[model]}, {c["card"]}')
            controls[model] = c['card']
    rows = []
    for c in cards:
        if c['kind'] != 'restoring' or c['counted'] != 'True':
            continue
        control = controls.get((c['screen'], c['fold'], c['model']))
        for comparison in net_cost.COMPARISONS:
            worst = {side: net_cost.worst_gap(gaps[c['card']], comparison) for side, gaps in sides.items()}
            own = {side: None if control is None else net_cost.own_control_gap(gaps[c['card']], gaps[control], comparison) for side, gaps in sides.items()}
            rows.append({'card': c['card'], 'screen': c['screen'], 'fold': c['fold'], 'model': c['model'], 'cell': c['cell'], 'comparison': comparison,
                         'recommended': net_cost.recommend(float(c['alpha_min']), worst['calibration'], own['calibration']),
                         'calibration_worst_vs_references': worst['calibration'], 'calibration_worst_vs_own_control': own['calibration'],
                         'heldout_worst_vs_references': worst['heldout'], 'heldout_worst_vs_own_control': own['heldout']})
    write_csv(out / 'RECOMMENDATIONS.csv', rows, mode='w')
    for screen in dict.fromkeys(r['screen'] for r in rows):
        counts = [sum(r['recommended'] for r in rows if (r['screen'], r['comparison']) == (screen, x)) for x in net_cost.COMPARISONS]
        print(screen, 'recommended', dict(zip(net_cost.COMPARISONS, counts)), 'of', len({r['card'] for r in rows if r['screen'] == screen}))


def common_share_stage(out):
    """``COMMON_SHARE.csv``: each restoring change of ``RESTORED_LIST.csv`` with its cost's total, common and contrast
    parts, mean over seeds (``CARDS.csv`` for the matched comparisons, the cost readings for the full chain), whether
    it raised error (total above zero) and the common part's share of the total."""
    out = Path(out)
    names = {'cost_total_pts': 'total', 'cost_common_pts': 'common', 'cost_differential_pts': 'contrast'}
    frames = [pd.read_csv(out / 'CARDS.csv', low_memory=False)[['screen', 'fold', 'model', 'cell', *names]]]
    for path, model, _ in COST_ONLY:
        table = pd.read_csv(path).rename(columns={'dataset': 'screen'})
        frames.append(table.groupby(['screen', 'fold', 'cell'], as_index=False)[list(names)].mean().assign(model=model))
    costs = pd.concat(frames, ignore_index=True).drop_duplicates(['screen', 'fold', 'model', 'cell'], keep='last').rename(columns=names)
    changes = pd.read_csv(out / 'RESTORED_LIST.csv')
    changes = changes[changes.kind == 'restoring'][['screen', 'fold', 'model', 'cell', 'levels']]
    table = changes.merge(costs, on=['screen', 'fold', 'model', 'cell'], validate='one_to_one')
    table['raised'] = table.total > 0
    table['common_share'] = table.common / table.total
    table.to_csv(out / 'COMMON_SHARE.csv', index=False)
    up = table[table.raised]
    print(len(table), 'changes,', len(up), 'raised error;', int((up.common_share >= .5).sum()), 'with at least half in the common part')


def plate_free_seed_stage(out):
    """Per Tahoe-100M model, fold and seed: the original and the second training run (the seed averaged with the next
    one), both calibrated toward the plain baseline with ``menu.pick_two`` on calibration lines; the second minus the
    original, and the calibrated original minus the original as trained, on calibration and held-out lines."""
    rows = []
    for screen, fold, model in CASES:
        if screen != 'tahoe':
            continue
        cal = read_side(screen, load_case(screen, fold, model, False), 'calibration', None)
        held = read_side(screen, load_case(screen, fold, model, True), 'held-out', None)
        for s in SEEDS:
            other = net_cost.partner(cal, s)
            mix_c, mix_h = (cal['p'][s] + cal['p'][other]) / 2, (held['p'][s] + held['p'][other]) / 2
            original = menu.pick_two(cal['a'], cal['p'][s], cal['u'])[0]
            second = menu.pick_two(cal['a'], mix_c, cal['u'])[0]
            row = {'fold': fold, 'model': model, 'seed': s, 'original_amounts': original, 'second_amounts': second}
            for side, sd, mix in (('calibration', cal, mix_c), ('held-out', held, mix_h)):
                u, a, p = sd['u'], sd['a'], sd['p'][s]
                base = net_cost.total(p, u)
                o = net_cost.total(menu.two_part(a, p, *original, u), u)
                m = net_cost.total(menu.two_part(a, mix, *second, u), u)
                row[f'{side}_second_minus_original_pts'] = 100 * (m - o) / base
                row[f'{side}_original_field_vs_trained_pts'] = 100 * (o - base) / base
            rows.append(row)
        print(fold, model, 'done', flush=True)
    with open(Path(out) / 'PLATE_FREE_SEED.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plate_free_cards_stage(out):
    """Per Tahoe-100M restoring change: the change over the calibrated original, the second run over it and their
    difference, held-out seed means; and the smallest calibration-line gap of set 1 alone, with the calibrated
    original, and with the second run too."""
    out = Path(out)
    seeds = pd.read_csv(out / 'PLATE_FREE_SEED.csv').set_index(['fold', 'model', 'seed'])
    cards = pd.read_csv(out / 'CARDS.csv', low_memory=False)
    cards = cards[(cards.screen == 'tahoe') & (cards.kind == 'restoring')].set_index('card')

    def gaps(name, column):
        table = pd.read_csv(out / name, dtype={'stage': str})
        rows = []
        for card in cards.index:
            _, fold, model, _ = card.split('|', 3)
            for seed, part in table[table.card == card].groupby('seed'):
                original = part[part.stage == 'variant'].gap_pts.iloc[0]
                rows.append({'card': card, 'model': model, 'seed': seed, 'set1': part[part.stage == '1'].gap_pts.min(), 'original': original,
                             'second': original + seeds.loc[(fold, model, int(seed)), column]})
        return pd.DataFrame(rows)

    calibration = gaps('GATE_CALIBRATION.csv', 'calibration_second_minus_original_pts')
    held = gaps('GATE_HELDOUT.csv', 'held-out_second_minus_original_pts')
    calibration['with_original'] = calibration[['set1', 'original']].min(axis=1)
    calibration['with_second'] = calibration[['set1', 'original', 'second']].min(axis=1)
    least = calibration.groupby('card')[['set1', 'with_original', 'with_second']].min()
    held['seed_gain'] = held.original - held.second
    table = held.groupby(['card', 'model'])[['original', 'seed_gain', 'second']].mean().reset_index()
    table.columns = ['card', 'model', 'change_over_calibrated_original', 'seed_over_calibrated_original', 'change_minus_seed']
    table.merge(least.reset_index(), on='card').to_csv(out / 'PLATE_FREE_CARDS.csv', index=False)


def uniform_case(job):
    """One model and fold: per seed, each arm's gain over the original as trained and the two-part minus the uniform
    arm (total, common, contrast), in points of the original's held-out total; then the seed mean and, where there
    is more than one held-out line, the line-resampled interval of the two-part minus the uniform arm."""
    screen, fold, model = job
    case = load_case(screen, fold, model, True)
    cal = read_side(screen, case, 'calibration', None)
    m = net_cost.calibrate_original(cal)
    T_cal = net_cost.design_baseline(cal, m['kappa'])
    uniform_design = {s: card.calibrate(lambda y, s=s: T_cal + y * (cal['p'][s] - T_cal), cal['u'])[0] for s in SEEDS}
    held = read_side(screen, case, 'held-out', None)
    u = held['u']
    n_lines = len(np.unique([c.split('|')[0] for c in u['context']]))
    rows, lined = [], {t: [] for t in ('field', 'design')}
    for s in SEEDS:
        base = scores(held['p'][s], u)
        a, p, v = held['a'], held['p'][s], m['seeds'][str(s)]
        T = net_cost.design_baseline(held, m['kappa'])
        x = {'two_field': menu.two_part(a, p, *v['original_field'], u),
             'uniform_field': predict('global_parent_shrink', v['cheap']['global_parent_shrink'], p, None, a, u),
             'two_design': menu.two_part(T, p, *v['original_design'], u),
             'uniform_design': T + uniform_design[s] * (p - T)}
        r = {k: scores(y, u) for k, y in x.items()}
        pts = 100 / base['total_mse']
        row = {'screen': screen, 'fold': fold, 'model': model, 'seed': s, 'held_out_lines': n_lines, 'kappa': m['kappa'],
               'amounts_two_field': v['original_field'], 'amount_uniform_field': v['cheap']['global_parent_shrink'],
               'amounts_two_design': v['original_design'], 'amount_uniform_design': uniform_design[s]}
        for k in x:
            row[f'{k}_gain'] = pts * (base['total_mse'] - r[k]['total_mse'])
        for t in ('field', 'design'):
            two, uni = r[f'two_{t}'], r[f'uniform_{t}']
            row[f'{t}_two_minus_uniform'] = pts * (uni['total_mse'] - two['total_mse'])
            row[f'{t}_two_minus_uniform_common'] = pts * (uni['common_error_mse'] - two['common_error_mse'])
            row[f'{t}_two_minus_uniform_contrast'] = pts * (uni['difference_error_mse'] - two['difference_error_mse'])
            lt, lu = line_parts(x[f'two_{t}'], u)['total'], line_parts(x[f'uniform_{t}'], u)['total']
            lined[t].append(({line: lu[line] - lt[line] for line in lt}, base['total_mse']))
        rows.append(row)
    mean = {'screen': screen, 'fold': fold, 'model': model, 'seed': 'mean', 'held_out_lines': n_lines, 'kappa': m['kappa']}
    for k in rows[0]:
        if k.endswith(('_gain', '_two_minus_uniform', '_common', '_contrast')):
            mean[k] = float(np.mean([r[k] for r in rows]))
    for t in ('field', 'design'):
        b = bootstrap(*pool_seeds(lined[t]))
        mean[f'{t}_interval_low'], mean[f'{t}_interval_high'] = (b['interval_pct'] if n_lines > 1 else (None, None))
    print(screen, fold, model, flush=True)
    return rows + [mean]


def uniform_vs_two_part_stage(out, workers=6):
    with Pool(workers) as pool:
        rows = [r for case in pool.imap_unordered(uniform_case, CASES, chunksize=1) for r in case]
    rows.sort(key=lambda r: (r['screen'], r['fold'], r['model'], str(r['seed'])))
    with open(Path(out) / 'UNIFORM_VS_TWO_PART.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def light_cases_stage(out):
    """``OTHER_SCREENS_LIGHT_CASES.csv``: every case of the light version on the screens other than Tahoe-100M, with its
    hand-off kind and status. A case still waiting for its second-seed check is marked so; Jiang et al.'s attention
    synthetic start is not read (its parents learned none of the planted statistics)."""
    names = {'summary': 'summary-statistics', 'deepsets': 'DeepSets', 'state': 'attention', 'official_state': 'official State'}
    cases = pd.concat([pd.read_csv(f, dtype={'target': str}, keep_default_na=False) for f in LIGHT_APPLIED])
    archive = pd.read_csv(LIGHT_ARCHIVE, dtype={'target': str}, keep_default_na=False)
    archive = archive[(archive.dataset == 'replogle') & archive.model.isin(['summary', 'deepsets'])
                      & (archive.primary_seed == 31002) & (archive.window == 0.02)]
    archive = archive.rename(columns={'light_kind': 'kind', 'light_cell': 'cell', 'behaviour_light': 'behaviour'})
    cases = pd.concat([cases, archive], ignore_index=True)
    cases['model'] = cases.model.map(names)
    cases['status'] = ['waiting for the second-seed check' if k == 'unconfirmed' else 'final' for k in cases.kind]
    cases.loc[(cases.dataset == 'jiang') & (cases.model == 'attention') & (cases.half == 'synthetic'), 'status'] = \
        'not read: the parents learned none of the planted statistics'
    columns = ['dataset', 'model', 'fold', 'half', 'target', 'target_name', 'kind', 'cell', 'behaviour', 'status']
    cases = cases[columns].sort_values(['dataset', 'model', 'fold', 'half', 'target']).reset_index(drop=True)
    cases.to_csv(Path(out) / 'OTHER_SCREENS_LIGHT_CASES.csv', index=False)


STAGES = {'calibration': calibration_stage, 'heldout': heldout_stage, 'tables': tables_stage, 'rule': rule_stage,
          'common-share': common_share_stage, 'light-cases': light_cases_stage, 'plate-free-seed': plate_free_seed_stage,
          'plate-free-cards': plate_free_cards_stage, 'uniform-vs-two-part': uniform_vs_two_part_stage}


def main(argv=None):
    parser = argparse.ArgumentParser(description='The matched comparisons of every restoring change and intervention.')
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    STAGES[args.stage](args.out)


if __name__ == '__main__':
    main()
