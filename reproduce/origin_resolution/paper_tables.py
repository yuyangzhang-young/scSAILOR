"""The paper's tables of the matched comparisons, with its labels, from the ``net_cost`` stages' tables.

Stages, each reading the tables of ``net_cost --stage tables`` (``CARDS.csv``, ``GATE_CALIBRATION.csv``,
``GATE_HELDOUT.csv``, ``DESIGN_BASELINE.csv``) and earlier stages of this module from ``--out``, and writing into it:

- ``figure5``: ``FIG5_SLOPES_AND_KEPT_FRACTION.csv`` (each change's slopes and kept fractions),
  ``FIG5_GAPS_BY_REFERENCE_AND_SEED.csv`` (every kept change against every reference, per seed, on calibration and
  held-out lines; positive: the kept change has the lower error, in points of the original model's total) and
  ``FIG5_PREDICTORS_BY_MODEL_AND_FOLD.csv`` (the model-level values, with the error relative to the baseline);
- ``rule``: ``RECOMMEND_RULE_BY_CHANGE.csv``, the recommendation rule per restoring change counted as
  recovered and per comparison, at three decimals, from the gaps and slopes above and the changes of ``CARDS.csv``
  (the values of ``net_cost --stage rule``); ``recommended_margin_0_2_no_control`` is the saved tables' screen at
  margin 0.2 (``net_cost.recommended``), not the paper's rule;
- ``figure6c``: ``FIG6_C_CHANGE_SHARE_BY_MODEL_AND_FOLD.csv`` (Tahoe-100M: each model's original calibrated toward the
  baseline and toward the design-aware baseline, and each kept change, held-out minus the original as trained, seed
  means; ``refused``: the saved tables' screen at margin 0.2) and ``FIG6_C_CHANGE_BENEFIT.csv`` (the calibrated original's held-out error
  minus the kept change's);
- ``supplement``: Supplementary Tables 24 (``SUPP_TABLE_24_PER_CHANGE.csv``: every restoring change and intervention,
  its held-out recovery of the target, cost at full strength and beside continued training), 24b
  (``SUPP_TABLE_24B_RESTORED_BY_FOLD.csv``: per model and fold, the change restoring the shared and the selective shift)
  and 25b (``SUPP_TABLE_25B_PER_CHANGE.csv``: per change, slopes, kept fraction, the smallest margins and the
  accounting), from ``CARDS.csv``, ``RESTORED_LIST.csv``, ``RECOVERY_CARDS.csv`` and ``GATE_HELDOUT.csv`` of ``net_cost``,
  ``CONTINUED_X1.csv`` (``il_cells --stage continued``), ``PLATE_FREE_CARDS.csv`` and ``OTHER_SCREENS_LIGHT_CASES.csv``
  (``net_cost``), ``FIG4_RUN_READINGS.csv`` and ``PARSE_PRIMARY_RECOVERY.csv`` (``recovery_readings``) in
  ``--out``, the full chain's cost readings and the genetic screens' continued training (``genetic_cells``) where they
  are saved; the attention model's light recovery is read from its runs' records;
- ``workbook --tables ROOT``: Supplementary Data 3, ``Supplementary_Data_3_origin_resolution.xlsx``: the README sheet
  (``WORKBOOK_README``) and the 32 data sheets of ``WORKBOOK_SHEETS``, each built from its source table under ``ROOT``
  (the data root, or a tree laid out as it: the saved tables, or the rebuilt ones) by the sheet's build expression, dropped
  columns, header renames and label changes. Needs openpyxl.

    python -m reproduce.origin_resolution.paper_tables --stage figure5 --out DIR
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from . import ROOT, GENETIC_READINGS, at
from .net_cost import COST_ONLY

__all__ = ["SCREEN", "MODEL", "ARM", "CUT", "REFERENCE", "change_name", "labelled", "slopes", "gaps", "predictors",
           "figure5_stage", "rule_stage", "figure6c_stage", "supplement_stage", "workbook_stage", "main"]

SCREEN = {'tahoe': 'Tahoe-100M', 'gse306429': 'GSE306429', 'parse': 'Parse PBMC', 'replogle': 'Replogle–Nadig', 'jiang': 'Jiang et al.'}
MODEL = {'summary': 'Summary-statistics', 'deepsets': 'DeepSets', 'attention': 'Attention', 'official_state': 'State'}
ARM = {'e': 'front reset', 'f': 'back reset', 'a': 'fine-tuning', 'e_mean': 'mean-projection reset'}
CUT = {'head': 'Head ', 'block3': 'Block 3 ', 'junction': 'Junction ', 'projections': '', 'phi': ''}
LIGHT_MODEL = {'summary-statistics': 'summary', 'DeepSets': 'deepsets', 'attention': 'attention', 'official State': 'official_state'}
TARGET = {'common mean': 'shared shift', 'common variance': 'shared spread', 'selective mean': 'selective shift',
          'statistic-related perturbation identity': 'selective shift'}
# the attention model's light runs: (folder of <fold>/map/, run-name prefix)
ATTENTION_RUNS = {'gse306429': (ROOT / 'information-localization-light/gse306429', 'g'), 'jiang': (ROOT / 'information-localization-light/jiang', 'j'),
                  'replogle': (ROOT / 'information-localization-replogle-nadig/replogle', 'g')}
KEY = ['screen', 'fold', 'model', 'cell']
# Supplementary Data 3, as specified beside this module (workbook/): the sheets (SD3_MAP.tsv: sheet, source, build,
# dropped, renamed, relabelled) and the README sheet's text; a source names a table by an alias of its folder under the tables root
WORKBOOK_SHEETS = Path(__file__).parent / 'workbook/SD3_MAP.tsv'
WORKBOOK_README = Path(__file__).parent / 'workbook/SD3_README.tsv'
WORKBOOK_ALIAS = {'NCL/': 'net-cost', 'ST/': 'supplementary-tables', 'LIB/': 'common-share',
                  'CPU/': 'metric-and-known-answer-readings', 'CX/': 'continued-training-readings', 'SSH/': 'statistic-swap-shams',
                  'SW6/': 'information-localization/statistic-swap/results', 'TA/': 'information-localization/recovery-by-training-length',
                  'CU/': 'control-use-by-training-length/results', 'CUS/': 'control-use-by-training-length/official-state/results',
                  'AP/': 'origin-resolution/cards/results', 'DL/': '',
                  'RA/': 'training-coverage', 'RRF/': 'layer-fractions', 'JA/': 'weight-decay-and-uniform-shrink', 'FM5/': 'figure-tables/fig5',
                  'FM6/': 'figure-tables/fig6', 'SF23/': 'state-statistic-swap-readings-folds-2-3'}
WORKBOOK_DATA = {'JSH/': ROOT / 'jiang-split-half-reference', 'GD/': ROOT / 'jiang-correction-alignment'}   # computed by ``inputs/``
REFERENCE = {'no change': 'no_change', 'baseline': 'baseline', 'original': 'original_as_trained',
             'cell-line shrink': 'cell_line_shrink', 'uniform shrink': 'uniform_shrink', 'shrink toward no change': 'no_change_shrink',
             'design baseline': 'design_baseline', 'original, same calibration': 'original_design_calibrated',
             'second seed, same calibration': 'extra_seed_design_calibrated', 'original, field two-amount calibration': 'original_field_calibrated'}
# the rule's comparisons: (stage of the kept change, stages whose references enter, reference fixed for the own-control gap)
COMPARISONS = {'standard': ('1', ('1',), 'original_as_trained'),
               'calibrated': ('1', ('1', 'variant'), 'original_as_trained'),
               'design-aware': ('2', ('2',), 'original_design_calibrated')}
RULE_MARGIN, EARLIER_MARGIN = 0.1, 0.2
SHARE_STAGE = {'1': ('kept change, toward the baseline', 'Toward the baseline'),
               '2': ('kept change, toward the design-aware baseline', 'Toward the design-aware baseline')}


def change_name(cell):
    cut, arm, rate = cell.split('/')
    name = CUT[cut] + ARM[arm]
    return f"{name[0].upper()}{name[1:]}, ×{rate[1:]}"


def labelled(cards):
    return cards.assign(Screen=cards.screen.map(SCREEN), Model=cards.model.map(MODEL), Change=cards.cell.map(change_name),
                        Type=cards.kind.map({'restoring': 'restoring change', 'control': 'intervention recovering nothing'}))


def slopes(cards):
    keep = {'s_fit_total': 'slope_fit', 's_fit_common': 'slope_fit_common', 's_fit_differential': 'slope_fit_contrast',
            's_cal_total': 'slope_calibration', 's_cal_common': 'slope_calibration_common',
            's_cal_differential': 'slope_calibration_contrast', 's_held_total': 'slope_heldout',
            's_held_common': 'slope_heldout_common', 's_held_differential': 'slope_heldout_contrast',
            's_held_31002': 'slope_heldout_seed1', 's_held_31003': 'slope_heldout_seed2', 's_held_31004': 'slope_heldout_seed3',
            'alpha_min': 'kept_fraction_min', 'alpha_mean': 'kept_fraction_mean', 'kept_held': 'kept_fraction_heldout_slope',
            'calibration_lines': 'calibration_lines', 'levels': 'statistic_level'}
    return cards[['Screen', 'fold', 'Model', 'Change', 'Type', *keep]].rename(columns=keep)


def gaps(cards, out):
    """Every candidate against every reference, per seed, on calibration and held-out lines. Positive: the candidate
    has the lower error, in points of the original model's total."""
    side = [pd.read_csv(out / name, dtype={'stage': str}).assign(lines=lines)
            for lines, name in (('calibration', 'GATE_CALIBRATION.csv'), ('held-out', 'GATE_HELDOUT.csv'))]
    g = pd.concat(side, ignore_index=True)
    g = g.merge(cards[['card', 'Screen', 'fold', 'Model', 'Change', 'Type']], on='card')
    g['reference'] = g.reference.map(REFERENCE)
    g['candidate'] = np.where(g.stage == '1', 'kept change, toward the baseline', 'kept change, toward the design-aware baseline')
    g.loc[g.stage == 'variant', 'candidate'] = 'kept change, toward the baseline'
    return g[['Screen', 'fold', 'Model', 'Change', 'Type', 'lines', 'stage', 'candidate', 'reference', 'seed', 'gap_pts']]


def predictors(out):
    """Model x fold values in points of the original model's total, with the error relative to the baseline."""
    m = pd.read_csv(out / 'DESIGN_BASELINE.csv')
    for c in ('no_change', 'stratum_mean', 'design_baseline', 'original_field_calibrated', 'original_design_calibrated', 'second_seed_design_calibrated'):
        m[f'{c}_relative_to_baseline'] = (1 + m[c] / 100) / (1 + m.baseline / 100)
    m['original_relative_to_baseline'] = 1 / (1 + m.baseline / 100)
    return m.assign(Screen=m.screen.map(SCREEN), Model=m.model.map(MODEL)).drop(columns=['screen', 'model'])


def figure5_stage(out):
    out = Path(out)
    cards = labelled(pd.read_csv(out / 'CARDS.csv', low_memory=False))
    slopes(cards).to_csv(out / 'FIG5_SLOPES_AND_KEPT_FRACTION.csv', index=False)
    gaps(cards, out).to_csv(out / 'FIG5_GAPS_BY_REFERENCE_AND_SEED.csv', index=False)
    predictors(out).to_csv(out / 'FIG5_PREDICTORS_BY_MODEL_AND_FOLD.csv', index=False)


def write_rows(path, rows):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rule_stage(out):
    """A kept change is recommended when, on calibration lines and in every seed, it keeps a positive fraction and has
    lower error, by more than the margin, than every reference of the comparison and than its model's fixed
    intervention calibrated the same way. Gaps are reference minus candidate; the own-control gap is the fixed
    intervention's error minus the change's, against the comparison's fixed reference. Held-out columns are read
    after the rule and take no part in it."""
    out = Path(out)
    cards = labelled(pd.read_csv(out / 'CARDS.csv', low_memory=False))
    kind = {(r.Screen, r.Model, r.fold, r.Change): r.Type for r in cards.itertuples()
            if r.kind == 'control' or str(r.counted) == 'True'}   # interventions, and the changes counted as recovered
    gap = defaultdict(dict)
    for r in csv.DictReader(open(out / 'FIG5_GAPS_BY_REFERENCE_AND_SEED.csv')):
        k = (r['Screen'], r['Model'], r['fold'], r['Change'])
        if k in kind:
            gap[k][(r['lines'], r['stage'], r['reference'], r['seed'])] = float(r['gap_pts'])
    alpha0 = {(r['Screen'], r['Model'], r['fold'], r['Change']): float(r['kept_fraction_min']) == 0
              for r in csv.DictReader(open(out / 'FIG5_SLOPES_AND_KEPT_FRACTION.csv'))}
    control = {k[:3]: k for k, t in kind.items() if t.startswith('intervention')}

    def worst(k, lines, comparison):
        """Per seed, the smallest gap over the comparison's references; then the smallest over seeds."""
        _, stages, _ = COMPARISONS[comparison]
        seeds = sorted({x[3] for x in gap[k] if x[0] == lines})
        return min(min(v for (l, st, _, s2), v in gap[k].items() if l == lines and st in stages and s2 == s) for s in seeds)

    def worst_own(k, lines, comparison):
        stage, _, ref = COMPARISONS[comparison]
        c = control.get(k[:3])
        if c is None:
            return None
        seeds = sorted({x[3] for x in gap[k] if x[0] == lines})
        return min(gap[k][(lines, stage, ref, s)] - gap[c][(lines, stage, ref, s)] for s in seeds)

    rows = []
    for k in sorted(kind):
        if not kind[k].startswith('restoring'):
            continue
        for comparison in COMPARISONS:
            cal, cal_own = worst(k, 'calibration', comparison), worst_own(k, 'calibration', comparison)
            held_own = worst_own(k, 'held-out', comparison)
            rows.append(dict(Screen=k[0], Model=k[1], fold=k[2], Change=k[3], comparison=comparison,
                             recommended=not alpha0[k] and cal > RULE_MARGIN and (cal_own is None or cal_own > RULE_MARGIN),
                             recommended_margin_0_2_no_control=not alpha0[k] and cal > EARLIER_MARGIN,
                             calibration_worst_vs_references=round(cal, 3),
                             calibration_worst_vs_own_control='' if cal_own is None else round(cal_own, 3),
                             heldout_worst_vs_references=round(worst(k, 'held-out', comparison), 3),
                             heldout_worst_vs_own_control='' if held_own is None else round(held_own, 3)))
    write_rows(out / 'RECOMMEND_RULE_BY_CHANGE.csv', rows)
    for screen in dict.fromkeys(r['Screen'] for r in rows):
        print(screen, {c: sum(r['recommended'] for r in rows if (r['Screen'], r['comparison']) == (screen, c)) for c in COMPARISONS})


def figure6c_stage(out):
    """Tahoe-100M, held-out lines, minus the original model as trained, in points of its total MSE, seed means: each
    model's original calibrated (model level) and each change kept, toward the baseline (stage 1) and toward the
    design-aware baseline (stage 2); then each change's benefit, the calibrated original's value minus the kept
    change's. ``refused`` is the saved tables' screen at margin 0.2, not the paper's rule: a seed keeps nothing, or a
    calibration-line gap of 0.2 or less against any reference of the stage."""
    out = Path(out)
    rows = []
    for r in csv.DictReader(open(out / 'FIG5_PREDICTORS_BY_MODEL_AND_FOLD.csv')):
        if r['Screen'] == 'Tahoe-100M':
            for col, group in (('original_field_calibrated', 'Toward the baseline'),
                               ('original_design_calibrated', 'Toward the design-aware baseline')):
                rows.append(dict(Model=r['Model'], fold=r['fold'], Change='', Type='', group=group, situation='Calibrated',
                                 vs_original=float(r[col])))
    alpha0 = {(r['Model'], r['fold'], r['Change']): float(r['kept_fraction_min']) == 0
              for r in csv.DictReader(open(out / 'FIG5_SLOPES_AND_KEPT_FRACTION.csv')) if r['Screen'] == 'Tahoe-100M'}
    held, cal, kind = defaultdict(dict), defaultdict(list), {}
    for r in csv.DictReader(open(out / 'FIG5_GAPS_BY_REFERENCE_AND_SEED.csv')):
        if r['Screen'] != 'Tahoe-100M' or r['stage'] not in SHARE_STAGE or r['candidate'] != SHARE_STAGE[r['stage']][0]:
            continue
        k = (r['Model'], r['fold'], r['Change'], r['stage'])
        kind[k] = r['Type']
        if r['lines'] == 'held-out' and r['reference'] == 'original_as_trained':
            held[k][r['seed']] = -float(r['gap_pts'])
        if r['lines'] == 'calibration':
            cal[k + (r['seed'],)].append(float(r['gap_pts']))
    for k, by_seed in sorted(held.items()):
        m, f, ch, st = k
        refused = alpha0[(m, f, ch)] or any(min(cal[k + (s,)]) <= EARLIER_MARGIN for s in by_seed)
        rows.append(dict(Model=m, fold=f, Change=ch, Type=kind[k], group=SHARE_STAGE[st][1], situation='Kept',
                         vs_original=mean(by_seed.values()), refused=refused))
    cols = ['Model', 'fold', 'Change', 'Type', 'group', 'situation', 'vs_original', 'refused']
    write_rows(out / 'FIG6_C_CHANGE_SHARE_BY_MODEL_AND_FOLD.csv', [{c: r.get(c, '') for c in cols} for r in rows])
    calibrated = {(r['Model'], r['fold'], r['group']): r['vs_original'] for r in rows if r['situation'] == 'Calibrated'}
    write_rows(out / 'FIG6_C_CHANGE_BENEFIT.csv',
               [dict(Model=r['Model'], fold=r['fold'], Change=r['Change'], Type=r['Type'], group=r['group'],
                     benefit=calibrated[(r['Model'], r['fold'], r['group'])] - r['vs_original'], refused=r['refused'])
                for r in rows if r['situation'] == 'Kept'])


def costs(out):
    """Each change's cost (total, common, contrast) and continued training, means over seeds: the matched comparisons'
    from ``CARDS.csv``, the full chain's from the cost readings, continued training on Tahoe from ``CONTINUED_X1.csv`` and
    on the genetic screens from their saved per-seed readings."""
    cards = pd.read_csv(out / 'CARDS.csv', low_memory=False)
    rows = [cards[KEY + ['cost_total_pts', 'cost_common_pts', 'cost_differential_pts']].assign(continued=float('nan'), minus=float('nan'))]
    for path, model, _ in COST_ONLY:
        table = pd.read_csv(path).rename(columns={'dataset': 'screen'})
        means = table.groupby(['screen', 'fold', 'cell'], as_index=False)[['cost_total_pts', 'cost_common_pts', 'cost_differential_pts', 'continued_total_pts', 'change_minus_continued_total_pts']].mean()
        rows.append(means.rename(columns={'continued_total_pts': 'continued', 'change_minus_continued_total_pts': 'minus'}).assign(model=model))
    table = pd.concat(rows, ignore_index=True).drop_duplicates(KEY, keep='last')
    tahoe = pd.read_csv(out / 'CONTINUED_X1.csv')
    tahoe = tahoe.assign(model=tahoe.model.map(LIGHT_MODEL)).groupby(['fold', 'model', 'cell'], as_index=False)[['continued_total_pts', 'change_minus_continued_total_pts']].mean().assign(screen='tahoe')
    genetic = []
    for path in sorted((GENETIC_READINGS / 'il-cells').glob('PER_SEED_*.csv')):
        t = pd.read_csv(path)
        t = t.assign(screen=path.stem.split('_')[2], minus=t.cell_total_pts - t.continuation_total_pts)   # PER_SEED_<screen>_<fold>
        genetic.append(t.groupby(['screen', 'fold', 'model', 'cell'], as_index=False)[['continuation_total_pts', 'minus']].mean())
    genetic = pd.concat(genetic).rename(columns={'continuation_total_pts': 'continued_total_pts', 'minus': 'change_minus_continued_total_pts'})
    table = table.merge(pd.concat([tahoe, genetic], ignore_index=True), on=KEY, how='left')
    table['continued'] = table.continued.fillna(table.continued_total_pts)
    table['minus'] = table.minus.fillna(table.change_minus_continued_total_pts)
    return table.drop(columns=['continued_total_pts', 'change_minus_continued_total_pts'])


def recovery(out):
    """(screen, fold, model, cell) -> 'target value; ...', held-out, seed means: the matched comparisons' from
    ``RECOVERY_CARDS.csv``; the light cases' from their runs (Parse PBMC: ``PARSE_PRIMARY_RECOVERY.csv``)."""
    found = {}
    for r in pd.read_csv(out / 'RECOVERY_CARDS.csv').itertuples():
        screen, fold, model, cell = r.card.split('|')
        found.setdefault((screen, fold, model, cell), []).append((TARGET[r.target], r.heldout_Q_full))
    cases = pd.read_csv(out / 'OTHER_SCREENS_LIGHT_CASES.csv')
    cases = cases[cases.kind == 'qualified repair']
    runs = pd.read_csv(out / 'FIG4_RUN_READINGS.csv')
    primary = pd.read_csv(out / 'PARSE_PRIMARY_RECOVERY.csv')
    for r in cases.itertuples():
        model = LIGHT_MODEL[r.model]
        key = (r.dataset, r.fold, model, r.cell)
        if key in found and r.dataset in ('tahoe', 'jiang', 'replogle') and model in ('summary', 'deepsets'):
            continue
        cut, arm, rate = r.cell.split('/')
        bit = str(r.target)
        if r.dataset == 'parse':
            q = primary[(primary.group == r.fold) & (primary.model == r.model) & (primary.half == r.half) & (primary.cell == r.cell)
                        & (primary.target.astype(str) == bit)].Q_primary.tolist()
        else:
            if model == 'attention':
                root, tag = ATTENTION_RUNS[r.dataset]
                paths = sorted((root / r.fold / 'map').glob(f'{tag}{r.fold}_at_nat_{cut}_{arm}_{rate}_*'))
            else:
                hit = runs[(runs.dataset == r.dataset) & (runs.fold == r.fold) & (runs.model == model) & (runs.half == r.half)
                           & (runs.cut == cut) & (runs.arm == arm) & (runs.rate == float(rate[1:]))]
                paths = [at(x) for x in hit.run]
            q = [json.loads((x / 'EVAL.json').read_text())['after']['supports']['heldout']['bits'][bit]['Q'] for x in paths if (x / 'EVAL.json').is_file()]
        if q:
            label = TARGET[r.target_name] + (' (synthetic-trained)' if r.half == 'synthetic' else '')
            found.setdefault(key, []).append((label, sum(q) / len(q)))
    return {k: '; '.join(f'{name} {value:.2f}' for name, value in v) for k, v in found.items()}


def table24(out):
    listing = pd.read_csv(out / 'RESTORED_LIST.csv')
    table = listing.merge(costs(out), on=KEY, how='left', validate='one_to_one')
    read = recovery(out)
    rows = []
    for r in table.itertuples():
        restoring = r.kind == 'restoring'
        rows.append({'Screen': SCREEN[r.screen], 'Fold': r.fold, 'Model': MODEL[r.model], 'Change': change_name(r.cell),
                     'Type': 'Restoring change' if restoring else 'Intervention recovering nothing',
                     'Statistic level': r.levels if restoring else '',
                     'Held-out recovery of the target': read.get((r.screen, r.fold, r.model, r.cell), '') if restoring else '',
                     'Cost': round(r.cost_pts, 2), 'Common': round(r.cost_common_pts, 2), 'Contrast': round(r.cost_differential_pts, 2),
                     'Cost by seed': r.cost_seeds, 'Continued training': round(r.continued, 2) if pd.notna(r.continued) else '',
                     'Change minus continued training': round(r.minus, 2) if pd.notna(r.minus) else '', 'Raised error': 'yes' if r.cost_pts > 0 else 'no'})
    return pd.DataFrame(rows)


def table25b(out):
    cards = pd.read_csv(out / 'CARDS.csv', low_memory=False)
    free = pd.read_csv(out / 'PLATE_FREE_CARDS.csv').set_index('card')
    gate = pd.read_csv(out / 'GATE_HELDOUT.csv', dtype={'stage': str})
    below_original = gate[(gate.stage == '1') & (gate.reference == 'original')].groupby('card').gap_pts.mean()
    rows = []
    for r in cards.itertuples():
        one = free.loc[r.card] if r.card in free.index else None
        rows.append({'Screen': SCREEN[r.screen], 'Fold': r.fold, 'Model': MODEL[r.model], 'Change': change_name(r.cell),
                     'Type': 'Restoring change' if r.kind == 'restoring' else 'Intervention recovering nothing',
                     'Calibration lines': r.calibration_lines, 'Slope, calibration': round(r.s_cal_total, 3), 'Slope, held-out': round(r.s_held_total, 3),
                     'Kept fraction, mean over seeds': round(r.alpha_mean, 3),
                     'Kept change below the original as trained, held-out': round(below_original[r.card], 2),
                     'Kept change, error relative to the baseline, held-out': round((100 - below_original[r.card]) / (100 + r.baseline_above_original), 3),
                     'Margin, standard references': round(r.min_gap_set1, 2),
                     'Margin, same calibration toward the baseline': round(one.with_original, 2) if one is not None else round(r.min_gap_set1_with_field_calibration, 2),
                     'Margin, extra seed toward the baseline': round(one.with_second, 2) if one is not None else '',
                     'Margin, same calibration toward the design-aware baseline': round(r.min_gap_set2, 2),
                     'Margin, extra seed toward the design-aware baseline': round(r.min_gap_set3, 2),
                     'Held-out margin, standard references': round(r.held_min_gap_set1, 2), 'Held-out margin, design-aware': round(r.held_min_gap_set2, 2),
                     'Held-out margin, extra seed': round(r.held_min_gap_set3, 2),
                     'Baseline to kept change': round(r.gap, 2), 'Design-aware baseline term': round(r.design, 2), "Original's correction term": round(r.correction, 2),
                     'Change term': round(r.change, 2), 'Extra seed term': round(r.second_seed, 2), 'Change term, reverse order': round(r.reverse_change, 2)})
    return pd.DataFrame(rows)


def restored_by_fold(out):
    """Per model and fold, the first change listed for the shared and for the selective shift, if any."""
    listing = pd.read_csv(out / 'RESTORED_LIST.csv')
    full_map = (listing.screen == 'replogle') & listing.model.isin(['summary', 'deepsets'])   # the full maps apply there
    listing = listing[(listing.kind == 'restoring') & ((listing.screen == 'tahoe') | full_map)]
    cases = pd.read_csv(out / 'OTHER_SCREENS_LIGHT_CASES.csv')
    cases = cases[(cases.kind == 'qualified repair') & ~((cases.dataset == 'replogle') & cases.model.isin(['summary-statistics', 'DeepSets']))]
    found = {}
    for r in listing.itertuples():
        for target in r.targets.split('; '):
            found.setdefault((r.screen, r.fold, r.model, TARGET[target]), []).append(r.cell)
    for half in ('natural', 'synthetic'):
        for r in cases[cases.half == half].itertuples():
            found.setdefault((r.dataset, r.fold, LIGHT_MODEL[r.model], TARGET[r.target_name]), []).append(r.cell)
    folds = {'tahoe': ['F1', 'F2', 'F3'], 'gse306429': ['F1'], 'parse': ['G1', 'G2', 'G3'], 'replogle': ['F1', 'F2'], 'jiang': ['F4', 'F5']}
    rows = []
    for model in MODEL:
        for screen, names in folds.items():
            for fold in names:
                if screen == 'parse' and fold != 'G1' and model in ('attention', 'official_state'):
                    continue
                shared, selective = (found.get((screen, fold, model, k), []) for k in ('shared shift', 'selective shift'))
                rows.append({'Model': MODEL[model], 'Screen': SCREEN[screen], 'Fold': fold, 'Shared shift': change_name(shared[0]) if shared else 'none',
                             'Selective shift': change_name(selective[0]) if selective else 'none'})
    return pd.DataFrame(rows)


def supplement_stage(out):
    out = Path(out)
    table24(out).to_csv(out / 'SUPP_TABLE_24_PER_CHANGE.csv', index=False)
    restored_by_fold(out).to_csv(out / 'SUPP_TABLE_24B_RESTORED_BY_FOLD.csv', index=False)
    table25b(out).to_csv(out / 'SUPP_TABLE_25B_PER_CHANGE.csv', index=False)


def workbook_source(source, tables):
    """A sheet's source: its tables read exactly (``float_precision='round_trip'``) and stacked, ``file`` holding each
    row's path relative to its folder; JSON as loaded (a list where the path has a wildcard)."""
    files = []
    for part in source.split(';'):
        part = part.strip()
        alias = next(a for a in sorted({**WORKBOOK_ALIAS, **WORKBOOK_DATA}, key=len, reverse=True) if part.startswith(a))
        root = WORKBOOK_DATA[alias] if alias in WORKBOOK_DATA else Path(tables) / WORKBOOK_ALIAS[alias]
        files += [(root, f) for f in sorted(glob.glob(str(root / part[len(alias):])))]
    if not files:
        raise FileNotFoundError(source)
    if source.endswith('.json'):
        return [json.load(open(f)) for _, f in files] if '*' in source else json.load(open(files[0][1]))
    return pd.concat([pd.read_csv(f, float_precision='round_trip').assign(file=os.path.relpath(f, root)) for root, f in files], ignore_index=True)


def workbook_sheet(m, tables):
    """One sheet of the map: ``build`` over the source ``d`` (default: the table as read, without ``file``), then the
    dropped columns, the header renames and the label changes."""
    d = workbook_source(m['source'], tables)
    t = eval(m['build'], {'d': d, 'pd': pd, 'np': np}) if m['build'] else d.drop(columns='file')
    t = t.drop(columns=[c.strip() for c in m['dropped'].split(';') if c.strip()])
    t = t.rename(columns=dict(x.split(' => ') for x in m['renamed'].split('; ') if x))
    for x in [x for x in m['relabelled'].split('; ') if x]:
        col, pair = x.split(': ', 1)
        old, new = pair.split(' => ')
        t[col] = t[col].replace(old, new)
    return t.reset_index(drop=True)


def workbook_cell(v):
    """A table value as the workbook stores it: numpy scalars as Python values, missing values empty, digit strings
    as integers."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return v


def workbook_stage(out, tables):
    """Floats are written exactly (``repr``): openpyxl's own '%.16g' drops the last digit of some doubles."""
    import openpyxl
    import openpyxl.cell._writer as cell_writer
    from openpyxl.utils import get_column_letter
    wb = openpyxl.Workbook()
    readme = wb.active
    readme.title = 'README'
    for row in csv.reader(open(WORKBOOK_README, encoding='utf-8', newline=''), delimiter='\t'):
        readme.append(row)
    for m in csv.DictReader(open(WORKBOOK_SHEETS, encoding='utf-8'), delimiter='\t'):
        t = workbook_sheet(m, tables)
        ws = wb.create_sheet(m['sheet'])
        ws.append(list(t.columns))
        for record in t.itertuples(index=False):
            ws.append([workbook_cell(v) for v in record])
        ws.freeze_panes = 'A2'
        ws.auto_filter.ref = f'A1:{get_column_letter(t.shape[1])}{t.shape[0] + 1}'
        ws.column_dimensions['A'].width = 24
    safe = cell_writer.safe_string
    cell_writer.safe_string = lambda v: repr(v) if isinstance(v, float) and math.isfinite(v) else safe(v)
    try:
        wb.save(Path(out) / 'Supplementary_Data_3_origin_resolution.xlsx')
    finally:
        cell_writer.safe_string = safe


STAGES = {'figure5': figure5_stage, 'rule': rule_stage, 'figure6c': figure6c_stage, 'supplement': supplement_stage}


def main(argv=None):
    parser = argparse.ArgumentParser(description="The paper's tables of the matched comparisons, with its labels.")
    parser.add_argument('--stage', choices=(*STAGES, 'workbook'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tables', type=Path, help='workbook: the root of the tables tree (default: the saved tables)')
    args = parser.parse_args(argv)
    if args.stage == 'workbook':
        return workbook_stage(args.out, args.tables or ROOT)
    STAGES[args.stage](args.out)


if __name__ == '__main__':
    main()
