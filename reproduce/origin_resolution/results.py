"""One command per Results row: the stages that write the tables behind the printed numbers of Results 5-6, the
Fig. 5-6 legends, Supplementary Table 19 and Supplementary Data 3, run in order into one ``--out``.

``STEPS`` names each stage run: its arguments, the files it writes into ``--out`` with the place of each one's copy in
the tables record under the data root (the tables list), the steps whose files it reads from ``--out``, and its BLAS
threads where the saved table was made with a set number. ``ROWS`` names the steps whose tables hold a row's numbers,
with ``SI`` for the numbers in the Supplementary Information's notes and ``SD3`` for every sheet's source and the
workbook. A row's command runs those steps and, first, every step they read from ``--out``; a step whose files are all
in ``--out`` already is not run again, so rows share one ``--out``. Other inputs are read where they are saved under the
data root: the cards, selections and cost readings the stages read by path, each rebuilt by its own steps (row
``upstream``).

    python -m reproduce.origin_resolution.results --row R5P1 --out DIR [--data ROOT] [--compare RECORD]
    python -m reproduce.origin_resolution.results --list > TABLES_LIST.tsv

``--data`` sets ``SCSAILOR_DATA`` for the stages; ``--compare RECORD`` compares the row's files with the tables record
laid out under ``RECORD`` (the data root holds it). The workbook (``SD3``) needs openpyxl and reads the saved tables;
three tables of the rows are made elsewhere and deposited (``INPUTS``).
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

__all__ = ["STEPS", "UPSTREAM", "ROWS", "INPUTS", "closure", "run", "compare", "tables_list", "main"]

PACKAGE = Path(__file__).resolve().parents[2]   # the repository root, which holds reproduce
NCL = f'net-cost'
SSH = f'statistic-swap-shams'
SW6 = f'information-localization/statistic-swap/results'
FM5, FM6 = f'figure-tables/fig5', f'figure-tables/fig6'
CPU = f'metric-and-known-answer-readings'
CX = f'continued-training-readings'
AP = f'origin-resolution/cards/results'
RRF = f'layer-fractions'
JA = f'weight-decay-and-uniform-shrink'
RA = f'training-coverage'
REBUILT = f'source-tables'   # source tables of the paper's numbers, as these stages write them
F23 = {'attention': f'cards-attention-model-folds-2-3/nested',
       'official_state': f'cards-state-folds-2-3/nested'}
FAMILY = ('SCORES', 'GEOMETRY', 'PAIRED', 'LINE_CONTRIBUTIONS', 'SCORE_MEANS')
KINDS = ('CARD', 'ROWS', 'WHERE', 'REFERENCES', 'READINGS')   # a card's tables
GR = f'cards-genetic-screens'
IL_SELECTIONS = f'origin-resolution/restoring-changes/results'
F23_DIR = {'attention': f'cards-attention-model-folds-2-3', 'official_state': f'cards-state-folds-2-3'}
TAHOE_CELLS = {   # fold -> model (IL's name; 'state' is the attention model) -> the restoring changes, as IL_SELECTIONS names them
    'F1': {'deepsets': ('phi-a-x1', 'phi-e-x10'), 'official_state': ('head-a-x1', 'junction-a-x1'), 'state': ('block3-f-x1', 'head-a-x1'),
           'summary': ('projections-e-x30', 'projections-e_mean-x30')},
    'F2': {'deepsets': ('phi-a-x1', 'phi-e-x10'), 'summary': ('projections-e-x30', 'projections-e_mean-x30'), 'state': ('head-a-x1', 'head-e-x1'),
           'official_state': ('head-a-x1', 'junction-e-x1', 'junction-f-x1')},
    'F3': {'deepsets': ('phi-a-x1', 'phi-e-x10'), 'summary': ('projections-e-x30', 'projections-e_mean-x30'), 'state': ('block3-f-x1', 'head-a-x1'),
           'official_state': ('head-a-x1', 'junction-e-x1', 'junction-f-x1')}}
GENETIC_CELLS = (('jiang', 'F4', 'deepsets', 'phi/a/x1'), ('jiang', 'F4', 'summary', 'projections/e/x30'), ('jiang', 'F4', 'summary', 'projections/e_mean/x30'),
                 ('jiang', 'F5', 'deepsets', 'phi/a/x1'), ('jiang', 'F5', 'summary', 'projections/e/x30'), ('jiang', 'F5', 'summary', 'projections/e_mean/x30'),
                 ('replogle', 'F1', 'deepsets', 'phi/a/x1'), ('replogle', 'F1', 'deepsets', 'phi/e/x10'), ('replogle', 'F1', 'summary', 'projections/a/x1'),
                 ('replogle', 'F1', 'summary', 'projections/e/x30'), ('replogle', 'F1', 'summary', 'projections/e_mean/x10'),
                 ('replogle', 'F2', 'deepsets', 'phi/a/x1'), ('replogle', 'F2', 'deepsets', 'phi/e/x10'), ('replogle', 'F2', 'summary', 'projections/a/x1'),
                 ('replogle', 'F2', 'summary', 'projections/e/x1'), ('replogle', 'F2', 'summary', 'projections/e/x30'))


TAHOE_OUTER = tuple(f'il-cells-outer-{f}' for f in ('F1', 'F2', 'F3'))   # the restoring changes' readings per seed


def saved(folder, *names):
    return {name: f'{folder}/{name.split("/")[-1]}' for name in names}


def step(module, *arguments, outputs, needs=(), threads=None):
    return {'argv': (module, *arguments), 'outputs': outputs, 'needs': tuple(needs), 'threads': threads}


def selections_folder(fold, model):
    """Where a Tahoe-100M restoring change's selection is kept: the stages' readings for State and the attention model
    (information localization's 'state') on folds 2-3, information localization's cards otherwise."""
    other = {'state': 'attention', 'official_state': 'official_state'}.get(model)
    return f'{F23_DIR[other]}/il-cells' if other and fold != 'F1' else IL_SELECTIONS


STEPS = {
    # the matched comparisons
    'net-cost-calibration': step('net_cost', '--stage', 'calibration', outputs=saved(
        NCL, 'AMOUNTS.json', 'CARDS_CALIBRATION.csv', 'GATE_CALIBRATION.csv', 'SEED_SLOPES_CALIBRATION.csv', 'MODELS_CALIBRATION.csv',
        'RECOVERY_CALIBRATION.csv', 'FORMULA_CHECK.csv')),
    'net-cost-heldout': step('net_cost', '--stage', 'heldout', needs=['net-cost-calibration'], outputs=saved(
        NCL, 'CARDS_HELDOUT.csv', 'GATE_HELDOUT.csv', 'MODELS_HELDOUT.csv', 'SEED_SLOPES_HELDOUT.csv', 'RECOVERY_HELDOUT.csv')),
    'net-cost-tables': step('net_cost', '--stage', 'tables', needs=['net-cost-calibration', 'net-cost-heldout'], outputs=saved(
        NCL, 'CARDS.csv', 'SEED_BAND.csv', 'SLOPE_SUMMARY.csv', 'LADDER.csv', 'LEDGER.csv', 'RECOVERY_CARDS.csv', 'RECOVERY_SUMMARY.csv',
        'RESTORED_LIST.csv', 'DESIGN_BASELINE.csv')),
    'common-share': step('net_cost', '--stage', 'common-share', needs=['net-cost-tables'],
                         outputs=saved(f'common-share', 'COMMON_SHARE.csv')),
    'light-cases': step('net_cost', '--stage', 'light-cases',
                        outputs=saved(f'light-version-cases', 'OTHER_SCREENS_LIGHT_CASES.csv')),
    'plate-free-seed': step('net_cost', '--stage', 'plate-free-seed', outputs=saved(f'second-run-plain-baseline', 'PLATE_FREE_SEED.csv')),
    'plate-free-cards': step('net_cost', '--stage', 'plate-free-cards', needs=['plate-free-seed', 'net-cost-calibration', 'net-cost-heldout', 'net-cost-tables'],
                             outputs=saved(f'second-run-plain-baseline', 'PLATE_FREE_CARDS.csv')),
    'uniform-vs-two-part': step('net_cost', '--stage', 'uniform-vs-two-part', threads=1, outputs=saved(JA, 'UNIFORM_VS_TWO_PART.csv')),
    # the paper's tables of the matched comparisons
    'figure5': step('paper_tables', '--stage', 'figure5', needs=['net-cost-calibration', 'net-cost-heldout', 'net-cost-tables'], outputs=saved(
        FM5, 'FIG5_SLOPES_AND_KEPT_FRACTION.csv', 'FIG5_GAPS_BY_REFERENCE_AND_SEED.csv', 'FIG5_PREDICTORS_BY_MODEL_AND_FOLD.csv')),
    'rule': step('paper_tables', '--stage', 'rule', needs=['net-cost-tables', 'figure5'], outputs=saved(FM6, 'RECOMMEND_RULE_BY_CHANGE.csv')),
    'figure6c': step('paper_tables', '--stage', 'figure6c', needs=['figure5'],
                     outputs=saved(FM6, 'FIG6_C_CHANGE_SHARE_BY_MODEL_AND_FOLD.csv', 'FIG6_C_CHANGE_BENEFIT.csv')),
    'supplement': step('paper_tables', '--stage', 'supplement',
                       needs=['net-cost-heldout', 'net-cost-tables', 'continued', 'plate-free-cards', 'light-cases', 'run-readings', 'parse-primary'],
                       outputs={**saved(REBUILT, 'SUPP_TABLE_24_PER_CHANGE.csv'),
                                **saved(f'supplementary-tables', 'SUPP_TABLE_24B_RESTORED_BY_FOLD.csv', 'SUPP_TABLE_25B_PER_CHANGE.csv')}),
    'workbook': step('paper_tables', '--stage', 'workbook', outputs={'Supplementary_Data_3_origin_resolution.xlsx': None}),
    # metrics other than squared error
    'metrics-tahoe': step('other_metrics', '--stage', 'tahoe', needs=['net-cost-calibration', 'plate-free-seed'], threads=1,
                          outputs={'METRIC_READINGS_TAHOE.csv': f'metric-readings-tahoe-100m/READINGS.csv'}),
    'metrics-genetic': step('other_metrics', '--stage', 'genetic', needs=['net-cost-calibration'], threads=1,
                            outputs={'METRIC_READINGS_GENETIC.csv': f'{CPU}/genetic-metrics/READINGS.csv'}),
    'metrics-other-screens': step('other_metrics', '--stage', 'other-screens', needs=['net-cost-tables'], threads=1,
                                  outputs={'METRIC_READINGS_OTHER_SCREENS.csv': f'{CPU}/other-screens-metrics/READINGS.csv'}),
    'by-change': step('other_metrics', '--stage', 'by-change', needs=['metrics-tahoe', 'metrics-genetic', 'metrics-other-screens', 'net-cost-tables'],
                      outputs=saved(f'{CPU}/all-screens', 'BY_CHANGE.csv')),
    # recovery from information localization's run records
    'parse-primary': step('recovery_readings', '--stage', 'parse-primary',
                          outputs=saved(f'parse-pbmc-primary-donor-recovery', 'PARSE_PRIMARY_RECOVERY.csv')),
    'run-readings': step('recovery_readings', '--stage', 'runs', outputs=saved(REBUILT, 'FIG4_RUN_READINGS.csv')),
    # the restoring changes, continued training
    **{f'il-cells-calibration-{f}': step('il_cells', '--stage', 'calibration', '--fold', f, outputs={
        f'SELECTIONS_{f}_{model}_{cell}.json': f"{selections_folder(f, model)}/SELECTIONS_{f}_{model}_{cell}.json"
        for model, cells in TAHOE_CELLS[f].items() for cell in cells}) for f in ('F1', 'F2', 'F3')},
    **{f'il-cells-outer-{f}': step('il_cells', '--stage', 'outer', '--fold', f, needs=[f'il-cells-calibration-{f}'],
                                   outputs=saved(f'{CX}/il-cells', f'CARDS_{f}.csv', f'PER_SEED_{f}.csv', f'PER_LINE_{f}.csv'))
       for f in ('F1', 'F2', 'F3')},
    'continued': step('il_cells', '--stage', 'continued', outputs=saved(CX, 'CONTINUED_X1.csv')),
    # Tahoe-100M's cards of the original models: their readings hold the corrections' alignment
    **{f'card-{f}-{m}-{s}': step('cards', '--dataset', 'tahoe', '--model', m, '--fold', f, '--stage', s,
                                 needs=[f'card-{f}-{m}-{earlier}' for earlier in {'calibration': ('selection',) if m == 'official_state' else (),
                                                                                  'attribution': ('calibration',)}.get(s, ())],
                                 outputs={'selection': saved(F23.get(m, AP) if f != 'F1' else AP, f'SELECTION_tahoe_{f}_{m}.json'),
                                          'calibration': saved(F23.get(m, AP) if f != 'F1' else AP, f'CALIBRATION_tahoe_{f}_{m}.json'),
                                          'attribution': saved(F23.get(m, AP) if f != 'F1' else AP, f'ATTRIBUTION_RUN_tahoe_{f}_{m}.json',
                                                               *(f'{k}_tahoe_{f}_{m}.csv' for k in ('CARD', 'ROWS', 'WHERE', 'REFERENCES', 'READINGS')))}[s])
       for f in ('F1', 'F2', 'F3') for m in ('summary', 'deepsets', 'attention', 'official_state')
       for s in (('selection',) if m == 'official_state' else ()) + ('calibration', 'attribution')},
    # the fixed-weight probes
    'swap-scores': step('probes', '--stage', 'swap-scores', outputs=saved(SW6, 'SWAP_SCORES.csv')),
    'swap-summary': step('probes', '--stage', 'swap-summary', needs=['swap-scores'], outputs=saved(REBUILT, 'SWAP_SUMMARY.csv')),
    'state-scores-folds-2-3': step('probes', '--stage', 'state-scores', '--folds', 'F2', 'F3', outputs=saved(
        f'state-statistic-swap-readings-folds-2-3', 'state-folds-2-3/STATE_SCORES.csv', 'state-folds-2-3/STATE_PAIRED.csv', 'state-folds-2-3/STATE_FOLD_MEANS.csv')),
    **{f'scores-{family}': step('probes', '--stage', 'scores', '--family', family, outputs=saved(SSH, *(f'{prefix}_{t}.csv' for t in FAMILY)))
       for family, prefix in (('summary', 'SUMMARY'), ('state', 'ATTENTION'), ('official_state', 'OFFICIAL_STATE'))},
    **{f'unseen-mean-{family}': step('probes', '--stage', 'unseen-mean', '--family', family, needs=[f'scores-{family}'], outputs=saved(
        SSH, *(f'{prefix}_S4_{t}.csv' for t in FAMILY[:4]), *(f'{prefix}_EXTENDED_{t}.csv' for t in FAMILY), f'{prefix}_S4_VS_S2.csv'))
       for family, prefix in (('summary', 'SUMMARY'), ('state', 'ATTENTION'))},
    'geometry': step('probes', '--stage', 'geometry', needs=['scores-summary', 'scores-state', 'unseen-mean-summary', 'unseen-mean-state'],
                     outputs=saved(SSH, *(f'ERROR_GEOMETRY_{t}' for t in ('BACKGROUNDS.csv.gz', 'LINES.csv.gz', 'FOLDS.csv', 'CONTRASTS.csv', 'RESPONSES.csv')))),
    'fractions': step('probes', '--stage', 'fractions', needs=['scores-summary', 'scores-state', 'scores-official_state', 'unseen-mean-summary',
                                                                'unseen-mean-state'], outputs=saved(SSH, 'ADDED_COST_FRACTIONS.csv')),
    'own-mean-value': step('probes', '--stage', 'own-mean-value', needs=['swap-summary', 'unseen-mean-summary', 'unseen-mean-state', 'scores-official_state'],
                           outputs=saved(REBUILT, 'OWN_MEAN_VALUE_H.csv')),
    'control-built': step('probes', '--stage', 'control-scores', '--family', 'built', outputs=saved(
        f'control-use-by-training-length/results', 'control-built/CONTROL_USE.csv', 'control-built/CONTROL_USE_SEED_MEANS.csv')),
    'control-state': step('probes', '--stage', 'control-scores', '--family', 'official_state', outputs=saved(
        f'control-use-by-training-length/official-state/results', 'control-state/CONTROL_USE.csv', 'control-state/CONTROL_USE_SEED_MEANS.csv')),
    # the layers' slopes and the known-answer test
    'slopes-calibration': step('slopes', '--stage', 'calibration', outputs=saved(RRF, 'CALIBRATION_READINGS.csv', 'RELIABILITY.csv', 'INERT_LAYERS.csv')),
    'slopes-heldout': step('slopes', '--stage', 'heldout', needs=['slopes-calibration'], outputs=saved(RRF, 'READINGS.csv')),
    'slopes-tables': step('slopes', '--stage', 'tables', needs=['slopes-heldout'], outputs=saved(
        RRF, 'PRIMARY.csv', 'TOLERANCE_CURVE.csv', 'USE.csv', 'SPECIFICITY.csv', 'SENSITIVITY.csv') | saved(REBUILT, 'RANK_CORRELATION.csv')),
    'read-back': step('planted', '--stage', 'read-back', '--workers', '6', threads=1, outputs=saved(f'{CPU}/known-answer', 'READ_BACK.csv')),
    'margins': step('planted', '--stage', 'margins', needs=['read-back'], outputs={'NULLS_BY_MARGIN.csv': f'{FM6}/KNOWN_ANSWER_NULLS_BY_MARGIN.csv'}),
    # the field's baselines, weight decay, coverage
    'parse-replay': step('baselines', '--stage', 'parse-replay', outputs=saved(REBUILT, 'PARSE_MENU.csv')),
    'baselines': step('baselines', '--stage', 'table', needs=['parse-replay', 'continued'],
                      outputs=saved(REBUILT, 'BASELINES_SIDE_BY_SIDE.csv')),
    'weight-decay-select': step('weight_decay', '--stage', 'select', outputs=saved(JA, 'SELECTION.json')),
    'weight-decay-reading': step('weight_decay', '--stage', 'reading', needs=['weight-decay-select'], threads=1, outputs=saved(JA, 'WEIGHT_DECAY_READING.csv')),
    'coverage': step('coverage', '--stage', 'coverage', threads=3, outputs=saved(RA, 'COVERAGE.csv')),
    **{f'ridge-{f}': step('coverage', '--stage', 'ridge', '--fold', f, threads=3, outputs=saved(RA, f'RIDGE_COMMON_{f}.json')) for f in ('F1', 'F2', 'F3')},
    # the figure tables no other stage writes (Figs. 5b-c, Supplementary Fig. 4b)
    'heldout-parts': step('figure_tables', '--stage', 'heldout-parts', needs=['net-cost-calibration', 'net-cost-tables'],
                          outputs=saved(FM5, 'CHANGE_PARTS_HELDOUT.csv', 'CLOSEST_REFERENCE_HELDOUT.csv')),
    'figure5b': step('figure_tables', '--stage', 'figure5b', needs=['net-cost-tables', 'heldout-parts', *TAHOE_OUTER, 'weight-decay-reading'],
                     outputs=saved(FM5, 'FIG5_B_CHANGES_AND_COMPARATORS.csv')),
    'figure5c': step('figure_tables', '--stage', 'figure5c', needs=['net-cost-tables'], outputs=saved(FM5, 'FIG5_C_CORRECTION_BY_SCREEN.csv')),
    'kept-fraction': step('figure_tables', '--stage', 'kept-fraction', needs=['net-cost-calibration', 'net-cost-tables', 'heldout-parts', *TAHOE_OUTER],
                          outputs=saved(FM6, 'KEPT_FRACTION_INCREMENT.csv')),
}

CARDS = [f'card-{f}-{m}-attribution' for f in ('F1', 'F2', 'F3') for m in ('summary', 'deepsets', 'attention', 'official_state')]


# The tables the steps above read by path, rebuilt from saved predictions by their own stages: the cards with a change in
# the menu and State's at other lengths and seed sets, the genetic screens' cards, the selections, the genetic
# cells' tables and the cost readings. Each writes into ``--out`` at its place in the tables record.
def record(folder, *names):
    return {f'{folder}/{name}': f'{folder}/{name}' for name in names}


def card(dataset, model, fold, folder, *extra):
    """A card's stages into ``folder``: State's selection first where no change is in its menu, then
    calibration and attribution, each reading the one before from the folder."""
    stem = f'{dataset}_{fold}_{model}'
    files = {'selection': [f'SELECTION_{stem}.json'], 'calibration': [f'CALIBRATION_{stem}.json'],
             'attribution': [f'ATTRIBUTION_RUN_{stem}.json', *(f'{kind}_{stem}.csv' for kind in KINDS)]}
    stages = (('selection',) if model == 'official_state' and '--cell' not in extra else ()) + ('calibration', 'attribution')
    steps, before = {}, ()
    for s in stages:
        name = f"{Path(folder).name if Path(folder).name.startswith('card') else 'card-' + Path(folder).name}-{stem}-{s}"
        steps[name] = step('cards', '--dataset', dataset, '--model', model, '--fold', fold, *extra, '--stage', s, needs=before,
                           outputs=record(folder, *files[s]))
        before = (name,)
    return steps


UPSTREAM_CARDS = [
    *[('tahoe', 'deepsets', f, f'{AP}/il-cell-deepsets-phi-e-x10', '--cell', 'phi/e/x10') for f in ('F1', 'F2', 'F3')],
    *[('tahoe', 'summary', f, f'{AP}/il-cell-summary-projections-e_mean-x30', '--cell', 'projections/e_mean/x30') for f in ('F1', 'F2', 'F3')],
    *[('tahoe', model, 'F1', f"{AP}/il-cell-{model}-{cell.replace('/', '-')}", '--cell', cell)
      for model, cell in (('attention', 'block3/f/x1'), ('attention', 'head/a/x1'), ('official_state', 'head/a/x1'), ('official_state', 'junction/a/x1'))],
    *[('tahoe', 'official_state', 'F1', f"{AP}/official-state-{n}-seeds-{seeds.replace(',', '-')}", '--state-length', str(n), '--seeds', seeds)
      for n in (10000, 15000, 20000, 40000) for seeds in ('31002', '31002,31003,31004')],
    ('tahoe', 'official_state', 'F1', f'{AP}/official-state-5000-seeds-31002', '--state-length', '5000', '--seeds', '31002'),
    *[('tahoe', 'official_state', f, f"{F23_DIR['official_state']}/il-cell-official_state-{cell.replace('/', '-')}", '--cell', cell)
      for cell in ('head/a/x1', 'junction/e/x1', 'junction/f/x1') for f in ('F2', 'F3')],
    *[('tahoe', 'attention', f, f"{F23_DIR['attention']}/il-cell-attention-{cell.replace('/', '-')}", '--cell', cell)
      for f, cell in (('F3', 'block3/f/x1'), ('F2', 'head/a/x1'), ('F3', 'head/a/x1'), ('F2', 'head/e/x1'))],
    *[(dataset, model, fold, f"{GR}/il-cell-{model}-{cell.replace('/', '-')}", '--cell', cell) for dataset, fold, model, cell in GENETIC_CELLS],
    *[(dataset, model, fold, f'{GR}/{folder}/nested') for dataset, fold, folder in
      (('replogle', 'F2', 'replogle-nadig-fold-2'), ('jiang', 'F4', 'jiang'), ('jiang', 'F5', 'jiang'))
      for model in ('summary', 'deepsets')],
    *[(dataset, model, 'F1', 'nested-parents-gse306429-replogle-nadig/cards') for dataset in ('gse306429', 'replogle') for model in ('summary', 'deepsets')],
]
BUILT_SCREENS = ('gse306429/F1', 'parse/G1', 'parse/G2', 'parse/G3')
COST_READINGS = (   # (readings folder, runs folder, model, screens and folds)
    ('cost-readings-state', 'state-other-screens', 'official_state',
     ('parse/G1', 'replogle/F2', 'gse306429/F1', 'jiang/F4', 'jiang/F5', 'replogle/F1')),
    ('cost-readings-attention-model', 'attention-model-other-screens', 'state', ('gse306429/F1', 'parse/G1', 'replogle/F1', 'replogle/F2')),
    ('cost-readings-attention-model-jiang', 'attention-model-jiang', 'state', ('jiang/F4', 'jiang/F5')),
    ('cost-readings-built-models/summary', 'built-models-other-screens', 'summary', BUILT_SCREENS),
    ('cost-readings-built-models/deepsets', 'built-models-other-screens', 'deepsets', BUILT_SCREENS))
UPSTREAM = {
    **{name: s for spec in UPSTREAM_CARDS for name, s in card(*spec[:4], *spec[4:]).items()},
    **{f'selections-repair-{f}': step('selections', '--kind', 'repair', '--fold', f,
                                       outputs=record(f'origin-resolution/selections/results', f'SELECTIONS_{f}.json'))
       for f in ('F1', 'F2', 'F3')},
    'selections-nested-F1': step('selections', '--kind', 'nested', '--fold', 'F1', outputs=record(
        'nested-parents-gse306429-replogle-nadig/selections', *(f'SELECTIONS_{d}_{m}.json' for d in ('gse306429', 'replogle') for m in ('deepsets', 'summary')))),
    **{f'selections-nested-{f}': step('selections', '--kind', 'nested', '--fold', f,
                                      outputs=record(f'{GR}/selections', *(f'SELECTIONS_{d}_{f}_{m}.json' for m in ('deepsets', 'summary'))))
       for d, f in (('replogle', 'F2'), ('jiang', 'F4'), ('jiang', 'F5'))},
    **{f'genetic-cells-{s}-{d}-{f}': step('genetic_cells', '--stage', s, '--dataset', d, '--fold', f,
                                          needs=[f'genetic-cells-calibration-{d}-{f}'] if s == 'outer' else [],
                                          outputs=record(f'{GR}/il-cells', *(
                                              [f"SELECTIONS_{d}_{f}_{m}_{c.replace('/', '-')}.json" for d2, f2, m, c in GENETIC_CELLS if (d2, f2) == (d, f)]
                                              if s == 'calibration' else [f'{k}_{d}_{f}.csv' for k in ('CARDS', 'PER_SEED', 'PER_LINE')])))
       for d, f in (('replogle', 'F1'), ('replogle', 'F2'), ('jiang', 'F4'), ('jiang', 'F5')) for s in ('calibration', 'outer')},
    **{folder.replace('/', '-'): step('cost_readings', '--root', runs, '--model', model, '--folds', *folds,
                                               outputs=record(f'{folder}', 'COST.csv', 'ALIGNMENT.csv'))
       for folder, runs, model, folds in COST_READINGS},
}
# These changes' WHERE tables were saved from runs with at most two BLAS threads; with more, the four-part rows change
# in the last digit
for f in ('F1', 'F2', 'F3'):
    UPSTREAM[f'card-il-cell-summary-projections-e_mean-x30-tahoe_{f}_summary-attribution']['threads'] = 2
assert not set(UPSTREAM) & set(STEPS)
STEPS |= UPSTREAM
# Made elsewhere and read where they are deposited: Jiang et al.'s split-half and offset readings were computed from the
# measured cells (``inputs/``); the recovery by training length is information localization's table.
INPUTS = {'jiang-split-half': 'jiang-split-half-reference/JIANG_SPLIT_HALF_POOLED.csv',
          'jiang-offsets': 'jiang-correction-alignment/jiang/F*/OFFSET_READINGS.csv',
          'recovery-by-length': 'information-localization/recovery-by-training-length/TIME_AXIS.csv'}
ROWS = {
    'R5P1': ['net-cost-tables', 'by-change', 'il-cells-outer-F1', 'il-cells-outer-F2', 'il-cells-outer-F3', 'supplement'],
    'R5P2': ['common-share', 'swap-scores'],
    'R5P3': ['fractions'],
    'R5P4': CARDS + ['fractions', 'geometry'],
    'R5P5': ['coverage', 'ridge-F1', 'ridge-F2', 'ridge-F3'],
    'R5P6': ['control-built', 'control-state'],
    'R5P7': ['net-cost-tables', 'slopes-tables'],
    'R5P8': ['baselines', 'weight-decay-reading'],
    'R5P9': ['rule', 'supplement'],
    'R6P1': ['figure5', 'net-cost-tables', 'uniform-vs-two-part', 'weight-decay-reading'],
    'R6P2': ['net-cost-tables'],
    'R6P3': ['net-cost-tables'],
    'R6P4': ['figure6c', 'margins', 'net-cost-tables', 'rule'],
    'R6P5': ['net-cost-tables'],
    # with the tables of the drawn values other steps write: the recommendations (b), the common shares (c), the control
    # use (d); Fig. 6a's calibrated originals, Fig. 6b's change share and Supplementary Fig. 4b's recovery (RECOVERY_CARDS.csv)
    'Fig5': ['net-cost-tables', 'figure5b', 'figure5c', 'common-share', 'control-built', 'control-state', 'rule'],
    'Fig6': ['figure5', 'net-cost-tables', 'figure6c', 'rule', 'kept-fraction'],
    'ST19': ['common-share', 'net-cost-tables', 'rule'],
    # the map's numbers that left Results 5-6 for the Supplementary Information's notes (its rows with no place)
    'SI': CARDS + ['baselines', 'by-change', 'common-share', 'control-built', 'control-state', 'figure5', 'fractions', 'geometry', 'net-cost-tables',
                   'own-mean-value', 'read-back', 'ridge-F1', 'ridge-F2', 'ridge-F3', 'rule', 'slopes-tables', 'state-scores-folds-2-3', 'swap-summary',
                   'uniform-vs-two-part', 'unseen-mean-state', 'weight-decay-reading'],
    # every sheet's source, then the workbook from the saved tables tree
    'SD3': CARDS + ['baselines', 'by-change', 'control-built', 'control-state', 'coverage', 'figure5', 'figure6c', 'fractions', 'geometry',
                    'il-cells-outer-F1', 'il-cells-outer-F2', 'il-cells-outer-F3', 'margins', 'net-cost-tables', 'read-back', 'ridge-F1', 'ridge-F2',
                    'ridge-F3', 'rule', 'slopes-tables', 'state-scores-folds-2-3', 'supplement', 'swap-scores', 'swap-summary', 'uniform-vs-two-part',
                    'unseen-mean-state', 'weight-decay-reading', 'workbook'],
    # the tables those steps read by path, from saved predictions
    'upstream': list(UPSTREAM),
}
ROW_INPUTS = {'R5P6': ['recovery-by-length'], 'SI': ['jiang-split-half', 'jiang-offsets'],
              'SD3': ['jiang-split-half', 'jiang-offsets', 'recovery-by-length']}


def closure(names):
    """The steps to run for ``names``, each after the steps it reads from ``--out``."""
    order = []

    def visit(name):
        if name not in order:
            for earlier in STEPS[name]['needs']:
                visit(earlier)
            order.append(name)
    for name in names:
        visit(name)
    return order


def present(out, name):
    return all(any(out.glob(pattern)) if '*' in pattern else (out / pattern).exists() for pattern in STEPS[name]['outputs'])


def run(names, out, data=None):
    """Run the steps of ``names`` and their producers into ``out``, skipping those whose files are there."""
    out = Path(out).resolve()   # the stages run in the repository root
    env = {**os.environ, **({'SCSAILOR_DATA': str(Path(data).resolve())} if data else {})}
    for name in closure(names):
        if present(out, name):
            print('present:', name, flush=True)
            continue
        s = STEPS[name]
        folder = out / next((Path(o).parent for o in s['outputs'] if '/' in o), Path())
        folder.mkdir(parents=True, exist_ok=True)
        print('run:', name, flush=True)
        subprocess.run([sys.executable, '-B', '-m', f'reproduce.origin_resolution.{s["argv"][0]}', *s['argv'][1:], '--out', str(folder)],
                       cwd=PACKAGE, check=True, env={**env, **({'OMP_NUM_THREADS': str(s['threads'])} if s['threads'] else {})})


VOLATILE = ('wall_seconds', 'peak_rss_mb')   # a run's own wall time and memory, in the JSON records
SORTED = ('METRIC_READINGS_OTHER_SCREENS.csv',)   # written in the order its processes finished


def comparable(value, out):
    """A JSON value without wall time and memory, with ``out`` taken out of its strings."""
    if isinstance(value, dict):
        return {k: comparable(v, out) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [comparable(v, out) for v in value]
    return value.replace(out, '') if isinstance(value, str) else value


def same(new, kept, out):
    """Whether a file a step wrote equals its copy in the tables record, where ``out`` (the run's folder, as the
    records spell it) is taken out of the new file's text."""
    if new.suffix == '.gz':
        return gzip.decompress(new.read_bytes()) == gzip.decompress(kept.read_bytes())
    if new.suffix == '.json':
        return comparable(json.loads(new.read_text()), out) == comparable(json.loads(kept.read_text()), out)
    text = new.read_text().replace(out, '')
    if new.name in SORTED:
        return sorted(text.splitlines()) == sorted(kept.read_text().splitlines())
    return text == kept.read_text()


def compare(names, out, record):
    """Every file the steps of ``names`` wrote into ``out``, against its copy under ``record`` (the tables record laid
    out under the data root); the files that differ or are missing."""
    out = Path(out).resolve()
    differ = []
    for name in closure(names):
        for output, kept in STEPS[name]['outputs'].items():
            if kept is None:
                continue
            new = out / output
            if not new.exists() or not same(new, Path(record) / kept, f'{out}/'):
                differ.append((name, output))
    return differ


def tables_list(handle):
    """Every file a step writes, with its saved copy, the steps it reads from ``--out``, its threads and the rows that run
    it; then the deposited inputs."""
    rows = {}
    for row, names in ROWS.items():
        for name in closure(names):
            rows.setdefault(name, []).append(row)
    writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
    writer.writerow(['step', 'command', 'output', 'saved_copy', 'reads_from_out', 'threads', 'rows'])
    for name, s in STEPS.items():
        for output, copy in s['outputs'].items():
            writer.writerow([name, f"python -m reproduce.origin_resolution.{' '.join(s['argv'])} --out DIR", output, copy or '', ' '.join(s['needs']),
                             s['threads'] or '', ' '.join(rows.get(name, ()))])
    for name, path in INPUTS.items():
        writer.writerow([name, 'deposited', '', path, '', '', ' '.join(r for r, found in ROW_INPUTS.items() if name in found)])


def main(argv=None):
    parser = argparse.ArgumentParser(description='One command per Results row: the stages behind its printed numbers.')
    parser.add_argument('--row', choices=(*ROWS, 'all'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--data', type=Path, help='the data root (SCSAILOR_DATA); default: the environment')
    parser.add_argument('--list', action='store_true', help='print the tables list: every file of every step with its saved copy and rows')
    parser.add_argument('--compare', type=Path, metavar='RECORD',
                        help="after the row's steps, compare their files with the tables record laid out under RECORD")
    args = parser.parse_args(argv)
    if args.list:
        return tables_list(sys.stdout)
    if args.row is None or args.out is None:
        parser.error('--row and --out, or --list')
    names = [n for names in ROWS.values() for n in names] if args.row == 'all' else ROWS[args.row]
    run(names, args.out, args.data)
    if args.compare:
        differ = compare(names, args.out, args.compare)
        for name, output in differ:
            print('DIFFERS', name, output)
        print(f"{sum(1 for n in closure(names) for k in STEPS[n]['outputs'].values() if k) - len(differ)} files equal, {len(differ)} differ")
        sys.exit(1 if differ else 0)


if __name__ == '__main__':
    main()
