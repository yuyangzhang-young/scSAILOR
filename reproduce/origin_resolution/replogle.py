"""Replogle-Nadig's binding, folds 1 and 2: the case from the nested parents, the dataset's documented origin
rows and its evidence for the computed ones.

A context is a whole cell line. Fold 1: RPE1 and Jurkat are fitted, HepG2 calibrates and K562 is held out; fold 2:
K562 and Jurkat are fitted, HepG2 calibrates and RPE1 is held out. Each line is its own screen, so experiment, CRISPRi
effector and screen day coincide with line by design. The conditions without a fit-side training target
(``UNSUPPORTED.json`` of the nested runs) leave the calibration units and the nested parent's held-out units, with
weights recomputed by the datasets' rule.

Fold 1's files: the nested parents' data (``nested-parents-gse306429-replogle-nadig``: ``data/replogle/``,
``scoring/replogle_F1_HELDOUT_TRUTH.npz``); the nested runs' grids and ``UNSUPPORTED.json`` under
``gse306429.RESULTS``; the full parents' grids (``FULL``); the saved selections in ``gse306429.SELECTIONS``; the dataset
references the stages read (``REFERENCES``) and the scorer's split-half reliability (``RELIABILITY``).

Fold 2's files: the fold data of information localization's (IL's) genetic runs (``LATER``), the held-out truth
(``TRUTH``), its nested runs (records and grids, ``NESTED_RUNS``), the full parents' grids beside fold 1's and the
scorer's reliability table. Its D1 noise tables are in ``references/replogle-F2``; the GEM-group term and the knockdown
test are not computed.

With a restoring change (``cell``) the parents and the change's ends come from the change's real-task runs
(``runs.genetic_runs``), and the change's selection is ``genetic_cells``'.
"""
import json

import numpy as np

from scsailor.resolution import comparison
from scsailor.resolution.scoring import identities
from . import (ROOT, REFERENCES, SEEDS, cell_menu_inputs, gse306429, input_cells, listed_unsupported,
          nested_calibration_lines, nested_case, noise_table, read_csv, reliability_rows, runs, selection_file,
          without_references)

__all__ = ["SEEDS", "FOLDS", "PACKAGE", "FULL", "RELIABILITY", "files", "line_order", "line_of", "stratum_of", "calibration_lines",
           "pool_cells", "design_shift", "records", "grid_paths", "case", "menu_inputs", "selection_path", "t7_reading", "evidence",
           "STATIC", "STATIC_LATER"]

FOLDS = ('F1', 'F2')
PACKAGE = gse306429.PACKAGE
FULL = ROOT / 'genetic-screens-predictions/replogle/F1'   # <model>_<seed>_L2/PREDICTIONS.npz; fold 2's beside it
RELIABILITY = ROOT / 'replogle-nadig-unit-reliability/replogle/F1/UNIT_RELIABILITY.csv'
LATER = ROOT / 'information-localization-replogle-nadig/data/replogle_natural'   # <fold>/: fold 2's training data and order
TRUTH = ROOT / 'genetic-screens-predictions/replogle/truth'   # <fold>_HELDOUT_TRUTH.npz
NESTED_RUNS = (ROOT / 'information-localization-replogle-nadig/replogle', ROOT / 'information-localization-replogle-nadig/replogle')   # <fold>/nested/<model>_<seed>/
INPUTS = ROOT / 'replogle-nadig-inputs/inputs'   # <fold>/INPUT_LIBRARY.json
N_FIT = 2
GROUPS = None   # no perturbation grouping is declared


def files(fold='F1'):
    if fold == 'F1':
        return {'train': PACKAGE / 'data/replogle/train/train_targets.npz', 'constants': PACKAGE / 'data/replogle/training_constants.npz',
                'outer': PACKAGE / 'scoring/replogle_F1_HELDOUT_TRUTH.npz', 'order': PACKAGE / 'data/replogle/LINE_ORDER.json',
                'inputs': PACKAGE / 'data/replogle/evaluation_inputs.h5', 'n_fit': N_FIT}
    return {'train': LATER / fold / 'train/train_targets.npz', 'constants': LATER / fold / 'training_constants.npz',
            'outer': TRUTH / f'{fold}_HELDOUT_TRUTH.npz', 'order': LATER / fold / 'LINE_ORDER.json',
            'inputs': LATER / fold / 'evaluation_inputs.h5', 'n_fit': N_FIT}


def line_order(fold):
    return json.loads(files(fold)['order'].read_text())['line_order']


line_of, stratum_of, design_shift = gse306429.line_of, gse306429.stratum_of, gse306429.design_shift   # no design stratum


def calibration_lines(fold='F1'):
    return nested_calibration_lines(files(fold), line_of)


def pool_cells(fold='F1'):
    return input_cells(INPUTS / fold)


def records(model, seed, fold='F1'):
    """A nested run's record folder (``COMPLETE.json``, ``UNSUPPORTED.json``, ``nested-parent/RUN_IDENTITY.json``)."""
    return gse306429.RESULTS / 'replogle' / f'{model}_{seed}' if fold == 'F1' else NESTED_RUNS[0] / fold / 'nested' / f'{model}_{seed}'


def grid_paths(model, seeds, fold='F1', cell=None):
    if cell is not None:
        return runs.genetic_grids(runs.genetic_runs('replogle', fold, model, cell, line_order(fold), N_FIT, seeds))
    if fold == 'F1':
        return {s: {'nested': gse306429.RESULTS / 'replogle' / f'{model}_{s}' / 'predictions-nested.npz',
                    'full': FULL / f'{model}_{s}_L2' / 'PREDICTIONS.npz'} for s in seeds}
    return {s: {'nested': NESTED_RUNS[1] / fold / 'nested' / f'{model}_{s}' / 'predictions-nested.npz',
                'full': FULL.parent / fold / f'{model}_{s}_L2' / 'PREDICTIONS.npz'} for s in seeds}


def case(model, fold='F1', outer=True, seeds=SEEDS, cell=None):
    if fold not in FOLDS:
        raise ValueError(f'replogle has nested parents for folds {FOLDS} only')
    unsupported = listed_unsupported(records(model, s, fold) for s in seeds)
    out = nested_case('replogle', model, files(fold), grid_paths(model, seeds, fold, cell), unsupported, outer, fold)
    if fold != 'F1':   # fold 2: the D1 noise tables only
        ref = REFERENCES / f'replogle-{fold}'
        meta = {'coincident': 'line = its experiment, CRISPRi effector and screen day (Replogle-Nadig design)',
                'type_variable': {'RPE1': 'ZIM3-KRAB, day 7'},
                'composition': ('the fit lines K562 (KOX1-KRAB) and Jurkat (ZIM3-KRAB) and the calibration line HepG2 (KOX1-KRAB) '
                                'against the outer line RPE1 (ZIM3-KRAB)')}
        later = without_references(out, RELIABILITY.parents[1] / fold / RELIABILITY.name, meta, STATIC_LATER)
        later['noise'] = {'fit': noise_table([ref / 'UNIT_NOISE_K562.csv', ref / 'UNIT_NOISE_Jurkat.csv'], out['units']['fit']),
                          'calibration': noise_table(ref / 'UNIT_NOISE_HepG2.csv', out['units']['calibration']),
                          'outer': noise_table(ref / 'UNIT_NOISE_RPE1.csv', out['units']['outer']) if outer else None,
                          'outer_nested': noise_table(ref / 'UNIT_NOISE_RPE1.csv', out['units']['outer_nested']) if outer else None}
        return later
    ref = REFERENCES / 'replogle'
    out['noise'] = {'fit': noise_table([ref / 'UNIT_NOISE_RPE1.csv', ref / 'UNIT_NOISE_Jurkat.csv'], out['units']['fit']),
                    'calibration': noise_table(ref / 'UNIT_NOISE_HepG2.csv', out['units']['calibration']),
                    'outer': noise_table(ref / 'UNIT_NOISE_K562.csv', out['units']['outer']) if outer else None,
                    'outer_nested': noise_table(ref / 'UNIT_NOISE_K562.csv', out['units']['outer_nested']) if outer else None}
    out['reliability'] = RELIABILITY
    out['meta'] = {'coincident': 'line = its experiment, CRISPRi effector and screen day (Replogle-Nadig design)',
                   'type_variable': {'K562': 'KOX1-KRAB, essential screen, day 6'},
                   'composition': ('cells per unit, median: fit lines RPE1 78 and Jurkat 88, calibration line HepG2 53, outer line K562 '
                                   '111; HepG2 shares K562\'s KOX1-KRAB effector, the fit lines carry ZIM3-KRAB')}
    out['static_rows'] = {'static': STATIC, 'fold1_only': (), 'seed_permutation': 'not tested (precondition unavailable)'}
    return out


menu_inputs = cell_menu_inputs   # no plate-aware target or published model; a restoring change's end where the case has one


def selection_path(model, fold, cell, out):
    if fold == 'F1' and cell is None:
        return gse306429.SELECTIONS / f'SELECTIONS_replogle_{model}.json'
    return selection_file('replogle', fold, model, cell)


def t7_reading(path):
    """T7's reading rule at the one-sided level ALPHA: supported when the response ratio falls and
    the excess over no change rises with the knockdown difference beyond the stratified null, for the anchor and both
    parents (all units); unresolved when the directions agree without all four beyond the level; weakened otherwise.
    The test reads associations with the knockdown difference; it does not give knockdown mismatch's causal share."""
    table = {r['quantity']: r for r in read_csv(path) if r['unit_set'] == 'all'}
    keys = ('rho', 'excess_total:anchor', 'excess_total:summary', 'excess_total:deepsets')
    p = {k: float(table[k]['p_one_sided']) for k in keys}
    direction = [float(table['rho']['slope']) < 0] + [float(table[k]['slope']) > 0 for k in keys[1:]]
    beyond = [p[k] <= comparison.ALPHA for k in keys]
    status = ('supported' if all(beyond) and all(direction) else
              'unresolved (directions agree; not all beyond the level)' if all(direction) else 'weakened')
    rho = [float(table['rho'][f'median_q{i}']) for i in range(1, 6)]
    everywhere = all(float(table[k][f'median_q{i}']) > 0 for k in keys[1:] for i in range(1, 6))
    return {'status': status, 'evidence': (f"response ratio along the anchor's own response by knockdown-difference quintile {', '.join(f'{x:.2f}' for x in rho)} "
                                           "(its level is biased toward zero by cell-sampling noise in the training response; only its trend is read); "
                                           + '; '.join(f"{k} p {p[k]:.3f}" for k in keys)
                                           + ('; excess error over no change occurs in every knockdown-difference quintile for the anchor and both parents'
                                              if everywhere else '; the excess over no change is not positive in every quintile')
                                           + "; this comparison does not quantify knockdown mismatch's causal share or separate it from effector, day and "
                                             "line biology (references/replogle/T7_KNOCKDOWN.csv)")}


def evidence(case, held, mean_parts, pool, parent_part, preds):
    """D2 (the within-screen GEM-group term), D4, D7 and T7 for attribution's rows, with their reference rows.
    A later fold has none of fold 1's references: D2 and T7 say so,
    D4 and D8 read the scorer's reliability table."""
    if case['fold'] != 'F1':
        line = ', '.join(np.unique(identities(held)))
        d4, d8 = reliability_rows(case, held, line, 'n_cells')
        return {'d2': {'status': 'not available',
                       'evidence': "no replicate screen; the within-screen GEM-group term is computed for fold 1's K562 only "
                                   "(origin-resolution/cards/references/replogle/D1_GEM_RUN.json)"},
                'd7': {'status': 'confounded by design', 'evidence': f'experiment, effector and day coincide with line; within {line} no '
                                                                     'GEM-group composition check has run'},
                'd4': d4, 'd8': d8,
                't7': {'status': 'not computed', 'evidence': "the knockdown-difference test (references/replogle/T7_KNOCKDOWN.csv) "
                                                             "covers fold 1's outer units only"},
                'reference_rows': [], 'where_rows': []}
    ev, refs, where_rows = {}, [], []
    gem = json.loads((REFERENCES / 'replogle/D1_GEM_RUN.json').read_text())['gem']
    bag, inside = gem['term_against_bag'], gem['term_in_target']
    ev['d2'] = {'status': 'no data between screens',
                'evidence': (f"no replicate screen; within the screen, the GEM-group component estimated from the controls (tau2 "
                             f"{1e3 * gem['tau2']:.2f}e-3 over {gem['groups']} GEM groups) is, against the bag, "
                             f"{100 * bag['common_share_of_common_error']['no_change']:.2f}% of no change's common error "
                             f"({100 * bag['common_share_of_common_error']['anchor']:.2f}% of the anchor's); in the target, an upper reference, "
                             f"{100 * inside['share_of_common_error']['no_change']:.1f}%; a control-derived within-screen term, not between-screen "
                             "replication (origin-resolution/cards/references/replogle/D1_GEM_RUN.json)")}
    refs += [{'reference': f'GEM-group term against the {who} (K562, within the screen)', 'total': t['total'], 'common': t['common'],
              'differential': t['difference'], 'source': 'references/replogle/D1_GEM_RUN.json'}
             for who, t in (('bag', bag), ('pool', gem['term_against_pool']))]
    refs.append({'reference': 'GEM-group term in the target (upper reference; weighted mean per unit)', 'value': inside['weighted_mean'],
                 'source': 'references/replogle/D1_GEM_RUN.json'})
    ev['d7'] = {'status': 'confounded by design',
                'evidence': 'experiment, effector and day coincide with line; within K562, GEM-group composition matches sampling'}
    rel = read_csv(case['reliability'])
    ev['d4'] = {'status': 'reported', 'evidence': f"split-half reliability: {sum(r['reliable'] == 'True' for r in rel)} of {len(rel)} K562 units reliable"}
    ev['t7'] = t7_reading(REFERENCES / 'replogle/T7_KNOCKDOWN.csv')
    ev['reference_rows'], ev['where_rows'] = refs, where_rows
    return ev


# The rows attribution does not compute.
STATIC = {
    'M1': ('needs training and information localization', "no L2 synthetic parents", '8-12 fits, 10-20 V100-h, then information localization'),
    'M2': ('needs training and information localization', "as M1", 'as M1'),
    'M3': ('not applicable', "squared-error training of the scored targets", 'none'),
    'M4': ('not tested', "needs crossed tests", 'fits'),
    'M6': ('not tested', 'checkpoints on the cluster; no symptom scan', 'fits'),
    'D3': ('not tested', "training units have 30-3,472 cells (HepG2 median 53): a strong candidate", 'noise refits: fits'),
    'D5': ('undefined', "no stated reference state", 'a stated reference state'),
    'D6': ('reference', "control distances (K562 nearest: Jurkat)", 'none (split undefined: one calibration line)'),
    'D8': ('see counts', "1,334 outer units (1,297 for the nested parent); K562 30-724 cells per unit (median 111)", 'none'),
    'D9': ('not tested', "no reader fit", 'CPU reader fits'),
    'T1': ('cannot be tested', "one outer line: no matched comparison", 'further folds'),
    'T4': ('holds', "selections written before outer truth", 'none'),
    'T5': ('not applicable', "no synthetic parents", 'none'),
    'T6': ('not tested', "only 40k is saved", 'fits'),
    'T8': ('not applicable', "one line per screen", 'none'),
    'O2': ('not tested', 'anchors not yet compared with the saved baseline', 'local, seconds'),
}

# A later fold's documented rows: fold 1's that hold for the screen, the rest not documented for the fold.
STATIC_LATER = {
    **{k: STATIC[k] for k in ('M3', 'M4', 'M6', 'D5', 'D9', 'T1', 'T4', 'T6', 'T8', 'O2')},
    'M1': ('not tested', "information localization's presence tables are not read by this card", 'none'),
    'M2': ('see the information localization card', "information localization's map of the fold (information-localization-replogle-nadig/MAP_CARRIED*.csv); the restoring "
           "changes' readings and cards are in this program's restoring-change outputs", 'none'),
    'D3': ('not tested', "needs noise refits", 'fits'),
    'D6': ('not tested', "no control-distance reference read for this fold", 'none'),
    'T5': ('not used', "no synthetic reading enters this card", 'none'),
}
