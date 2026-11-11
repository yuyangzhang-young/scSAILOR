"""Jiang et al.'s binding, folds 4 and 5: the case from the nested parents, the dataset's
documented origin rows and its evidence for the computed ones.

A context is one cell line under one stimulus (``A549_IFNB``) and carries no ``|``, so the code reads each context as
a line. The nested split is by cell line: ``LINE_ORDER.json``
lists the 25 training contexts cell line by cell line; the nested parent fits the first 15 (three cell lines) and the
last 10 (two cell lines) are the calibration lines. Fold 4 holds HT29 out and fits K562, HAP1 and A549; fold 5 holds
K562 out and fits HT29, HAP1 and A549; both calibrate on BXPC3 and MCF7. The conditions without a fit-side training
target (``UNSUPPORTED.json`` of the nested runs) leave the calibration units and the nested parent's held-out units,
with weights recomputed by the datasets' rule.

Files: the natural task's training arrays and inputs (``BANK``), the held-out truth and the scorer's reliability table
(``TRUTH``), the line order (``FOLD_DATA``), the nested runs' records and grids (``NESTED_RUNS``) and the full parents'
grids (``FULL``). The D1 noise tables are in ``references/jiang``, computed from the treated cells; no replicate or batch
reference is computed. With a restoring change (``cell``) the parents and the change's ends come from the change's
real-task runs (``runs.genetic_runs``), and the change's selection is ``genetic_cells``'.
"""
import json
from functools import cache

from . import (ROOT, REFERENCES, SEEDS, cell_menu_inputs, input_cells, listed_unsupported, nested_calibration_lines,
          nested_case, noise_table, reliability_rows, runs, selection_file, stratum_design_shift, training_units,
          without_references)

__all__ = ["SEEDS", "FOLDS", "BANK", "TRUTH", "FOLD_DATA", "NESTED_RUNS", "FULL", "files", "line_order", "line_of", "stratum_of",
           "calibration_lines", "pool_cells", "design_shift", "records", "grid_paths", "case", "menu_inputs", "selection_path", "evidence",
           "STATIC"]

FOLDS = ('F4', 'F5')
BANK = ROOT / 'jiang-inputs'   # training-arrays/<fold>/, inputs/<fold>/
TRUTH = ROOT / 'genetic-screens-predictions/jiang/truth'   # <fold>_HELDOUT_TRUTH.npz, <fold>_UNIT_RELIABILITY.csv
FOLD_DATA = ROOT / 'genetic-screens-runs/data/jiang_natural'   # <fold>/LINE_ORDER.json
NESTED_RUNS = (ROOT / 'genetic-screens-runs/jiang', ROOT / 'genetic-screens-runs/jiang')   # <fold>/nested/<model>_<seed>/
FULL = ROOT / 'genetic-screens-predictions/jiang'   # <fold>/<model>_<seed>_L2/PREDICTIONS.npz
N_FIT = 15   # contexts of the nested chain: three cell lines
GROUPS = None   # no perturbation grouping is declared


def files(fold):
    return {'train': BANK / 'training-arrays' / fold / 'train_targets.npz', 'constants': BANK / 'inputs' / fold / 'training_constants.npz',
            'outer': TRUTH / f'{fold}_HELDOUT_TRUTH.npz', 'order': FOLD_DATA / fold / 'LINE_ORDER.json',
            'inputs': BANK / 'inputs' / fold / 'evaluation_inputs.h5', 'n_fit': N_FIT}


def line_order(fold):
    return json.loads(files(fold)['order'].read_text())['line_order']


def line_of(context):
    return context.split('_')[0]


def stratum_of(context):
    """The design stratum of a context: its stimulus."""
    return context.split('_')[1]


def calibration_lines(fold):
    return nested_calibration_lines(files(fold), line_of)


def pool_cells(fold):
    return input_cells(BANK / 'inputs' / fold)


@cache
def fold_training_units(fold):
    f = files(fold)
    return training_units(f['train'], f['constants'])


def design_shift(case, u, nested, bank):
    """Per unit, the perturbation mean within the unit's stimulus minus the mean over stimuli, from the chain's lines."""
    order = line_order(case['fold'])
    lines = sorted({line_of(c) for c in (order[:N_FIT] if nested else order)})
    return stratum_design_shift(u, fold_training_units(case['fold']), case['bags']['train'], lines, line_of, stratum_of)


def records(model, seed, fold):
    """A nested run's record folder (``COMPLETE.json``, ``UNSUPPORTED.json``, ``nested-parent/RUN_IDENTITY.json``)."""
    return NESTED_RUNS[0] / fold / 'nested' / f'{model}_{seed}'


def grid_paths(model, seeds, fold, cell=None):
    if cell is not None:
        return runs.genetic_grids(runs.genetic_runs('jiang', fold, model, cell, line_order(fold), N_FIT, seeds))
    return {s: {'nested': NESTED_RUNS[1] / fold / 'nested' / f'{model}_{s}' / 'predictions-nested.npz',
                'full': FULL / fold / f'{model}_{s}_L2' / 'PREDICTIONS.npz'} for s in seeds}


def case(model, fold, outer=True, seeds=SEEDS, cell=None):
    if fold not in FOLDS:
        raise ValueError(f'jiang has nested parents for folds {FOLDS} only')
    unsupported = listed_unsupported(records(model, s, fold) for s in seeds)
    out = nested_case('jiang', model, files(fold), grid_paths(model, seeds, fold, cell), unsupported, outer, fold)
    later = without_references(out, TRUTH / f'{fold}_UNIT_RELIABILITY.csv',
                               {'coincident': None, 'type_variable': None, 'composition': None}, STATIC)
    table = REFERENCES / 'jiang' / f'UNIT_NOISE_{fold}.csv'
    later['noise'] = {'fit': noise_table(table, out['units']['fit']), 'calibration': noise_table(table, out['units']['calibration']),
                      'outer': noise_table(table, out['units']['outer']) if outer else None,
                      'outer_nested': noise_table(table, out['units']['outer_nested']) if outer else None}
    return later


menu_inputs = cell_menu_inputs   # no plate-aware target or published model; a restoring change's end where the case has one


def selection_path(model, fold, cell, out):
    return selection_file('jiang', fold, model, cell)


def evidence(case, held, mean_parts, pool, parent_part, preds):
    """No replicate or batch reference is computed for Jiang et al.: D2 and D7 say so; D4 and D8 read the scorer's
    reliability table."""
    line = ', '.join(sorted({c.split('_')[0] for c in held['context']}))
    d4, d8 = reliability_rows(case, held, line, 'n_after_cell')
    return {'d2': {'status': 'not computed', 'evidence': 'no replicate reference is computed for Jiang et al.'},
            'd7': {'status': 'not computed', 'evidence': 'no plate, batch or campaign reference is computed for Jiang et al.'},
            'd4': d4, 'd8': d8, 'reference_rows': [], 'where_rows': []}


# The rows attribution does not compute: the built models' facts and the design's, the rest not documented.
STATIC = {
    'M1': ('not tested', "information localization's presence tables are not read by this card", 'none'),
    'M2': ('see the information localization card', "information localization's light version passed on restoring changes "
           "(light-version-cases/applications/JIANG_PARSE_RESTORING_CHANGES.csv); their readings and cards are in this program's "
           "restoring-change outputs", 'none'),
    'M3': ('not applicable', "squared-error training of the scored targets", 'none'),
    'M4': ('not tested', "needs crossed tests", 'fits'),
    'M6': ('not tested', "no symptom scan", 'fits'),
    'D3': ('not tested', "needs noise refits", 'fits'),
    'D5': ('undefined', "no stated reference state", 'a stated reference state'),
    'D6': ('not tested', "no control-distance reference read for this fold", 'none'),
    'D9': ('not tested', "no reader fit", 'CPU reader fits'),
    'T1': ('cannot be tested', "one outer cell line (five contexts): no matched comparison", 'further folds'),
    'T4': ('holds', "selections written before outer truth", 'none'),
    'T5': ('not used', "no synthetic reading enters this card", 'none'),
    'T6': ('not tested', "only 40k is saved", 'fits'),
    'T7': ('not documented', "", 'per-context treatment metadata'),
    'T8': ('not documented', "", 'none'),
    'O2': ('not tested', 'anchors not yet compared with the saved baseline', 'local, seconds'),
}
