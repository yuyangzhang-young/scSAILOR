"""Parse PBMC's binding: one group of donors held out, from the parents trained for origin resolution on Parse PBMC.

A context is one donor's one cell type (``Donor8|B``). Each group has eight training donors: the nested chain fits six
and the other two calibrate (``DONOR_ORDER.json``, the lines' permutation as on Tahoe). Every held-out donor is
evaluated, with the datasets' unit weights. Parse has no dataset references and no restoring change on origin
resolution's nested chain: the binding serves the slopes, the net-cost comparison, the cost readings and the baselines
table, which also reads Parse's repair (its end at step 8,000 on both chains).
"""
import json
from functools import cache

import numpy as np

from scsailor.resolution.scoring import identities, retained
from . import ROOT, SEEDS, bag_means, heldout_units, input_cells, loader, stratum_design_shift, training_units

__all__ = ["SEEDS", "FOLDS", "EXCLUDED", "files", "order", "line_of", "stratum_of", "calibration_lines", "pool_cells", "case", "design_shift"]

FOLDS = ('G1', 'G2', 'G3')
EXCLUDED = ()   # no donor is left out of the evaluation
ORDER = ROOT / 'origin-resolution/parse-pbmc-donor-orders/inputs'   # <group>/DONOR_ORDER.json
STORE = ROOT / 'origin-resolution/parse'   # <group>/nested-repair/<model>-<seed>/, <group>/real/<model>-<seed>/
INPUTS = ROOT / 'parse-pbmc-inputs/inputs'
ARRAYS = ROOT / 'training-arrays-parse-pbmc-tahoe-100m/training-arrays'
TRUTH = ROOT / 'parse-pbmc-held-out-truth/natural'   # <group>/HELDOUT_TRUTH.npz


def files(group):
    return {'train': ARRAYS / f'parse_outer_{group}/train_targets.npz', 'constants': INPUTS / group / 'training_constants.npz',
            'outer': TRUTH / group / 'HELDOUT_TRUTH.npz', 'inputs': INPUTS / group / 'evaluation_inputs.h5', 'excluded': EXCLUDED}


def order(group):
    """``line_order``, ``nested_fit`` and ``calibration``: the group's donors."""
    return json.loads((ORDER / group / 'DONOR_ORDER.json').read_text())


def line_of(context):
    return context.split('|')[0]


def stratum_of(context):
    """The design stratum of a context: its cell type."""
    return context.split('|')[1]


def calibration_lines(group):
    return order(group)['calibration']


def pool_cells(group):
    return input_cells(INPUTS / group)


@cache
def fold_training_units(group):
    f = files(group)
    return training_units(f['train'], f['constants'])


def case(model, fold, outer=True, seeds=SEEDS):
    """The calibration donors on the nested chain, the held-out donors on both chains; ``grids``: seed -> the
    parents ('nested', 'full') and the repair's ends ('cell_nested', 'cell_full')."""
    f = files(fold)
    train = fold_training_units(fold)
    units = {'calibration': retained(train, np.isin(identities(train), order(fold)['calibration']))}
    if outer:
        units['outer'] = units['outer_nested'] = heldout_units(f['outer'], EXCLUDED)
    paths = {}
    for s in seeds:
        paths[s] = {}
        for chain, kind in (('nested', 'nested-repair'), ('full', 'real')):
            folder = STORE / fold / kind / f'{model}-{s}'
            parent = folder / 'predictions-parent.npz'
            paths[s][chain] = parent if parent.exists() else folder / 'predictions-step00000.npz'
            paths[s][f'cell_{chain}'] = folder / 'predictions-step08000.npz'
    return {'units': units, 'grids': paths, 'load': loader(paths), 'bags': bag_means(f['inputs']), 'fold': fold}


def design_shift(case, u, nested, bank):
    """Per unit, the perturbation mean within the unit's cell type minus the mean over cell types, from the chain's donors."""
    donors = order(case['fold'])['nested_fit' if nested else 'line_order']
    return stratum_design_shift(u, fold_training_units(case['fold']), case['bags']['train'], donors, line_of, stratum_of)
