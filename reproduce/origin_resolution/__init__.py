"""Origin resolution's results from the deposited data: where the deposited files are, the dataset bindings and the
stages that read and write them. The method is ``scsailor.resolution``; nothing here is method.

The data root: the environment variable ``SCSAILOR_DATA`` names the folder that holds the deposited files
(``reproduce/data/FOLDERS.tsv`` lists its folders), and every constant is a path under it. Paths stored in saved files
are read through ``at`` (relative to the root; a file that was not deposited reads as ``not-deposited:<name>`` and is
never opened) and written through ``stored`` (relative to the root).

Dataset bindings (each builds the case of ``scsailor.resolution.stages`` for one model and fold, and holds the dataset's
documented origin rows and its evidence for attribution's rows):

* ``tahoe``: Tahoe-100M folds 1-3, the development data: the built models' parents from information localization's
  (IL's) real-task runs, the attention model's and State's parents on origin resolution's grid, the plate-aware target
  of the menu;
* ``gse306429`` and ``replogle``: fold 1 of GSE306429 and Replogle-Nadig, from their nested parents; Replogle-Nadig's
  fold 2 and ``jiang`` (Jiang et al., folds 4 and 5), from the nested parents of the genetic screens' runs
  (``information-localization-replogle-nadig``, ``genetic-screens-runs``), with their noise tables as the only dataset
  references;
* ``parse``: Parse PBMC's three groups, for the slopes, the matched comparisons and the cost readings.

Each binding also answers ``line_of``, ``stratum_of``, ``calibration_lines``, ``pool_cells`` and ``design_shift``, the
facts of a screen that ``slopes`` and ``net_cost`` read through their sides.

Stages, each a command-line entry (``python -m reproduce.origin_resolution.<stage> --help``) that writes its files once
into ``--out``:

* ``selections``: the selections the stages read as saved, made on calibration units before them (the repair
  selections on Tahoe-100M; the parent-only menu on GSE306429 and Replogle-Nadig);
* ``cards``: one model on one fold (selection, calibration, attribution and the held-out check);
* ``il_cells``: recovery and cost of every restoring change IL names on Tahoe-100M (``runs`` finds its runs);
  ``genetic_cells`` the same readings on the genetic screens, with a second training run beside continued training;
* ``slopes``: the layers' slopes on every screen, model and fold;
* ``net_cost``: the matched comparisons of every change, and the recommendation rule from their saved tables;
* ``planted``: the known-answer test, planted changes on every model and fold read back from saved grids;
* ``cost_readings``: cost beside continued training and alignment on the screens read on the full chain only;
* ``probes``: the fixed-weight probes' tables on Tahoe-100M, from their saved forward passes;
* ``baselines``, ``coverage``, ``other_metrics``, ``recovery_readings`` and ``weight_decay``: the field's baselines
  beside origin resolution's action, training coverage, the metrics other than squared error, recovery read from IL's
  run records, and weight decay as a comparator;
* ``paper_tables``: the paper's tables of the matched comparisons and the Supplementary Data 3 workbook.
* ``figure_tables``: the figure tables no other stage writes (Figs. 5b-c, Supplementary Fig. 4b).

``results`` runs the stages behind each Results row, in order, into one folder; ``one_case`` runs one card from its 18
deposited inputs and compares it with ``one_case_expected/``. The dataset references the stages read, the nested
parents and the probes' forward passes were computed from raw cells or trained weights and are deposited as inputs
(README.md).

This module holds the readers the bindings share.
"""
import csv
import json
import os
from functools import cache
from pathlib import Path

import h5py
import numpy as np

from scsailor.resolution import scoring
from scsailor.resolution.net_cost import stratum_shift   # not ``net_cost``: importing the stage ``net_cost`` rebinds that name

# the data root; unset, every read fails on a path that names the variable
ROOT = Path(os.environ.get('SCSAILOR_DATA', 'SCSAILOR_DATA-not-set'))


def at(path):
    """A path given by a saved file, as this machine reads it: relative to the data root (or already under it)."""
    path = Path(path)
    if path.is_absolute() and not path.is_relative_to(ROOT):
        raise ValueError(f'{path} lies outside the data root {ROOT}')
    return path if path.is_absolute() else ROOT / path


def name_of(path):
    """The file or folder name a saved path ends in (a file that was not deposited reads as ``not-deposited:<name>``)."""
    return Path(str(path).removeprefix('not-deposited:')).name


def stored(path):
    """A path as a saved file records it: relative to the data root when it lies under one, otherwise as it is (the
    stage's own output folder)."""
    return Path(path).relative_to(ROOT).as_posix() if Path(path).is_relative_to(ROOT) else str(path)


SEEDS = (31002, 31003, 31004)   # the three training seeds of every model
# The dataset references the stages read, per dataset
REFERENCES = ROOT / 'origin-resolution/cards/references'
# The genetic screens' selections: the parent-only menu's (``selections``) in selections/, a restoring change's
# (``genetic_cells``) in il-cells/
GENETIC_READINGS = ROOT / 'cards-genetic-screens'
# State on Tahoe folds 2-3: its runs (state-folds-2-3; <fold>/ holds st_anchored_or21_L2_<seed>/,
# full_train28_L2_<seed>/, continued_train28_L2_<seed>/ and natural/) and the stages' readings of them (il-cells/: the
# restoring changes' selections)
STATE_F23 = ROOT / 'state-folds-2-3'
STATE_F23_READINGS = ROOT / 'cards-state-folds-2-3'
# The attention model on Tahoe folds 2-3: its runs (attention-model-folds-2-3; <fold>/natural/ and
# continued-training/<fold>/state_<seed>_L2/) and the stages' readings of them (il-cells/ as for State)
ATTENTION_F23 = ROOT / 'attention-model-folds-2-3'
CONTINUED_X1 = ROOT / 'continued-training-runs'   # continued training of the whole network (``il_cells``)
ATTENTION_F23_READINGS = ROOT / 'cards-attention-model-folds-2-3'


def read_csv(path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def write_once(path, value):
    """A JSON record, written once."""
    text = json.dumps(value, indent=2, allow_nan=False, default=lambda x: None if x is None else float(x)) + '\n'
    with Path(path).open('x') as handle:
        handle.write(text)


def write_csv(path, table, lineterminator='\r\n', mode='x'):
    """A table, written once unless ``mode`` says otherwise; its columns are every row's keys in order of appearance."""
    with Path(path).open(mode, newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in table for k in r)), lineterminator=lineterminator)
        writer.writeheader()
        writer.writerows(table)


@cache
def bag_means(path):
    """side -> context -> (mean of the input bag, pool mean C), both float64."""
    out = {}
    with h5py.File(path, 'r') as bank:
        for side in bank:
            out[side] = {k: (bank[side][k]['x'][:].astype(np.float64).mean(axis=0), bank[side][k]['C'][:].astype(np.float64))
                         for k in bank[side]}
    return out


def per_unit(table, units, column):
    """A per-unit column of a CSV keyed by (context, p_code), in the units' order."""
    with open(table) as handle:
        lookup = {(r['context'], int(r['p_code'])): float(r[column]) for r in csv.DictReader(handle)}
    return np.asarray([lookup[(c, int(p))] for c, p in zip(units['context'], units['p_code'])])


def noise_table(paths, units):
    """Per-unit noise from one or more references tables (context, p_code, noise)."""
    paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    lookup = {}
    for x in paths:
        with open(x) as handle:
            lookup |= {(r['context'], int(r['p_code'])): float(r['noise']) for r in csv.DictReader(handle)}
    return np.asarray([lookup[(c, int(q))] for c, q in zip(units['context'], units['p_code'])])


def training_units(targets, constants):
    """A fold's training units: target, context, perturbation code and weight per unit."""
    with np.load(targets) as z, np.load(constants) as c:
        return {'truth': z['target'].astype(np.float64), 'context': c['context_keys'].astype(str)[z['parent']],
                'p_code': z['p_code'].astype(np.int64), 'weight': z['weight'].astype(np.float64)}


def heldout_units(path, excluded=()):
    """A fold's held-out units (reads held-out truth), without the lines ``excluded`` from the evaluation."""
    with np.load(path) as z:
        units = {'truth': z['truth'].astype(np.float64), 'context': z['context'].astype(str),
                 'p_code': z['p_code'].astype(np.int64), 'weight': z['weight'].astype(np.float64)}
    return scoring.retained(units, ~np.isin(scoring.identities(units), excluded)) if excluded else units


def gene_arrays(files):
    """The training arrays the gene groups read (``stages.gene_groups``): the model constants, the training
    units and, per training context, its control cells' gene variances and pool mean."""
    with np.load(files['constants']) as c:
        c_train, keys = c['c_train'].astype(np.float64), c['context_keys'].astype(str)
    with np.load(files['targets']) as z:
        target, parent, codes = z['target'].astype(np.float64), z['parent'], z['p_code'].astype(np.int64)
    var, pools = [], {}
    with h5py.File(files['controls'], 'r') as bank:
        for k in bank:
            x = bank[k]['input/x'][:].astype(np.float64)
            var.append(x.var(axis=0, ddof=1))
            pools[int(k)] = x.mean(axis=0)
    return {'c_train': c_train, 'context_keys': keys, 'target': target, 'parent': parent, 'codes': codes,
            'variance': var, 'pools': pools}


def input_cells(folder):
    """'<side>/<context>' -> the cells of the context's control input pool, from a fold's ``INPUT_LIBRARY.json``."""
    bags = json.loads((Path(folder) / 'INPUT_LIBRARY.json').read_text())['evaluation_bags']
    return {k: v['input_pool_cells'] for k, v in bags.items()}


def nested_calibration_lines(files, line_of):
    """The lines of a nested chain's calibration contexts, from a binding's ``files``."""
    order = json.loads(files['order'].read_text())['line_order']
    return sorted({line_of(c) for c in order[files['n_fit']:]})


def stratum_design_shift(u, train, train_bags, lines, line_of, stratum_of):
    """``net_cost.stratum_shift`` from the training units of ``lines``, for a screen whose design stratum is read from
    its context keys."""
    strata = lambda contexts: np.asarray([stratum_of(c) for c in contexts])
    keep = np.isin(np.asarray([line_of(c) for c in train['context']]), lines)
    return stratum_shift(u, strata(u['context']), train, train_bags, keep, strata(train['context']))


def selection_file(dataset, fold, model, cell=None):
    """Where a genetic screen's selection of a later fold, or of a restoring change, is kept (``GENETIC_READINGS``)."""
    if cell:
        return GENETIC_READINGS / 'il-cells' / f"SELECTIONS_{dataset}_{fold}_{model}_{cell.replace('/', '-')}.json"
    return GENETIC_READINGS / 'selections' / f'SELECTIONS_{dataset}_{fold}_{model}.json'


def listed_unsupported(folders):
    """The conditions without a fit-side training target, as every nested run in ``folders`` lists them."""
    found = {tuple(json.loads((Path(f) / 'UNSUPPORTED.json').read_text())['conditions']) for f in folders}
    if len(found) != 1:
        raise ValueError(f'the nested runs list different unsupported conditions: {sorted(map(str, folders))}')
    return list(found.pop())


def without_references(case, reliability, meta, static):
    """A case without the dataset references of fold 1: the scorer's split-half reliability table (``reliability``, read
    by ``reliability_rows``) and the binding's notes and documented rows."""
    return case | {'reliability': reliability, 'meta': meta, 'static_rows': {'static': static, 'fold1_only': (), 'seed_permutation': 'not tested (precondition unavailable)'}}


def reliability_rows(case, held, line, column):
    """D4 and D8 of a later fold from the scorer's split-half reliability table of the held-out units (one row per
    unit, its cell count in ``column``): the reliable units, the outer units and their cells."""
    rel = read_csv(case['reliability'])
    cells = np.asarray([int(r[column]) for r in rel])
    source = f"{case['reliability'].parent.name}/{case['reliability'].name}"
    return ({'status': 'reported', 'evidence': f"split-half reliability: {sum(r['reliable'] == 'True' for r in rel)} of {len(rel)} "
                                               f"{line} units reliable ({source})"},
            {'status': 'see counts', 'evidence': f"{len(held['truth'])} outer units ({len(case['units']['outer_nested']['truth'])} for the "
                                                 f"nested parent); {line} {cells.min()}-{cells.max()} cells per unit (median "
                                                 f"{np.median(cells):g}; {source})"})


def cell_menu_inputs(case, which, seeds, a):
    """``menu.side``'s inputs on a screen without a plate-aware target or a published model: with a restoring change, its
    end on the side's chain ('calibration': nested, on the calibration units; 'outer': full, on the held-out units)."""
    if 'cell_full' not in case['grids'][seeds[0]]:
        return {}
    u = case['units']['calibration' if which == 'calibration' else 'outer']
    chain, side = ('nested', 'train') if which == 'calibration' else ('full', 'outer')
    return {'repair_end': lambda s: case['load'](s, f'cell_{chain}', side, u)}


def loader(grid_paths):
    """``load(seed, chain, side, units, member='raw')`` over seed -> chain -> prediction file."""
    from scsailor.resolution.grids import load
    return lambda seed, chain, side, units, member='raw': load(grid_paths[seed][chain], side, units, member)


def nested_case(dataset, model, files, grid_paths, unsupported, outer=True, fold='F1'):
    """A screen's fold from its nested parents: the fit and calibration units by the fixed line order, the calibration and nested
    held-out units without the conditions that have no fit-side training target, weights recomputed by the datasets'
    rule."""
    train = training_units(files['train'], files['constants'])
    held = heldout_units(files['outer']) if outer else None
    order = json.loads(files['order'].read_text())['line_order']
    fit, calibration = order[:files['n_fit']], order[files['n_fit']:]
    drop = np.asarray(unsupported, dtype=np.int64)
    lines = scoring.identities(train)
    eligible = lambda u: ~np.isin(u['p_code'], drop)
    units = {'fit': scoring.retained(train, np.isin(lines, fit)),
             'calibration': scoring.retained(train, np.isin(lines, calibration) & eligible(train)),
             'outer': held, 'outer_nested': scoring.retained(held, eligible(held)) if outer else None}
    return {'dataset': dataset, 'fold': fold, 'model': model, 'units': units, 'grids': grid_paths, 'load': loader(grid_paths),
            'bags': bag_means(files['inputs']), 'fit': fit, 'calibration_lines': calibration,
            'train_files': {'targets': files['train'], 'constants': files['constants'],
                            'controls': files['train'].parent / 'controls.h5'}, 'gene_variance': None}
