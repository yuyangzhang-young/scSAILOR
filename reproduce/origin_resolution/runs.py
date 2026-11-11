"""What information localization (IL) passes to origin resolution: where its map tables and real-task runs are, and how
a restoring change's runs are found and checked.

IL's map tables (``MAP_CARRIED<suffix>.csv``, ``MAP_CELLS<suffix>.csv``) name the restoring changes per model and
target: the statistic-related perturbation identity (no suffix), the common mean (``_bit0``) and the common variance
(``_bit5``). A cell is ``<cut>/<arm>/x<rate multiple>``.

A real-task (setting-3) run of a cell, as IL's runner writes it:

- ``RUN_RECORD.json`` with ``row`` (stage ``natural``, model, fold, seed, n_lines, cut, arm, rate_multiple and the
  fold's files: train_directory, constants_path, heldout_units_path, line_subsets), ``updates`` and ``lines``, the
  lines the parent and the run trained on (the first n_lines of the fold's line order);
- ``COMPLETE.json`` written last, or ``DIVERGED.json``;
- ``predictions-parent.npz`` (the parent) and the read-out grid, ``predictions-step<updates>.npz``, or
  ``predictions-average.npz`` for PrePR-CT and CPA (the final-quarter weight mean), on origin resolution's grid.

Each cell has six runs: seeds 31002-31004 on the 28-line parent (``n28``) and on the 21-line nested parent (``n21``).
Models by IL's names: summary, deepsets, state (the built attention model), official_state, prepr, cpa.

On the genetic screens (``genetic_runs``) a cell's six runs are on the full parent and on origin resolution's nested
parent, and continued training (``COMPARATOR``) has six runs of its own; records and grids are under the same
relative paths (``GENETIC``). The cells' own parent grids are not saved: a cell's parent is read from the start grid of
its parent's continued-training run, which every run of that parent shares.
"""
import csv
import json
from pathlib import Path

from scsailor.resolution.il_cells import AVERAGED
from . import ATTENTION_F23, ROOT, SEEDS, STATE_F23, name_of

__all__ = ["SEEDS", "AVERAGED", "MODELS", "TARGETS", "GENETIC", "COMPARATOR", "find_runs", "read_run", "check_record",
           "cell_runs", "genetic_runs", "genetic_grids", "carried_cells", "synthetic_card"]

# IL's map tables (beside its map folder) and real-task runs, per fold and model
LOCAL = {'map': ROOT / 'information-localization-maps-fold-1', 'runs': ROOT / 'information-localization-runs-fold-1/natural'}
GPU = {'state': ROOT / 'information-localization-attention-model', 'official_state': ROOT / 'information-localization-state',
       'prepr': ROOT / 'information-localization-published-models/prepr', 'cpa': ROOT / 'information-localization-published-models/cpa'}
MODELS = {'F1': {'summary': LOCAL, 'deepsets': LOCAL,
                 **{model: {'map': folder, 'runs': folder / 'natural'} for model, folder in GPU.items()}},
          **{fold: {**dict.fromkeys(('summary', 'deepsets'), {'map': ROOT / f'information-localization-maps-fold-{fold[1]}',
                                                              'runs': ROOT / f'information-localization-runs-fold-{fold[1]}/natural'}),
                    # the attention model and State: IL's map tables beside its fold 2-3 map runs
                    'state': {'map': ROOT / 'information-localization-folds-2-3' / fold, 'runs': ATTENTION_F23 / fold / 'natural'},
                    'official_state': {'map': ROOT / 'information-localization-folds-2-3' / fold, 'runs': STATE_F23 / fold / 'natural'}}
             for fold in ('F2', 'F3')}}
ROW_FILES = {'train_directory': 'train_targets', 'constants_path': 'constants', 'heldout_units_path': 'outer',
             'line_subsets': 'line_subsets'}   # a run's row field -> the fold file it must name
# IL's three planted targets and the suffix of their tables
TARGETS = (('', 'statistic-related perturbation identity'), ('_bit0', 'common mean'), ('_bit5', 'common variance'))
# The genetic screens' real-task runs, per screen (records root, grids root): <root>/<fold>/natural/<run>/. Replogle-
# Nadig's restoring changes ran with IL's genetic runs, everything else with origin resolution's.
GENETIC = {'replogle': [(ROOT / 'information-localization-replogle-nadig/replogle', ROOT / 'information-localization-replogle-nadig/replogle'),
                        (ROOT / 'genetic-screens-runs/replogle', ROOT / 'genetic-screens-runs/replogle')],
           'jiang': [(ROOT / 'genetic-screens-runs/jiang', ROOT / 'genetic-screens-runs/jiang')]}
COMPARATOR = 'input/a/x1'   # continued training: the whole network, no reset, at the parent's rate, the cells' recipe


def find_runs(folder):
    """IL's setting-3 runs in a folder, by their record's row: (model, fold, seed, lines, cell) -> run folders."""
    found = {}
    for record in sorted(Path(folder).glob('*/RUN_RECORD.json')):
        saved = json.loads(record.read_text())
        if 'row' not in saved:
            print(f'not read, no row in its record: {record.parent}', flush=True)
            continue
        row = saved['row']
        if row.get('stage') == 'natural':
            cell = f"{row['cut']}/{row['arm']}/x{float(row['rate_multiple']):g}"
            found.setdefault((row['model'], row['fold'], int(row['seed']), f"n{row['n_lines']}", cell), []).append(record.parent)
    return found


def read_run(folder, model):
    """State of one real-task run, its record, and its parent and read-out grids."""
    if (folder / 'DIVERGED.json').is_file():
        return {'state': 'diverged at update {update} ({what})'.format(**json.loads((folder / 'DIVERGED.json').read_text()))}
    if not (folder / 'COMPLETE.json').is_file():
        return {'state': 'waiting'}
    record = json.loads((folder / 'RUN_RECORD.json').read_text())
    end = 'predictions-average.npz' if model in AVERAGED else f"predictions-step{record['updates']:05d}.npz"
    if not ((folder / 'predictions-parent.npz').is_file() and (folder / end).is_file()):   # a record without its grids
        return {'state': 'complete, but its prediction grids are not in this folder'}
    return {'state': 'complete', 'folder': folder, 'record': record, 'parent': folder / 'predictions-parent.npz',
            'end': folder / end}


def check_record(record, fold, files, order):
    """A run names this fold and its scoring files (by file name: a row may name a copy of the file elsewhere), and its
    parent and the run trained on the first n_lines of the fold's line order, so the calibration lines stay out.
    ``files``: the fold's files (``tahoe.files``); ``order``: its line order. State's rows of folds 2-3 name
    the line order only."""
    row = record['row']
    named = {key: name_of(row[key]) for key in ROW_FILES if key in row}
    expected = {key: (files[ROW_FILES[key]].parent if key == 'train_directory' else files[ROW_FILES[key]]).name for key in named}
    if row['fold'] != fold or named != expected:
        raise ValueError(f"run {row.get('name')}: fold {row['fold']} and files {named} differ from fold {fold}'s {expected}")
    if record.get('lines') != order[:int(row['n_lines'])]:
        raise ValueError(f"run {row.get('name')}: its lines are not the first {row['n_lines']} of fold {fold}'s line order")


def cell_runs(found, model, fold, cell, files, order):
    """The cell's six runs (three seeds; 28- and 21-line parents) and the joint state of each chain."""
    runs = {}
    for seed in SEEDS:
        for lines in ('n28', 'n21'):
            folders = found.get((model, fold, seed, lines, cell), [])
            done = [f for f in folders if (f / 'COMPLETE.json').is_file()]
            if len(done) > 1:
                raise ValueError(f'{model} {cell}: two complete runs of {seed} {lines}: {done}')
            runs[(seed, lines)] = read_run((done or folders)[0], model) if folders else {'state': 'not started'}
            if runs[(seed, lines)]['state'] == 'complete':
                check_record(runs[(seed, lines)]['record'], fold, files, order)
    states = {}
    for lines in ('n28', 'n21'):
        pending = {seed: runs[(seed, lines)]['state'] for seed in SEEDS if runs[(seed, lines)]['state'] != 'complete'}
        kind = 'diverged' if any(s.startswith('diverged') for s in pending.values()) else 'waiting'
        states[lines] = (f'{kind}: ' + ', '.join(f'{seed} {s}' for seed, s in pending.items())) if pending else 'complete'
    return runs, states


def genetic_runs(dataset, fold, model, cell, order, n_fit, seeds=SEEDS):
    """seed -> chain ('nested': the first ``n_fit`` lines of ``order``; 'full': all) -> the cell's run: its record
    folder and record, its end grid, its parent's grid (the start grid of the same parent's ``COMPARATOR`` run) and
    continued training's end grid (``continued``). Every run must be complete, unique, of this screen and trained on
    the first n_lines of the fold's order."""
    found, grids = {}, {}
    for records, root in GENETIC[dataset]:
        for key, folders in find_runs(Path(records) / fold / 'natural').items():
            found.setdefault(key, []).extend(folders)
            grids |= {f: Path(root) / f.relative_to(records) for f in folders}
    lines = {'nested': n_fit, 'full': len(order)}

    def one(which, seed, chain):
        folders = found.get((model, fold, seed, f'n{lines[chain]}', which), [])
        if len(folders) != 1 or not (folders[0] / 'COMPLETE.json').is_file():
            raise ValueError(f'{dataset} {fold} {model} {which} {seed} {chain}: not one complete run ({folders})')
        record = json.loads((folders[0] / 'RUN_RECORD.json').read_text())
        if record['row']['dataset'] != dataset or record['lines'] != order[:lines[chain]]:
            raise ValueError(f"run {record['row']['name']}: not {dataset}'s run on the first {lines[chain]} lines of {fold}'s order")
        return {'folder': folders[0], 'record': record, 'end': grids[folders[0]] / f"predictions-step{record['updates']:05d}.npz",
                'parent': grids[folders[0]] / 'predictions-parent.npz'}
    out = {}
    for seed in seeds:
        for chain in lines:
            comparator = one(COMPARATOR, seed, chain)
            out.setdefault(seed, {})[chain] = one(cell, seed, chain) | {'parent': comparator['parent'], 'continued': comparator['end']}
    return out


def genetic_grids(found):
    """The grid paths a case reads for a cell (``genetic_runs``): the parents, the cell's ends and, on the full
    chain, continued training's end."""
    return {s: {'nested': r['nested']['parent'], 'full': r['full']['parent'], 'cell_nested': r['nested']['end'],
                'cell_full': r['full']['end'], 'continued_full': r['full']['continued']} for s, r in found.items()}


def carried_cells(map_dir, model):
    """IL's restoring changes of one model over the three targets; a change named for several targets or by both
    halves is read once. Rows name their target."""
    groups = {}
    for suffix, target in TARGETS:
        path = Path(map_dir) / f'MAP_CARRIED{suffix}.csv'
        if not path.is_file():
            continue
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row['model'] == model and row['carried']:
                    groups.setdefault(row['carried'], []).append(row | {'target': target, 'suffix': suffix})
    return list(groups.items())


def synthetic_card(map_dir, model, cell, suffix=''):
    """IL's outcome of the cell per setting and seed for one target (the parent case of MAP_CELLS<suffix>.csv)."""
    cut, arm, rate = cell.split('/')
    found = {}
    with (Path(map_dir) / f'MAP_CELLS{suffix}.csv').open() as handle:
        for r in csv.DictReader(handle):
            if (r['model'], r['case'], r['cut'], r['arm']) == (model, 'parent', cut, arm) and float(r['rate']) == float(rate[1:]):
                found[(r['half'], int(r['seed']))] = r
    return found
