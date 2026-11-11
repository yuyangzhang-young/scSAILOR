"""Recovery and cost of every restoring change information localization (IL) names (``scsailor.resolution.il_cells``),
on Tahoe folds 1-3, from saved predictions only. A change is a cell of IL's map (``<cut>/<arm>/x<rate>``).

Three stages, run in this order, writing into ``--out``:

- ``calibration``: once per cell whose three 21-line runs are complete, from training units only, the selection on
  the nested chain's calibration lines (``stages.repair_entry`` per seed; the rule over the model's menu,
  ``policies.select``): ``SELECTIONS_<fold>_<model>_<cell>.json``, not rewritten;
- ``outer``: each cell whose three 28-line runs are complete, scored on the held-out lines against its parent,
  continued training, training longer and simple calibration, with that cell's selection (from ``--out``, else where
  it was saved, ``tahoe.selection_path``) where it exists: ``CARDS_<fold>.csv``, ``PER_SEED_<fold>.csv`` and
  ``PER_LINE_<fold>.csv``;
- ``continued`` (folds 1-3 at once): ``CONTINUED_X1.csv``, continued training beside each restoring change, for the
  four models, per seed.

Models by IL's names (summary, deepsets, state = the built attention model, official_state, prepr, cpa), where IL
reported their map tables and runs (``runs.MODELS``). The two comparators: continued training (``continuation``),
the changes' route, IL's real-task runner on the whole network (input/a/x1), as many updates as a change, nothing reset
(``CONTINUED_X1``); PrePR-CT's own recipe for 10,000 updates. Training longer (``training_longer``): the built
parents' own 160k schedule from 40k to 80k (fold 1: the saved grids of the parents' trajectory; folds 2-3: the kept 80k
weights predicted on origin resolution's grid), State's from 5k to 10k.

    python -m reproduce.origin_resolution.il_cells --stage calibration --fold F1 --out DIR [--model official_state]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scsailor.resolution import grids, il_cells, policies, stages
from scsailor.resolution.scoring import PARTKEY, identities, scores
from . import ATTENTION_F23, CONTINUED_X1, ROOT, SEEDS, at, runs, stored, tahoe, write_once

__all__ = ["ORDER", "continuation", "training_longer", "selection_path", "calibration_stage", "outer_stage", "continued_stage",
           "main"]

INHOUSE = ('summary', 'deepsets', 'state')   # IL's "state" is the built attention model
ORDER = INHOUSE + ('official_state', 'prepr', 'cpa')
NESTED = tahoe.NESTED
# Fold-1 built predictions of the 160k-schedule runs, in the natural task's target order: steps 40k-120k, in either
# of two folders
F1_TRAJECTORY = ROOT / 'tahoe-100m-deepsets-fold-1/fits'
F1_RETRIEVED = ROOT / 'tahoe-100m-predictions-fold-1'
F1_ENDPOINT = ROOT / 'tahoe-100m-evaluation-units-fold-1/scoring/fits'   # step 160000
CONTINUED_F23 = ROOT / 'origin-resolution/continued-training-folds-2-3'   # the kept 80k weights, on origin resolution's grid
RUN_TAG = {'summary': 'summary', 'deepsets': 'deepsets', 'state': 'state', 'official_state': 'officialstate'}   # in the run names
CONTINUED_END = {'summary': 'predictions-step40000.npz', 'deepsets': 'predictions-step40000.npz',
                 'state': 'predictions-step40000.npz', 'official_state': 'predictions-step05000.npz'}   # continued training's last grid
MODEL_NAME = {'summary': 'summary-statistics', 'deepsets': 'DeepSets', 'state': 'attention', 'official_state': 'official State'}
CONTINUED = {
    'route': ("the whole network on the cells' route (input/a/x1): information localization's real-task runner on the same data "
              "stream, fresh AdamW, "
              "1,000-update warm-up to the parent's rate, cosine to a tenth, as many updates as a cell, nothing reset")}
LONGER = {
    'inhouse': ("training longer (own schedule): the whole network on the parent's own 160k schedule from 40k to 80k, at "
                "its 40k rate (x1) falling to about 0.63 of it, with its AdamW state, 40,000 updates x 8 bags"),
    'official_state': ("training longer (own schedule): the whole network on the anchored fit's own 160k cosine from 5,000 to "
                       "10,000 updates, at its 5k rate (x1) falling to 0.994 of it, with its AdamW state, 5,000 updates x 8 "
                       "sets on the cells' data stream (its 10k weights, predicted on the cells' 16 input draws)"),
    'official_state_f23': ("training longer (own schedule): the 28-line fit's own 160k cosine from 5,000 to 10,000 updates with "
                           "its AdamW state, 5,000 updates x 8 sets (fold 2: its saved 10k weights; fold 3: continued from its "
                           "saved 5k state), predicted on origin resolution's grid")}
NOT_YET = {   # what is missing where no comparator or nested chain exists yet
    'prepr': 'not available: not deposited',
    'inhouse': "the 80k grids of folds 2 and 3 were deleted; predict the kept step80000.pt weights on origin resolution's grid",
    'official_state': ("the anchored fits' saved 10k predictions on origin resolution's grid: "
                       "not-deposited:orsn_continued.pbs, not yet run"),
    'cpa': ("the bag-anchor natural fits (cpa_bagmean_train28) stop at 160k, 20,000 updates past the parent; a plain "
            "140,000-update run of the whole network with the cell's read-out (final-quarter weight mean, decoder "
            "batch-norm statistics re-estimated, adversaries off) is needed")}
NOT_YET_NESTED = {'official_state': ("official State's 21-line parents on origin resolution's line order exist; information "
                                     "localization's 21-line runs of the cell are missing")}


def f1_paths(model, seed, level, step):
    """(prediction file, its SCORE.json) of one fold-1 checkpoint of the parents' trajectory, or None."""
    fit = f'{model}_{seed}_{level}'
    if step == 160000:
        folder = F1_ENDPOINT / fit
    else:
        folder = next((p for p in (F1_TRAJECTORY / f'{fit}_step{step}', F1_RETRIEVED / f'{fit}_step{step}')
                       if (p / 'PREDICTIONS.npz').exists()), None)
    if folder is None or not (folder / 'PREDICTIONS.npz').exists():
        return None
    return folder / 'PREDICTIONS.npz', folder / 'SCORE.json'


def continuation(model, fold, seed):
    """(start, end) prediction files of the 28-line parent's continued training and what it is, or (None, what is
    missing). The start must equal the cell's parent grid."""
    if model not in RUN_TAG:
        return None, NOT_YET[model]
    run = runs.read_run(CONTINUED_X1 / fold / 'natural' / f'natural-{fold}-{RUN_TAG[model]}-{seed}-n28-input-a-x1', model)
    if run['state'] != 'complete':
        return None, f"continued training of {model} {seed}: {run['state']}"
    return [run['parent'], run['end']], CONTINUED['route']


def training_longer(model, fold, seed):
    """(start, end) prediction files of the 28-line parent's own schedule and what it is, or (None, what is missing)."""
    if model in INHOUSE:
        if fold != 'F1':   # the kept 80k weights on origin resolution's grid, IL's parent grid as the start;
            # the attention model's: its 40k weights predicted beside them
            folder = (ATTENTION_F23 / 'continued-training' if model == 'state' else CONTINUED_F23) / fold / f'{model}_{seed}_L2'
            if not (folder / 'RECORD.json').exists():
                return None, NOT_YET['inhouse']
            start = (folder / 'predictions-step40000.npz' if model == 'state'
                     else at(json.loads((folder / 'RECORD.json').read_text())['start_grid']))
            return [start, folder / 'predictions-step80000.npz'], LONGER['inhouse']
        found = [f1_paths(model, seed, 'L2', step) for step in (40000, 80000)]
        if None in found:
            return None, f'the 40k or 80k grid of {model} {seed} is missing'
        return [f[0] for f in found], LONGER['inhouse']
    if model == 'official_state':   # the anchored fits' saved 10k predictions on OR's grid
        start = tahoe.state_grids(seed, 5000, fold)['full']
        end = tahoe.state_root(fold) / f'continued_train28_L2_{seed}' / 'predictions-or-grid-step10000.npz'
        what = LONGER['official_state' if fold == 'F1' else 'official_state_f23']
        return ([start, end], what) if end.is_file() else (None, NOT_YET['official_state'])
    return None, f'{model} has no training longer'


def selection_path(out, fold, model, cell):
    return Path(out) / f"SELECTIONS_{fold}_{model}_{cell.replace('/', '-')}.json"


def saved_selection(fold, model, cell):
    """Where the calibration stage's selection of the cell was saved."""
    return tahoe.selection_path({il: name for name, il in tahoe.IL_NAME.items()}.get(model, model), fold, cell, None)


def sources(fold, model):
    """(map folder, runs folder) of a model, or None when IL has not reported them."""
    found = runs.MODELS.get(fold, {}).get(model)
    return (found['map'], found['runs']) if found else None


def calibration_stage(fold, out, models=ORDER):
    """Selections of each cell on its nested chain's calibration lines, written once per cell."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files, order = tahoe.files(fold), tahoe.line_order(fold)
    train = tahoe.units(fold, 'train')
    calibration_lines = order[NESTED:]
    cal = np.isin(identities(train), calibration_lines)
    for model in models:
        where = sources(fold, model)
        if where is None:
            continue
        if not (Path(where[0]) / 'MAP_CARRIED.csv').is_file():
            continue
        found = runs.find_runs(where[1])
        for cell, _ in runs.carried_cells(where[0], model):
            path = selection_path(out, fold, model, cell)
            if path.exists():
                continue
            cell_runs, states = runs.cell_runs(found, model, fold, cell, files, order)
            if states['n21'] != 'complete':
                print(f"{fold} {model} {cell}: 21-line chain {states['n21']}", flush=True)
                continue
            entries = {}
            for seed in runs.SEEDS:
                r = cell_runs[(seed, 'n21')]
                p, a = grids.load(r['parent'], 'train', train), grids.load(r['parent'], 'train', train, 'anchor_raw')
                R = grids.load(r['end'], 'train', train)
                entries[str(seed)] = stages.repair_entry(p, R, a, train, cal)
            action = policies.select(entries, policies.default_menu(model == 'summary'))['ordinary_validation']
            write_once(path, {'fold': fold, 'model': model, 'cell': cell, 'outer_truth_read': False,
                              'calibration_lines': calibration_lines,
                              'runs': [stored(cell_runs[(s, 'n21')]['folder']) for s in runs.SEEDS],
                              'entries': entries, 'current_action': action})
            print(f"{fold} {model} {cell}: lambda {[entries[str(s)]['amount']['cheapest_fix'] for s in runs.SEEDS]}, "
                  f"current action {action}", flush=True)


def score_cell(fold, model, cell, cell_runs, saved, missing, held, train):
    """One cell with complete 28-line runs: its predictions read and paired with continued training, then
    ``il_cells.score``."""
    found = {seed: continuation(model, fold, seed) for seed in runs.SEEDS}
    longer = {seed: training_longer(model, fold, seed) for seed in runs.SEEDS}
    absent = [why for files, why in found.values() if files is None]
    paired = not absent
    longer_paired = all(files is not None for files, _ in longer.values())
    continued = found[runs.SEEDS[0]][1] if paired else absent[0]
    if longer_paired:
        continued += '; ' + longer[runs.SEEDS[0]][1]

    def seed_inputs(seed):
        r = cell_runs[(seed, 'n28')]
        p, a = grids.load(r['parent'], 'outer', held), grids.load(r['parent'], 'outer', held, 'anchor_raw')
        x = {'p': p, 'a': a, 'R': grids.load(r['end'], 'outer', held),
             'train': {'parent': lambda: grids.load(r['parent'], 'train', train), 'cell': lambda: grids.load(r['end'], 'train', train)}}

        def add(name, start, late):
            gaps = {'start': np.abs(grids.load(start, 'outer', held) - p).max(),   # the same parent and grid as the cell
                    'end anchor': np.abs(grids.load(late, 'outer', held, 'anchor_raw') - a).max(),   # the end's row order
                    'end training anchor': np.abs(grids.load(late, 'train', train, 'anchor_raw')
                                                  - grids.load(r['parent'], 'train', train, 'anchor_raw')).max()}
            if max(gaps.values()) > 1e-4:
                raise ValueError(f'{model} {seed}: {name} does not pair with the cell ({gaps})')
            x[name] = grids.load(late, 'outer', held)
            x['train'][name] = lambda: grids.load(late, 'train', train)

        if paired:
            add('continuation', *found[seed][0])
        if longer_paired:
            add('longer', *longer[seed][0])
        return x

    return il_cells.score(fold, model, cell, cell_runs[(runs.SEEDS[0], 'n28')]['record'], held, train, seed_inputs, runs.SEEDS,
                          saved, missing, continued, paired)


def outer_stage(fold, out, models=ORDER):
    out = Path(out)
    files, order = tahoe.files(fold), tahoe.line_order(fold)
    held, train = tahoe.units(fold, 'outer'), tahoe.units(fold, 'train')   # held-out truth is read only in this stage
    cards, seed_rows, line_rows = [], [], []
    for model in models:
        where = sources(fold, model)
        if where is None:
            cards.append({'fold': fold, 'model': model, 'status': "not available: no information localization map tables or "
                          "setting-3 runs for this case"})
            continue
        if not (Path(where[0]) / 'MAP_CARRIED.csv').is_file():
            cards.append({'fold': fold, 'model': model, 'status': f'waiting: no map tables yet in {where[0]}'})
            continue
        found = runs.find_runs(where[1])
        for cell, rows in runs.carried_cells(where[0], model):
            card = {'fold': fold, 'model': model, 'cell': cell,
                    'handed_on_by': '; '.join(f"{r['target']}: {r['half']} half, {r['kind']}" for r in rows)}
            recovered = {}
            for suffix, target in dict.fromkeys((r['suffix'], r['target']) for r in rows):
                found_map = runs.synthetic_card(where[0], model, cell, suffix)
                recovered[target] = (il_cells.recovery(found_map, runs.SEEDS), il_cells.q1_text(found_map, runs.SEEDS))
            card['Q1_target_recovered'] = ' | '.join(f'{target}: {rec}' + (f'; {detail}' if detail else '')
                                                     for target, (rec, detail) in recovered.items())
            cell_runs, states = runs.cell_runs(found, model, fold, cell, files, order)
            if states['n28'] != 'complete':
                cards.append(card | {'status': f"28-line runs {states['n28']}"})
                continue
            path = selection_path(out, fold, model, cell)
            path = path if path.exists() else saved_selection(fold, model, cell)
            saved = json.loads(path.read_text()) if path.exists() else None
            if saved and [at(r) for r in saved['runs']] != [cell_runs[(s, 'n21')].get('folder') for s in runs.SEEDS]:
                raise ValueError(f'{path} was selected on other 21-line runs than the current ones')
            missing = (None if saved else NOT_YET_NESTED[model] if model in NOT_YET_NESTED and states['n21'] != 'complete'
                       else f"21-line chain {states['n21']}" if states['n21'] != 'complete' else 'calibration stage not yet run')
            scored, improvement, seeds_, lines_ = score_cell(fold, model, cell, cell_runs, saved, missing, held, train)
            seed_rows += seeds_
            line_rows += lines_
            cards.append(card | scored | {'credit': ' | '.join(f'{target}: {il_cells.credit(improvement, rec, model)}'
                                                               for target, (rec, _) in recovered.items())})
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (('CARDS', cards), ('PER_SEED', seed_rows), ('PER_LINE', line_rows)):
        with (out / f'{name}_{fold}.csv').open('w', newline='') as handle:   # an empty table is written empty
            if table:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in table for k in r)))
                writer.writeheader()
                writer.writerows(table)
    for card in cards:
        for name, value in card.items():
            print(f"{card['fold']} {card['model']} {card.get('cell', '')} | {name}: {value}", flush=True)


def continued_stage(out):
    """Per fold, model, restoring change and seed, in points of the original model's held-out total MSE (total, common,
    contrast): the change minus the original model, continued training minus the original model, and the change minus
    continued training. The original model is each change's 28-line parent grid; continued training (``CONTINUED_X1``,
    as many updates as a change, nothing reset) must start from it within 1e-4."""
    def parts(pred, base, held):
        s, b = scores(pred, held), scores(base, held)
        return {part: 100 * (s[key] - b[key]) / b['total_mse'] for part, key in PARTKEY.items()}

    table = []
    for fold in ('F1', 'F2', 'F3'):
        held = tahoe.units(fold, 'outer')
        files, order = tahoe.files(fold), tahoe.line_order(fold)
        for model, end in CONTINUED_END.items():
            where = runs.MODELS[fold][model]
            found = runs.find_runs(where['runs'])
            cells = runs.carried_cells(where['map'], model)
            chains = {cell: runs.cell_runs(found, model, fold, cell, files, order)[0] for cell, _ in cells}
            for seed in SEEDS:
                run = CONTINUED_X1 / fold / 'natural' / f'natural-{fold}-{RUN_TAG[model]}-{seed}-n28-input-a-x1'
                start, late = grids.load(run / 'predictions-parent.npz', 'outer', held), grids.load(run / end, 'outer', held)
                for cell, rows in cells:
                    r = chains[cell][(seed, 'n28')]
                    p, R = grids.load(r['parent'], 'outer', held), grids.load(r['end'], 'outer', held)
                    gap = float(np.abs(start - p).max())
                    if gap > 1e-4:
                        raise ValueError(f'{fold} {model} {seed} {cell}: continued training starts {gap:.1e} away from the parent')
                    cost, cont = parts(R, p, held), parts(late, p, held)
                    table.append({'fold': fold, 'model': MODEL_NAME[model], 'cell': cell,
                                  'handed_on_by': '; '.join(f"{x['target']}: {x['half']} half, {x['kind']}" for x in rows), 'seed': seed,
                                  **{f'cost_{k}_pts': v for k, v in cost.items()}, **{f'continued_{k}_pts': v for k, v in cont.items()},
                                  **{f'change_minus_continued_{k}_pts': cost[k] - cont[k] for k in cost},
                                  'continued_start_gap': gap, 'continued_run': stored(run)})
            print(fold, model, 'done', flush=True)
    with open(Path(out) / 'CONTINUED_X1.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recovery and cost of the restoring changes.")
    parser.add_argument('--stage', choices=('calibration', 'outer', 'continued'), required=True)
    parser.add_argument('--fold', default='F1', choices=('F1', 'F2', 'F3'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', dest='models', nargs='+', choices=ORDER, default=ORDER, help="models by IL's names (default: all)")
    args = parser.parse_args(argv)
    if args.stage == 'continued':
        return continued_stage(args.out)
    (calibration_stage if args.stage == 'calibration' else outer_stage)(args.fold, args.out, tuple(args.models))


if __name__ == '__main__':
    main()
