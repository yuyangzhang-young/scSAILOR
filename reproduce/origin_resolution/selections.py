"""The selections origin resolution's stages read as their saved selection, made before them on calibration units only.

- ``repair`` (Tahoe, one fold): the repair selections' calibration stage. Per model and seed, the nested chain of the
  localization repair (a parent and its repair fitted on the fold's first 21 lines; summary-statistics: origin resolution's S arm, DeepSets and the attention
  model: the localization exact recipe), every label, amount and policy total on the 7 calibration lines
  (``stages.repair_entry``), and the decision table as a rule (``policies.diagnose``). Writes
  ``SELECTIONS_<fold>.json``; the attention model's stages read fold 1's.
- ``nested`` (fold 1 of GSE306429 and Replogle-Nadig; fold 2 of Replogle-Nadig; folds 4 and 5 of Jiang et al.): the
  parent-only menu on the nested parents' calibration units: the parent and the cell-line-level shrink, chosen by
  ordinary validation (``policies.select``). Every nested run must be complete, have trained on the fit lines and list
  the same unsupported conditions; their calibration units leave the evaluation.
  Writes ``SELECTIONS_<dataset>_<model>.json`` on fold 1 and ``SELECTIONS_<dataset>_<fold>_<model>.json`` on the
  later folds, once per dataset and model.

    python -m reproduce.origin_resolution.selections --kind repair --fold F1 --out DIR
    python -m reproduce.origin_resolution.selections --kind nested --fold F1 --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scsailor.resolution import grids, policies, stages
from scsailor.resolution.scoring import identities, retained
from . import ROOT, SEEDS, gse306429, jiang, listed_unsupported, replogle, stored, tahoe, training_units, write_once

__all__ = ["SEEDS", "SYNTHETIC", "chain_files", "repair", "nested", "main"]

MODELS = ('summary', 'deepsets', 'state')   # the repair selections' models, by IL's names ("state": the attention model)
END = {'summary': 8000, 'deepsets': 8000, 'state': 10000}   # the repair's last update
# Synthetic readings on Tahoe: "not recovered" would stop the route
SYNTHETIC = {'summary': 'recovered', 'deepsets': 'recovered', 'state': 'partial'}
STORE = ROOT / 'origin-resolution'   # fold 1; folds 2 and 3 under folds/<fold>
NESTED_MODELS = ('summary', 'deepsets')
NESTED_MENU = ['parent', 'cheapest_fix']   # the parent-only menu: no restoring change on these datasets
NESTED_DATASETS = {'gse306429': gse306429, 'replogle': replogle, 'jiang': jiang}   # each lists its folds (FOLDS)


def chain_files(model, seed, fold):
    """The nested chain's (parent, repair) prediction files of the repair selections."""
    store = STORE if fold == 'F1' else STORE / 'folds' / fold
    if model == 'summary':
        run = store / 'nested-alpha' / f'repair-summary-S-{seed}-n21-nestedparent'
        return run / 'predictions-step00000.npz', run / 'predictions-step08000.npz'
    run = store / 'il-exact' / f'{model}-{seed}-n21-nestedparent'
    return run / 'predictions-parent.npz', run / f'predictions-step{END[model]:05d}.npz'


def repair(fold, out):
    path = Path(out) / f'SELECTIONS_{fold}.json'
    if path.exists():
        raise FileExistsError(f'{path} exists; the calibration stage is run once')
    Path(out).mkdir(exist_ok=True)
    train = tahoe.units(fold, 'train')
    order = tahoe.line_order(fold)
    calibration = np.isin(identities(train), order[tahoe.NESTED:])
    selected, cases = {}, {}
    for model in MODELS:
        seeds = {}
        for seed in SEEDS:
            nested_parent, nested_repair = chain_files(model, seed, fold)
            p, a = grids.load(nested_parent, 'train', train), grids.load(nested_parent, 'train', train, 'anchor_raw')
            R = grids.load(nested_repair, 'train', train)
            seeds[seed] = stages.repair_entry(p, R, a, train, calibration)
        cases[model] = policies.diagnose(seeds, SYNTHETIC[model], model == 'summary')
        selected[model] = {str(seed): entry for seed, entry in seeds.items()}
        print(f"{fold} {model:8s} calibration: {cases[model]}", flush=True)
    write_once(path, {'fold': fold, 'calibration_lines': order[tahoe.NESTED:], 'grid': list(policies.G),
                      'outer_truth_read': False, 'selected': selected, 'cases': cases})


def unsupported(binding, fold, fit):
    """Conditions without fit-side training target. Every run must be complete, have trained on the fit units, and
    list the same conditions."""
    folders = [binding.records(m, s, fold) for m in NESTED_MODELS for s in SEEDS]
    for folder in folders:
        if not (folder / 'COMPLETE.json').exists():
            raise FileNotFoundError(f'{folder}: run not complete')
        lines = json.loads((folder / 'nested-parent/RUN_IDENTITY.json').read_text())['lines']
        if lines != fit:
            raise ValueError(f'{folder} trained on {lines}, not on the fit units {fit}')
    return np.asarray(listed_unsupported(folders), dtype=np.int64)


def nested(out, fold='F1'):
    """The parent-only selections of the datasets with nested parents on ``fold``, once per dataset and model."""
    bindings = {d: binding for d, binding in NESTED_DATASETS.items() if fold in binding.FOLDS}
    if not bindings:
        raise ValueError(f'no dataset has nested parents on {fold}')
    paths = {(d, m): binding.grid_paths(m, SEEDS, fold) for d, binding in bindings.items() for m in NESTED_MODELS}
    missing = [str(f) for found in paths.values() for s in SEEDS for f in (found[s]['full'], found[s]['nested']) if not Path(f).exists()]
    if missing:   # stop before anything is chosen, so that a later stage cannot fail on a missing grid
        raise FileNotFoundError('missing grids: ' + ', '.join(missing))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for dataset, binding in bindings.items():
        f = binding.files(fold)
        order = json.loads(f['order'].read_text())['line_order']
        fit, calibration = order[:f['n_fit']], order[f['n_fit']:]
        drop = unsupported(binding, fold, fit)
        train = training_units(f['train'], f['constants'])
        in_calibration = np.isin(identities(train), calibration)
        removed = in_calibration & np.isin(train['p_code'], drop)
        cal = retained(train, in_calibration & ~removed)
        mask = np.ones(len(cal['truth']), dtype=bool)
        for model in NESTED_MODELS:
            path = out / binding.selection_path(model, fold, None, out).name
            if path.exists():
                continue
            entries = {}
            for seed in SEEDS:
                grid = paths[(dataset, model)][seed]['nested']
                entries[str(seed)] = stages.parent_entry(grids.load(grid, 'train', cal), grids.load(grid, 'train', cal, 'anchor_raw'), cal, mask)
            choice, action = stages.parent_selection(entries, list(NESTED_MENU))
            write_once(path, {'dataset': dataset, 'fold': fold, 'model': model, 'outer_truth_read': False,
                               'fit': fit, 'calibration': calibration, 'calibration_units': int(mask.sum()),
                               'removed_calibration_units': int(removed.sum()), 'unsupported_conditions': drop.tolist(),
                               'weights': 'recomputed on the retained units by the datasets\' rule',
                               'runs': [stored(paths[(dataset, model)][s]['nested']) for s in SEEDS], 'menu': choice['menu'],
                               'entries': entries, 'mean_normalized_calibration': choice['mean_normalized_calibration'],
                               'current_action': action})
            print(f"{dataset} {model}: lambda {[entries[str(s)]['amount']['cheapest_fix'] for s in SEEDS]}, action {action}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="The selections origin resolution's stages read.")
    parser.add_argument('--kind', choices=('repair', 'nested'), required=True)
    parser.add_argument('--fold', default='F1', choices=('F1', 'F2', 'F3', 'F4', 'F5'),
                        help='repair: Tahoe folds 1-3; nested: the folds of the datasets with nested parents')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    repair(args.fold, args.out) if args.kind == 'repair' else nested(args.out, args.fold)


if __name__ == '__main__':
    main()
