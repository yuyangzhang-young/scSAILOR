"""Origin resolution's readings of the restoring changes information localization (IL) names on the genetic screens
(Replogle-Nadig folds 1-2, Jiang et al. folds 4-5; summary-statistics and DeepSets), from saved predictions only
(``scsailor.resolution.il_cells``; the Tahoe stage is ``il_cells``).

The changes and their recovery: on Replogle-Nadig, IL's map tables of the fold (``MAPS``: ``runs.carried_cells``, and
its synthetic readings per target, as on Tahoe); on Jiang et al., the cases of IL's light version (``LIGHT``), whose
reading is quoted as it stands, without new runs (seed 31002, with 31003 where IL ran a check). Each change has six
runs (three seeds; the full parent and origin resolution's nested parent), and so has continued training
(``runs.COMPARATOR``); ``runs.genetic_runs`` finds them. The case is the cell's own (``<binding>.case(..., cell=)``),
so the calibration units leave out the conditions the nested parent has no training target for.

Two stages, writing into ``--out`` (``GENETIC_READINGS``/il-cells/ is where the changes' stages read the selections):

- ``calibration``: per cell, from training units only, the selection on the nested chain's calibration units
  (``stages.repair_entry`` per seed; the rule over the model's menu, ``policies.select``):
  ``SELECTIONS_<dataset>_<fold>_<model>_<cell>.json``, written once;
- ``outer``: every cell on the held-out lines against its parent, continued training and simple calibration
  (``il_cells.score``), and against an extra seed at full strength: the parent averaged with the next seed's parent
  (31002 with 31003, 31003 with 31004, 31004 with 31002), as the equal-budget comparison on Tahoe did:
  ``CARDS_<dataset>_<fold>.csv``, ``PER_SEED_<dataset>_<fold>.csv``, ``PER_LINE_<dataset>_<fold>.csv``.

    python -m reproduce.origin_resolution.genetic_cells --stage calibration --dataset replogle --fold F1 --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scsailor.resolution import il_cells, net_cost, policies, stages
from scsailor.resolution.comparison import compare
from scsailor.resolution.scoring import PARTKEY, line_parts, scores
from . import (ROOT, SEEDS, heldout_units, jiang, read_csv, replogle, runs, selection_file, stored, training_units,
          write_csv, write_once)

__all__ = ["BINDINGS", "MODELS", "MAPS", "LIGHT", "CONTINUED", "from_maps", "from_light", "handed_on", "calibration_stage",
           "score_cell", "outer_stage", "main"]

BINDINGS = {'replogle': replogle, 'jiang': jiang}
MODELS = ('summary', 'deepsets')
MAPS = ROOT / 'information-localization-replogle-nadig/replogle'   # <fold>/MAP_CARRIED<suffix>.csv, MAP_CELLS<suffix>.csv
LIGHT = ROOT / 'light-version-cases/applications'   # Jiang et al.'s light version
CONTINUED = ("the whole network, no reset, on the cells' recipe (fresh AdamW, 1,000-update warm-up, cosine to a tenth) at x1 the "
             "parent's 40k rate, 40,000 updates x 8 bags, read out at the last update (information localization's input/a/x1 run on "
             "the same parent)")


def from_maps(fold, model):
    """Replogle-Nadig: the restoring changes of IL's map tables, each with its synthetic readings per target (as on
    Tahoe)."""
    out = {}
    for cell, rows in runs.carried_cells(MAPS / fold, model):
        recovered = {}
        for suffix, target in dict.fromkeys((r['suffix'], r['target']) for r in rows):
            found = runs.synthetic_card(MAPS / fold, model, cell, suffix)
            recovered[target] = (il_cells.recovery(found, SEEDS), il_cells.q1_text(found, SEEDS))
        out[cell] = ({'handed_on_by': '; '.join(f"{r['target']}: {r['half']} half, {r['kind']}" for r in rows),
                      'Q1_target_recovered': ' | '.join(f'{target}: {rec}' + (f'; {detail}' if detail else '')
                                                        for target, (rec, detail) in recovered.items())},
                     {target: rec for target, (rec, _) in recovered.items()})
    return out


def from_light(fold, model):
    """Jiang et al.: the restoring changes of IL's light version, with its reading quoted as it stands."""
    mine = lambda r: (r['dataset'], r['model'], r['fold']) == ('jiang', model, fold)
    cells = [c for r in read_csv(LIGHT / 'JIANG_PARSE_RESTORING_CHANGES.csv') if mine(r) for c in r['cells'].split(';') if c]
    cases = [r for r in read_csv(LIGHT / 'JIANG_PARSE_LIGHT_CASES.csv') if mine(r)]
    out = {}
    for cell in cells:
        rows = [r for r in cases if r['cell'] == cell]
        out[cell] = ({'handed_on_by': '; '.join(f"{r['target_name']}: {r['half']} half, {r['kind']}" for r in rows),
                      'Q1_target_recovered': ' | '.join(f"{r['target_name']} ({r['half']} half): {r['behaviour']}"
                                                        + (f"; checked {r['checked']}" if r['checked'] else '') for r in rows)
                                             + f" (information localization's light reading, seed {rows[0]['primary_seed']}, "
                                               "quoted as it stands)"},
                     dict.fromkeys((r['target_name'] for r in rows), "the light reading (quoted)"))
    return out


def handed_on(dataset, fold, model):
    """change -> (its ``handed_on_by`` and recovery columns, target -> recovery) for one model's restoring changes."""
    return {'replogle': from_maps, 'jiang': from_light}[dataset](fold, model)


def calibration_stage(dataset, fold, out):
    """Each cell's selection on its nested chain's calibration units, written once per cell."""
    binding, out = BINDINGS[dataset], Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for cell in handed_on(dataset, fold, model):
            path = out / selection_file(dataset, fold, model, cell).name
            if path.exists():
                continue
            case = binding.case(model, fold, outer=False, seeds=SEEDS, cell=cell)
            cal = case['units']['calibration']
            entries = {}
            for seed in SEEDS:
                p, a = case['load'](seed, 'nested', 'train', cal), case['load'](seed, 'nested', 'train', cal, 'anchor_raw')
                R = case['load'](seed, 'cell_nested', 'train', cal)
                entries[str(seed)] = stages.repair_entry(p, R, a, cal, np.ones(len(cal['truth']), dtype=bool))
            action = policies.select(entries, policies.default_menu(model == 'summary'))['ordinary_validation']
            write_once(path, {'dataset': dataset, 'fold': fold, 'model': model, 'cell': cell, 'outer_truth_read': False,
                              'calibration_lines': case['calibration_lines'], 'calibration_units': len(cal['truth']),
                              'runs': {str(s): stored(case['grids'][s]['cell_nested']) for s in SEEDS},
                              'parents': {str(s): stored(case['grids'][s]['nested']) for s in SEEDS},
                              'entries': entries, 'current_action': action})
            print(f"{dataset} {fold} {model} {cell}: lambda {[entries[str(s)]['amount']['cheapest_fix'] for s in SEEDS]}, "
                  f"size {[entries[str(s)]['amount']['size_fix'] for s in SEEDS]}, current action {action}", flush=True)


def score_cell(dataset, fold, model, cell, saved, held, train):
    """One cell's readings: ``il_cells.score`` with continued training paired on the same parent, then the extra seed."""
    binding = BINDINGS[dataset]
    case = binding.case(model, fold, seeds=SEEDS, cell=cell)
    record = runs.genetic_runs(dataset, fold, model, cell, binding.line_order(fold), binding.N_FIT)[SEEDS[0]]['full']['record']
    load = case['load']
    parents = {seed: load(seed, 'full', 'outer', held) for seed in SEEDS}

    def seed_inputs(seed):
        a = load(seed, 'full', 'outer', held, 'anchor_raw')
        gaps = {chain: np.abs(load(seed, chain, 'outer', held, 'anchor_raw') - a).max() for chain in ('cell_full', 'continued_full')}
        if max(gaps.values()) > 1e-4:   # the parent is read from another run of the same parent: the anchors must agree
            raise ValueError(f'{dataset} {fold} {model} {cell} {seed}: anchors differ from the parent ({gaps})')
        return {'p': parents[seed], 'a': a, 'R': load(seed, 'cell_full', 'outer', held), 'continuation': load(seed, 'continued_full', 'outer', held),
                'train': {name: (lambda chain=chain: load(seed, chain, 'train', train))
                          for name, chain in (('parent', 'full'), ('cell', 'cell_full'), ('continuation', 'continued_full'))}}

    missing = None if saved else 'calibration stage not yet run'
    card, improvement, seed_rows, line_rows = il_cells.score(fold, model, cell, record, held, train, seed_inputs, SEEDS,
                                                             saved, missing, CONTINUED, True)
    base, extra, below = [], [], []
    for seed, row in zip(SEEDS, seed_rows):
        mix = net_cost.extra_seed({'p': parents}, seed)
        b, e = scores(parents[seed], held), scores(mix, held)
        row |= {f'extra_seed_{part}_pts': 100 * (e[key] - b[key]) / b['total_mse'] for part, key in PARTKEY.items()}
        base.append((line_parts(parents[seed], held), b['total_mse']))
        extra.append((line_parts(mix, held), b['total_mse']))
        below.append(e['total_mse'] < b['total_mse'])
    card['extra seed - parent'] = il_cells.text(compare(extra, base), below)
    return card, improvement, seed_rows, line_rows


def outer_stage(dataset, fold, out):
    binding, out = BINDINGS[dataset], Path(out)
    f = binding.files(fold)
    train, held = training_units(f['train'], f['constants']), heldout_units(f['outer'])   # held-out truth: this stage only
    cards, seed_rows, line_rows = [], [], []
    for model in MODELS:
        for cell, (columns, recovered) in handed_on(dataset, fold, model).items():
            path = out / selection_file(dataset, fold, model, cell).name
            saved = json.loads(path.read_text()) if path.exists() else None
            scored, improvement, seeds_, lines_ = score_cell(dataset, fold, model, cell, saved, held, train)
            seed_rows += seeds_
            line_rows += lines_
            cards.append({'dataset': dataset, 'fold': fold, 'model': model, 'cell': cell, **columns, **scored,
                          'credit': ' | '.join(f'{target}: {il_cells.credit(improvement, rec, model)}' for target, rec in recovered.items())})
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (('CARDS', cards), ('PER_SEED', seed_rows), ('PER_LINE', line_rows)):
        write_csv(out / f'{name}_{dataset}_{fold}.csv', table)
    for card in cards:
        for name, value in card.items():
            print(f"{dataset} {fold} {card['model']} {card['cell']} | {name}: {value}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Origin resolution's readings of the restoring changes on the genetic screens.")
    parser.add_argument('--stage', choices=('calibration', 'outer'), required=True)
    parser.add_argument('--dataset', choices=sorted(BINDINGS), required=True)
    parser.add_argument('--fold', choices=('F1', 'F2', 'F4', 'F5'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    (calibration_stage if args.stage == 'calibration' else outer_stage)(args.dataset, args.fold, args.out)


if __name__ == '__main__':
    main()
