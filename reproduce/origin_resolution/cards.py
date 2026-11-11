"""Origin resolution's stages on one dataset, model and fold: read the saved inputs, call ``stages``, write the
card's files once into ``--out``.

- ``selection`` (State on Tahoe, without a restoring change): its action chosen once before its calibration stage;
  ``SELECTION_tahoe_<fold>_official_state.json``.
- ``calibration``: builds no held-out units, so it cannot read held-out truth; reads the model's saved selection
  (the restoring change's, the repair selections', the selection stage's or the parent-only menu's) and writes
  ``CALIBRATION_<dataset>_<fold>_<model>.json``. Check at this entry point: the recomputed parent calibration
  total equals the saved one.
- ``attribution``: reads held-out truth and the calibration file; writes ``CARD_``, ``ROWS_``, ``WHERE_``,
  ``REFERENCES_`` and ``READINGS_<dataset>_<fold>_<model>.csv`` and ``ATTRIBUTION_RUN_<dataset>_<fold>_<model>.json``.

Options: ``--seeds``, ``--state-length`` (State at another length of its trajectory) and ``--cell`` (a restoring
change, ``<cut>/<arm>/x<rate>``, for the model's menu; the attention model's parents then come from that change's runs).
Tahoe has folds 1-3; GSE306429 fold 1; Replogle-Nadig folds 1-2 and Jiang et al. folds 4-5, where a change's stages read
the parents from the change's runs and its selection from ``genetic_cells``. A change's files carry no change in their
names: each writes into its own ``--out``.

    python -m reproduce.origin_resolution.cards --dataset tahoe --model summary --fold F1 --stage calibration --out DIR
"""
import argparse
import json
import resource
import time
from pathlib import Path

from scsailor.resolution import menu, stages
from . import SEEDS, gene_arrays, gse306429, jiang, read_csv, replogle, stored, tahoe, write_csv, write_once

__all__ = ["BINDINGS", "SEEDS", "read_selection", "groups_mapping", "menu_side", "calibration_file", "selection_stage",
           "calibration_stage", "attribution_stage", "main"]

BINDINGS = {'tahoe': tahoe, 'gse306429': gse306429, 'replogle': replogle, 'jiang': jiang}
KINDS = ('CARD', 'ROWS', 'WHERE', 'REFERENCES', 'READINGS')


def read_selection(path, model):
    """A saved calibration-stage selection: the current action, and per seed its amounts and calibration totals.
    The repair selections' file keys the attention model by information localization's name for it, "state"."""
    saved = json.loads(Path(path).read_text())
    if 'selected' not in saved:
        return saved
    name = tahoe.IL_NAME.get(model, model)
    return {'current_action': saved['cases'][name]['action'],
            'entries': {s: {k: v[k] for k in ('amount', 'calibration')} for s, v in saved['selected'][name].items()}}


def groups_mapping(dataset):
    """The dataset's declared perturbation grouping, p_code -> group, and its file; (None, None) where it declares none."""
    path = BINDINGS[dataset].GROUPS
    return ({int(r['p_code']): r['group'] for r in read_csv(path)} if path is not None else None), path


def menu_side(case, which, alpha, seeds):
    """The menu's inputs on one side: 'calibration' (the nested chain on the calibration units) or 'outer' (the full
    chain on the held-out units), with the dataset's own additions."""
    u = case['units']['calibration' if which == 'calibration' else 'outer']
    chain, side = ('nested', 'train') if which == 'calibration' else ('full', 'outer')
    p = {s: case['load'](s, chain, side, u) for s in seeds}
    a = {s: case['load'](s, chain, side, u, 'anchor_raw') for s in seeds}
    return menu.side(u, p, a, repair_alpha=alpha, **BINDINGS[case['dataset']].menu_inputs(case, which, seeds, a))


def calibration_file(out, dataset, model, fold):
    return Path(out) / f'CALIBRATION_{dataset}_{fold}_{model}.json'


def selection_stage(dataset, model, fold, out, seeds=SEEDS, **variant):
    """State's action, chosen once (stages.selection); no held-out unit is built."""
    if (dataset, model) != ('tahoe', 'official_state') or variant.get('cell'):
        raise ValueError('the selection stage is for official State on Tahoe without a cell; the other models have saved selections')
    case = tahoe.case(model, fold, outer=False, seeds=seeds, **variant)
    entries, action = stages.selection(case, seeds)
    selection = {'dataset': dataset, 'fold': fold, 'model': model, 'outer_truth_read': False, 'outer_units_built': False,
                 'rule': stages.SELECTION_RULE, 'current_action': action, 'entries': entries}
    Path(out).mkdir(parents=True, exist_ok=True)
    write_once(tahoe.selection_path(model, fold, None, out), selection)
    print(f"{dataset} {fold} {model}: action {selection['current_action']}", flush=True)


def calibration_stage(dataset, model, fold, out, seeds=SEEDS, **variant):
    binding = BINDINGS[dataset]
    case = binding.case(model, fold, outer=False, seeds=seeds, **variant)
    path = binding.selection_path(model, fold, variant.get('cell'), out)
    saved = read_selection(path, model)
    side = menu_side(case, 'calibration', stages.repair_alpha(saved), seeds)
    mapping, groups = groups_mapping(dataset)
    record = stages.calibration(case, saved, seeds, side, mapping)
    if record['perturbation_group'] is not None:
        record['perturbation_group'] = {'file': stored(groups), **record['perturbation_group']}
    written = {'dataset': dataset, 'fold': fold, 'model': model}
    for key, value in record.items():
        if key == 'calibration_units':
            written |= {'grids': {str(s): stored(case['grids'][s]['nested']) for s in seeds}, 'saved_selection': stored(path)}
        written[key] = value
    Path(out).mkdir(parents=True, exist_ok=True)
    write_once(calibration_file(out, dataset, model, fold), written)
    print(f"{dataset} {fold} {model}: " + stages.calibration_message(record, seeds), flush=True)


def attribution_stage(dataset, model, fold, out, seeds=SEEDS, **variant):
    started = time.time()
    out = Path(out)
    names = [f'{k}_{dataset}_{fold}_{model}.csv' for k in KINDS]
    if any((out / n).exists() for n in names):
        raise FileExistsError('attribution outputs exist; the stage is run once')
    saved = json.loads(calibration_file(out, dataset, model, fold).read_text())
    if saved['outer_truth_read'] or saved['outer_units_built']:
        raise ValueError('the calibration file does not record a calibration stage without outer units')
    binding = BINDINGS[dataset]
    case = binding.case(model, fold, seeds=seeds, **variant)
    side = menu_side(case, 'outer', saved['menu_inputs']['repair_alpha'], seeds)
    mapping, _ = groups_mapping(dataset)
    tables, record = stages.attribution_stage(case, saved, seeds, side, binding.evidence,
                                                  stages.gene_groups(gene_arrays(case['train_files'])), mapping)
    out.mkdir(parents=True, exist_ok=True)
    for name, kind in zip(names, KINDS):
        write_csv(out / name, tables[kind])
    write_once(out / f'ATTRIBUTION_RUN_{dataset}_{fold}_{model}.json',
               {'dataset': dataset, 'model': model,
                'fold': fold, 'calibration': stored(calibration_file(out, dataset, model, fold)),
                'grids': {str(s): stored(case['grids'][s]['full']) for s in seeds}, **record,
                'wall_seconds': time.time() - started, 'peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    for h in tables['CARD']:
        print(h['line'], h['text'], flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="One stage of origin resolution on one model and fold.")
    parser.add_argument('--dataset', required=True, choices=sorted(BINDINGS))
    parser.add_argument('--model', required=True, choices=('summary', 'deepsets', 'attention', 'official_state'))
    parser.add_argument('--stage', required=True, choices=('selection', 'calibration', 'attribution'))
    parser.add_argument('--fold', default='F1', choices=('F1', 'F2', 'F3', 'F4', 'F5'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', default=','.join(map(str, SEEDS)), help='comma-separated seeds')
    parser.add_argument('--state-length', type=int, help="Tahoe, State: a length of its trajectory (default 5000)")
    parser.add_argument('--cell', help="Tahoe, Replogle-Nadig, Jiang et al.: a restoring change for the model's menu, "
                                       "<cut>/<arm>/x<rate> (Tahoe's default: the model's own)")
    args = parser.parse_args(argv)
    variant = {k: v for k, v in (('cell', args.cell), ('length', args.state_length)) if v is not None}
    if args.state_length is not None and args.dataset != 'tahoe':
        parser.error('--state-length applies to Tahoe only')
    if args.cell is not None and args.dataset == 'gse306429':
        parser.error('--cell does not apply to GSE306429: information localization names no restoring change there')
    seeds = tuple(int(s) for s in args.seeds.split(','))
    {'selection': selection_stage, 'calibration': calibration_stage, 'attribution': attribution_stage}[args.stage](
        args.dataset, args.model, args.fold, args.out, seeds, **variant)


if __name__ == '__main__':
    main()
