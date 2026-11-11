"""The cost of each change beside continued training, and the original model's alignment, on the screens read on the
full chain only, from saved grids. With no nested chain there is no calibration, so no recommendation or refusal.

Reads the runs of ``--folds`` under ``--root`` (``<root>/<dataset>/<fold>/natural/``) and writes ``COST.csv`` (one row per
fold, change and seed) and ``ALIGNMENT.csv`` (one row per fold, seed, side and part). Points are of the original model's
total MSE on the same units. A run of continued training must start from the same original model as the change.

    python -m reproduce.origin_resolution.cost_readings --root ROOT --model official_state --folds parse/G1 replogle/F2 --out DIR
"""
import argparse
from pathlib import Path

import numpy as np

from scsailor.resolution import card, grids
from scsailor.resolution.scoring import PARTKEY, retained, scores
from . import ROOT, heldout_units, runs, training_units, write_csv
from .runs import COMPARATOR
from .slopes import BINDINGS

__all__ = ["units", "on_grid", "read", "main"]


def units(dataset, fold):
    """The fold's training and held-out units, without the donors a screen leaves out of its evaluation."""
    f = BINDINGS[dataset].files(fold)
    return training_units(f['train'], f['constants']), heldout_units(f['outer'], f.get('excluded', ()))


def on_grid(u, path):
    """The units the grid's training side holds, with the datasets' weights recomputed."""
    with np.load(path) as saved:
        keys = set(zip(saved['side_context'].astype(str), saved['p_code'].astype(int)))
    return retained(u, np.asarray([('train/' + c, int(p)) in keys for c, p in zip(u['context'], u['p_code'])]))


def read(root, folds, model, out):
    """Reads the runs of ``folds`` under ``root`` and writes the two tables into ``out``."""
    cost, alignment = [], []
    for dataset, fold in folds:
        folder = Path(root) / dataset / fold / 'natural'
        found = {k: v for k, v in (runs.find_runs(folder) if folder.is_dir() else {}).items() if k[0] == model}
        cells = sorted({k[4] for k in found} - {COMPARATOR})
        if not cells:
            continue
        train, held = units(dataset, fold)
        seen = None
        for cell in cells:
            for seed in sorted({k[2] for k in found}):   # GSE306429's attention fits use their own seeds
                run = {c: runs.read_run(next(v[0] for k, v in found.items() if (k[2], k[4]) == (seed, c)), model)
                       for c in (cell, COMPARATOR)}
                if any(r['state'] != 'complete' for r in run.values()):
                    raise ValueError(f'{dataset} {fold} {cell} {seed}: {[r["state"] for r in run.values()]}')
                load = lambda r, which, member='raw': grids.load(r[which], 'outer', held, member)
                p, q = load(run[cell], 'parent'), load(run[COMPARATOR], 'parent')
                a = load(run[cell], 'parent', 'anchor_raw')
                gaps = (np.abs(p - q).max(), np.abs(load(run[cell], 'end', 'anchor_raw') - a).max(), np.abs(load(run[COMPARATOR], 'end', 'anchor_raw') - a).max())
                if max(gaps) > 1e-4:
                    raise ValueError(f'{dataset} {fold} {cell} {seed}: parents or anchors differ ({gaps})')
                base = scores(p, held)
                total = base['total_mse']
                pts = lambda s, name: 100 * (s[PARTKEY[name]] - base[PARTKEY[name]]) / total
                R, C = scores(load(run[cell], 'end'), held), scores(load(run[COMPARATOR], 'end'), held)
                row = {'dataset': dataset, 'fold': fold, 'cell': cell, 'seed': seed, 'units': len(held['truth']), 'parent_total': total,
                       'anchor_total': scores(a, held)['total_mse'], 'parent_gap': float(gaps[0])}
                for name in PARTKEY:
                    row |= {f'cost_{name}_pts': pts(R, name), f'continued_{name}_pts': pts(C, name), f'change_minus_continued_{name}_pts': pts(R, name) - pts(C, name)}
                cost.append(row)
                if cell == cells[0]:
                    seen = seen or on_grid(train, run[cell]['parent'])
                    on_training = lambda member: grids.load(run[cell]['parent'], 'train', seen, member)
                    sides = (('held-out', held, p, a), ('training', seen, on_training('raw'), on_training('anchor_raw')))
                    for side, u, pred, anchor in sides:
                        for part, v in card.alignment(pred, anchor, u).items():
                            alignment.append({'dataset': dataset, 'fold': fold, 'seed': seed, 'side': side, 'part': part, 'units': len(u['truth']),
                                              'cosine': v['cosine'], 'best_scale': v['best_scale'], 'size_over_anchor_error': v['size_over_anchor_error']})
        print(f'{dataset} {fold}: {cells}', flush=True)
    write_csv(Path(out) / 'COST.csv', cost, mode='w')   # rewritten as folds arrive
    write_csv(Path(out) / 'ALIGNMENT.csv', alignment, mode='w')


def main(argv=None):
    parser = argparse.ArgumentParser(description="The cost of each change beside continued training, and the original model's alignment.")
    parser.add_argument('--root', type=Path, required=True, help='the runs folder, under the data root')
    parser.add_argument('--model', required=True, help="the runs' model name: summary, deepsets, state (the attention model), official_state")
    parser.add_argument('--folds', nargs='+', required=True, help='dataset/fold, read in this order')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    read(args.root if args.root.is_absolute() else ROOT / args.root, [tuple(f.split('/')) for f in args.folds], args.model, args.out)


if __name__ == '__main__':
    main()
