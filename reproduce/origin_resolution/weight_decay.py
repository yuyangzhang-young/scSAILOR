"""Weight decay as a comparator on Tahoe-100M: each model's fits trained again with weight decay, the strength chosen on
calibration lines, read against the original model as trained and calibrated.

Two stages, writing into ``--out``:

- ``select``: ``SELECTION.json``, per model and fold the strength whose nested parent (21 lines, seed 31002) has the
  lowest clipped total on the fold's calibration units; 0 is the original nested parent, and ties go to the weaker
  decay. Only training units are read;
- ``reading``: ``WEIGHT_DECAY_READING.csv``, per model, fold and seed, the 28-line fits at the selected strength against
  the original full parents of the same seeds, on held-out lines: as trained, and against the original calibrated
  toward the plain and the design-aware baseline (two-part amounts from the nested chain's calibration lines); the
  common and contrast parts; the held-out alignment of each model's correction from the plain baseline
  (``card.alignment``). Points of the original model's total MSE, positive where weight decay has the lower error;
  seed means with paired line-resampled intervals. Reads ``SELECTION.json`` from ``--out``.

    OMP_NUM_THREADS=1 python -m reproduce.origin_resolution.weight_decay --stage reading --out DIR
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from scsailor.resolution import card, grids, menu, net_cost
from scsailor.resolution.scoring import line_parts, scores
from scsailor.resolution.comparison import bootstrap, pool_seeds
from . import ROOT, SEEDS, tahoe
from .net_cost import read_side
from .slopes import load_case

__all__ = ["built_folder", "state_folder", "select_stage", "reading_stage", "main"]

BUILT = ROOT / 'weight-decay-and-uniform-shrink/built'   # <fold>/<model>-<seed>-n<lines>-wd<strength>/predictions-parent.npz
STATE = ROOT / 'weight-decay-and-uniform-shrink/state'   # <fold>/st_anchored_or21_L2_<seed>_wd<c>, train28_L2_<seed>_wd<c>
FOLDS = ('F1', 'F2', 'F3')
BUILT_MODELS = ('summary', 'deepsets')
SCHEDULE_SUM = 3.7819   # sum of the learning rates over 40,000 updates of the 160k schedule at peak 1e-4
# built models' strengths: cumulative decay 0.1, 0.5, 2 and 8 over the schedule; fold 2 adds 32, since 8 was chosen
BUILT_GRID = tuple(round(e / SCHEDULE_SUM, 4) for e in (0.1, 0.5, 2.0, 8.0))
BUILT_EXTRA = {('F2', 'summary'): (round(32 / SCHEDULE_SUM, 4),), ('F2', 'deepsets'): (round(32 / SCHEDULE_SUM, 4),)}
STATE_COEFFICIENTS = ('0.74', '3.70', '14.8')   # State: cumulative decay 0.1, 0.5 and 2 over its 5,000 updates
REFERENCES = ('original', 'original_field', 'original_design')


def built_folder(fold, model, seed, lines, w):
    return BUILT / fold / f"{model}-{seed}-n{lines}-wd{f'{w:.4f}'.replace('.', 'p')}"


def state_folder(fold, seed, lines, text):
    stem = 'st_anchored_or21' if lines == 21 else 'train28'
    return STATE / fold / f"{stem}_L2_{seed}_wd{text.replace('.', 'p')}"


def calibration_total(model, fold, path):
    u = tahoe.case(model, fold, outer=False)['units']['calibration']
    return scores(grids.load(path, 'train', u), u)['total_mse']


def select_stage(out):
    chosen = {}
    choose = lambda totals: min(sorted(totals), key=lambda w: totals[w])   # the lowest total; ties to the weaker decay
    for model in BUILT_MODELS:
        for fold in FOLDS:
            totals = {0.0: calibration_total(model, fold, tahoe.grid_paths(model, fold, (31002,))[31002]['nested'])}
            for w in BUILT_GRID + BUILT_EXTRA.get((fold, model), ()):
                totals[w] = calibration_total(model, fold, built_folder(fold, model, 31002, 21, w) / 'predictions-parent.npz')
            chosen.setdefault(model, {})[fold] = {'weight_decay': choose(totals), 'calibration_totals': {str(k): v for k, v in totals.items()}}
    for fold in FOLDS:
        totals = {0.0: calibration_total('official_state', fold, tahoe.state_grids(31002, 5000, fold)['nested'])}
        for d in sorted((STATE / fold).glob('st_anchored_or21_L2_31002_wd*')):   # the original's order: by folder name
            totals[float(d.name.split('_wd')[1].replace('p', '.'))] = calibration_total('official_state', fold, d / 'predictions-or-grid.npz')
        chosen.setdefault('official_state', {})[fold] = {'weight_decay': choose(totals), 'calibration_totals': {str(k): v for k, v in totals.items()}}
    (Path(out) / 'SELECTION.json').write_text(json.dumps(chosen, indent=1) + '\n')


def fit_path(model, fold, seed, w):
    if model == 'official_state':
        return state_folder(fold, seed, 28, next(t for t in STATE_COEFFICIENTS if np.isclose(float(t), w))) / 'predictions-or-grid.npz'
    return built_folder(fold, model, seed, 28, w) / 'predictions-parent.npz'


def read_case(job):
    model, fold, w = job
    case = load_case('tahoe', fold, model, True)
    m = net_cost.calibrate_original(read_side('tahoe', case, 'calibration', None))
    held = read_side('tahoe', case, 'held-out', None)
    u, a = held['u'], held['a']
    T = net_cost.design_baseline(held, m['kappa'])
    rows, lined = [], {ref: [] for ref in REFERENCES}
    for s in SEEDS:
        p, v = held['p'][s], m['seeds'][str(s)]
        x = {'original': p, 'weight_decay': grids.load(fit_path(model, fold, s, w), 'outer', u),
             'original_field': menu.two_part(a, p, *v['original_field'], u),
             'original_design': menu.two_part(T, p, *v['original_design'], u)}
        r = {k: scores(y, u) for k, y in x.items()}
        pts = 100 / r['original']['total_mse']
        row = {'model': model, 'fold': fold, 'seed': s, 'weight_decay': w, 'kappa': m['kappa']}
        for k in ('weight_decay', 'original_field', 'original_design'):
            row[f'{k}_gain'] = pts * (r['original']['total_mse'] - r[k]['total_mse'])
        lines_wd = line_parts(x['weight_decay'], u)['total']
        for ref in REFERENCES:
            row[f'wd_vs_{ref}'] = pts * (r[ref]['total_mse'] - r['weight_decay']['total_mse'])
            row[f'wd_vs_{ref}_common'] = pts * (r[ref]['common_error_mse'] - r['weight_decay']['common_error_mse'])
            row[f'wd_vs_{ref}_contrast'] = pts * (r[ref]['difference_error_mse'] - r['weight_decay']['difference_error_mse'])
            lines_ref = line_parts(x[ref], u)['total']
            lined[ref].append(({line: lines_ref[line] - lines_wd[line] for line in lines_wd}, r['original']['total_mse']))
        for k in ('original', 'weight_decay'):
            al = card.alignment(x[k], a, u)
            for part, label in (('common', 'context'), ('differential', 'perturbation')):
                for q in ('best_scale', 'cosine', 'size_over_anchor_error'):
                    row[f'{k}_{label}_{q}'] = al[part][q]
        rows.append(row)
    mean = {'model': model, 'fold': fold, 'seed': 'mean', 'weight_decay': w, 'kappa': m['kappa']}
    for k in rows[0]:
        if k not in mean:
            mean[k] = float(np.mean([r[k] for r in rows]))
    for ref in REFERENCES:
        mean[f'wd_vs_{ref}_low'], mean[f'wd_vs_{ref}_high'] = bootstrap(*pool_seeds(lined[ref]))['interval_pct']
    print(model, fold, flush=True)
    return rows + [mean]


def reading_stage(out, workers=3):
    selection = json.loads((Path(out) / 'SELECTION.json').read_text())
    jobs = [(model, fold, selection[model][fold]['weight_decay']) for model in ('summary', 'deepsets', 'official_state') for fold in FOLDS]
    with Pool(workers) as pool:
        rows = [r for case in pool.imap(read_case, jobs) for r in case]
    with open(Path(out) / 'WEIGHT_DECAY_READING.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Weight decay as a comparator on Tahoe-100M.')
    parser.add_argument('--stage', choices=('select', 'reading'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    (select_stage if args.stage == 'select' else reading_stage)(args.out)


if __name__ == '__main__':
    main()
