"""The known-answer test of origin resolution: planted changes on every model and fold, read back from saved grids.

Two stages, writing into ``--out``:

- ``read-back``: every planted change of ``planted.grid`` on each model and fold of ``slopes.CASES``, with the
  model-level amounts of the matched comparisons (``AMOUNTS``, written by ``net_cost --stage calibration``):
  ``cases/<screen>_<fold>_<model>.csv``, one per model and fold, then ``READ_BACK.csv``, their rows in file-name order.
  The worst gaps keep the matched comparisons' column names: ``min_gap_set1`` and ``min_gap_set2`` are those of the
  standard and the design-aware comparisons, and ``min_gap_set3`` adds the second training run;
  ``recommended_set*_m*`` are the saved tables' screen at each margin (``net_cost.recommended``), not the paper's rule;
- ``margins``: from ``READ_BACK.csv``, the planted changes the recommendation rule recommends at each margin, over all
  screens and on Tahoe-100M: ``NULLS_BY_MARGIN.csv``. A planted change has no fixed intervention, so the rule reads its
  kept fraction and the worst gap of the standard or the design-aware comparison.

The saved tables were read with one thread per process:

    OMP_NUM_THREADS=1 python -m reproduce.origin_resolution.planted --stage read-back --out DIR --workers 6
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from scsailor.resolution import net_cost, planted
from . import ROOT, read_csv, write_csv
from .net_cost import MARGINS as SET_MARGINS, read_side, recommended
from .slopes import BINDINGS, CASES, load_case

__all__ = ["AMOUNTS", "MARGINS", "SCOPES", "read_back_case", "read_back_stage", "margins_stage", "main"]

AMOUNTS = ROOT / 'net-cost/AMOUNTS.json'
MARGINS = (0., .05, .1, .2, .5)   # % of the original model's total MSE
SCOPES = (('all screens', None), ('Tahoe-100M', 'tahoe'))


def read_back_case(job):
    """The planted changes of one model and fold, written to ``cases/``. ``job``: (stream, (screen, fold, model),
    model-level amounts, cases folder)."""
    stream, (screen, fold, model), amounts, out = job
    cal = read_side(screen, load_case(screen, fold, model, False), 'calibration', None)
    held = read_side(screen, load_case(screen, fold, model, True), 'held-out', None)
    lines = len(BINDINGS[screen].calibration_lines(fold))
    rows = []
    for part, s, g, energy in planted.grid():
        row = {'screen': screen, 'fold': fold, 'model': model, 'calibration_lines': lines, 'part': part, 'slope': s, 'return_pts': g,
               'energy_pts': energy}
        found = planted.read_back(cal, held, amounts, stream, part, s, energy)
        if found is None:
            rows.append({**row, 'feasible': False})
            continue
        readings, gaps = found
        row |= {'feasible': True, **readings}
        for stage in (1, 2, 3):
            row[f'min_gap_set{stage}'] = min(min(v.values()) for v in gaps[stage].values())
            for margin in SET_MARGINS:
                row[f'recommended_set{stage}_m{margin:g}'] = recommended(gaps[stage], [readings['alpha_min']], margin)
        rows.append(row)
    write_csv(Path(out) / f'{screen}_{fold}_{model}.csv', rows)
    print(screen, fold, model, len(rows), flush=True)


def read_back_stage(out, workers):
    out = Path(out)
    (out / 'cases').mkdir(parents=True)
    amounts = json.loads(AMOUNTS.read_text())
    jobs = [(stream, case, amounts['|'.join(case)], out / 'cases') for stream, case in enumerate(CASES)]
    with Pool(workers) as pool:
        list(pool.imap_unordered(read_back_case, jobs, chunksize=1))
    write_csv(out / 'READ_BACK.csv', [r for path in sorted((out / 'cases').glob('*.csv')) for r in read_csv(path)])


def standing(rows, column, margin):
    """How many of the planted changes the rule recommends at the margin, on the worst gaps of ``column``."""
    return sum(net_cost.recommend(float(r['alpha_min']), float(r[column]), margin=margin) for r in rows)


def margins_stage(out):
    out = Path(out)
    rows = [r for r in read_csv(out / 'READ_BACK.csv') if r['feasible'] == 'True']
    table = []
    for margin in MARGINS:
        for label, screen in SCOPES:
            part = [r for r in rows if screen in (None, r['screen'])]
            nulls = [r for r in part if float(r['slope']) == 0]
            row = {'margin': margin, 'scope': label, 'nulls_standing_standard': standing(nulls, 'min_gap_set1', margin),
                   'nulls_standing_matched': standing(nulls, 'min_gap_set2', margin), 'nulls': len(nulls)}
            for g in planted.RETURNS:
                hit = [r for r in part if float(r['slope']) > 0 and float(r['return_pts']) == g]
                row[f'return_{g}_standing_matched'] = f"{standing(hit, 'min_gap_set2', margin)}/{len(hit)}"
            table.append(row)
    write_csv(out / 'NULLS_BY_MARGIN.csv', table, mode='w')


def main(argv=None):
    parser = argparse.ArgumentParser(description='The known-answer test of origin resolution on planted changes.')
    parser.add_argument('--stage', choices=('read-back', 'margins'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6, help='models and folds read in parallel (read-back)')
    args = parser.parse_args(argv)
    if args.stage == 'read-back':
        read_back_stage(args.out, args.workers)
    else:
        margins_stage(args.out)


if __name__ == '__main__':
    main()
