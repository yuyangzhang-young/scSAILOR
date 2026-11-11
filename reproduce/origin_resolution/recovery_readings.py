"""Held-out recovery read from information localization's run records, with nothing trained, predicted or rescored:
inputs of the per-change supplementary table.

Two stages, writing into ``--out``:

- ``parse-primary``: ``PARSE_PRIMARY_RECOVERY.csv``, the Parse PBMC light runs' recovery on all twelve donors, one
  row per run and target. The stored value is the pair-weighted 1 - error / energy over contexts, reproduced exactly
  over all contexts, so the reading is a recomputation of saved fields. ``primary`` names the evaluated donors, which
  are all of them, so the ``*_primary`` columns equal the ``*_all_contexts`` columns (the pairs exactly, Q to
  rounding);
- ``runs``: ``FIG4_RUN_READINGS.csv``, the held-out recovery Q of the shared shift (bit 0) and the selective
  shift (bit 1) after retraining (``EVAL.json``) and of the original model (``EVAL_PARENT.json``), for every run with an
  evaluation: Tahoe-100M's attention model and State on folds 2-3 (the full map), the light version on the
  other screens, and Replogle-Nadig's attention runs in the light menu (fold 2's synthetic half from the parents
  recovered at learning rate x0.3 only, as the collapse rule and the light reading do).

    python -m reproduce.origin_resolution.recovery_readings --stage parse-primary --out DIR
"""
import argparse
import csv
import json
import math
from pathlib import Path

from . import ROOT, stored

__all__ = ["recovery", "parse_primary_stage", "runs_stage", "main"]

LIGHT_RUNS = ROOT / 'information-localization-light'   # <screen>/<fold>/map/<run>/: the light version's runs
F23_RUNS = ROOT / 'information-localization-folds-2-3'   # F2, F3 /map/<run>/: Tahoe's attention model and State, the full map
REPLOGLE_RUNS = ROOT / 'information-localization-replogle-nadig/replogle'   # F1, F2 /map/<run>/
LIGHT_MENUS = ROOT / 'light-version-cases/MENUS.csv'
OUT_DONORS = ()   # no donor is left out of the evaluation
MODEL = {'ds': 'DeepSets', 'su': 'summary-statistics', 'at': 'attention', 'os': 'official State'}
COLUMNS = ['source', 'dataset', 'model', 'half', 'fold', 'seed', 'stage', 'cut', 'arm', 'rate', 'Q0', 'Q1', 'parent_Q0',
           'parent_Q1', 'run']


def recovery(bit, keep=lambda context: True):
    """(pair-weighted 1 - error / energy over the kept contexts, their pairs) of one bit's per-context fields. The sums
    are correctly rounded (``math.fsum``): the saved table was written with Python 3.13, whose ``sum`` compensates, and
    this gives its values on any Python, where Python 3.11's ``sum`` moves the last digit of 421 of its 764 rows."""
    rows = [(v['n_pairs'], v['truth_halfcontrast_energy'], v['halfcontrast_error']) for c, v in bit['per_context'].items() if keep(c)]
    n = sum(r[0] for r in rows)
    energy = math.fsum(r[0] * r[1] for r in rows) / n
    return 1 - math.fsum(r[0] * r[2] for r in rows) / n / energy, n


def write_rows(path, rows, columns=None):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_primary_stage(out):
    primary = lambda context: context.split('|')[0] not in OUT_DONORS
    rows = []
    for run in sorted((LIGHT_RUNS / 'parse').glob('G*/map/*')):
        path = run / 'EVAL.json'
        if not path.is_file():
            continue
        name = run.name.split('_')
        group, tag, half, seed = name[0][1:], name[1], name[2], name[-1]
        cell = f"{name[3]}/{'_'.join(name[4:-2])}/{name[-2]}"
        after = json.loads(path.read_text())['after']['supports']['heldout']['bits']
        parent = json.loads((run / 'EVAL_PARENT.json').read_text())
        parent = parent.get('after', parent)['supports']['heldout']['bits']
        for bit, values in after.items():
            if values.get('status') != 'defined':
                continue
            whole, n_all = recovery(values)
            if abs(whole - values['Q']) >= 1e-9:
                raise ValueError(f"{run.name} bit {bit}: the per-context fields give {whole}, the record {values['Q']}")
            kept, n_kept = recovery(values, primary)
            rows.append({'group': group, 'model': MODEL[tag], 'half': {'nat': 'natural', 'syn': 'synthetic'}[half], 'cell': cell,
                         'seed': seed, 'target': bit, 'pairs_all': n_all, 'pairs_primary': n_kept, 'Q_all_contexts': values['Q'],
                         'Q_primary': kept, 'parent_Q_all_contexts': parent[bit]['Q'], 'parent_Q_primary': recovery(parent[bit], primary)[0]})
    write_rows(Path(out) / 'PARSE_PRIMARY_RECOVERY.csv', rows)


def heldout_q(path):
    e = json.loads(path.read_text())
    e = e.get('after', e)
    bits = e['supports']['heldout']['bits']
    return bits['0']['Q'], bits['1']['Q']


def read_run(run, source, dataset=None, keep=None):
    """One run's row, or None where it has no evaluation or lies outside ``keep`` (a set of (cut, arm, rate))."""
    if not (run / 'EVAL.json').exists():
        return None
    row = json.loads((run / 'RUN_RECORD.json').read_text())['row']
    rate = float(row.get('rate_multiple', row.get('rate')))
    if keep is not None and (row['cut'], row['arm'], rate) not in keep:
        return None
    q0, q1 = heldout_q(run / 'EVAL.json')
    p0 = p1 = ''
    if (run / 'EVAL_PARENT.json').exists():
        p0, p1 = heldout_q(run / 'EVAL_PARENT.json')
    return dict(source=source, dataset=dataset or row.get('dataset', 'tahoe'), model=row['model'], half=row['half'],
                fold=row['fold'], seed=row['seed'], stage=row.get('stage', 'map'), cut=row['cut'], arm=row['arm'],
                rate=rate, Q0=q0, Q1=q1, parent_Q0=p0, parent_Q1=p1, run=stored(run))


def attention_light_menu():
    """(cut, arm, rate) of the attention model's light menu (Tahoe fold 1 rows of MENUS.csv)."""
    menu = set()
    for r in csv.DictReader(open(LIGHT_MENUS, newline='')):
        if r['model'] == 'attention' and r['fold'] == 'F1' and r['light_rates'] not in ('', '—'):
            for rate in r['light_rates'].split(','):
                menu.add((r['cut'], r['arm'], float(rate.strip().lstrip('x'))))
    return menu


def runs_stage(out):
    rows = []
    for fold in ('F2', 'F3'):
        rows += [r for run in sorted((F23_RUNS / fold / 'map').iterdir()) if (r := read_run(run, 'tahoe folds 2-3, full first wave', 'tahoe'))]
    for screen in ('parse', 'jiang', 'gse306429', 'replogle'):
        base = LIGHT_RUNS / screen
        for fold in sorted(p for p in base.iterdir() if (p / 'map').is_dir()):
            rows += [r for run in sorted((fold / 'map').iterdir()) if (r := read_run(run, 'light', screen))]
    menu = attention_light_menu()
    for fold in ('F1', 'F2'):
        for run in sorted((REPLOGLE_RUNS / fold / 'map').iterdir()):
            if '_at_' not in run.name or (fold == 'F2' and '_syn_' in run.name and not run.name.endswith('_lr03p')):
                continue   # attention runs only; fold 2's collapsed synthetic parents: the lr x0.3 recovered ones stand in
            if r := read_run(run, 'light (Replogle-Nadig attention, light menu)', 'replogle', keep=menu):
                rows.append(r)
    write_rows(Path(out) / 'FIG4_RUN_READINGS.csv', rows, COLUMNS)


STAGES = {'parse-primary': parse_primary_stage, 'runs': runs_stage}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Held-out recovery read from information localization's run records.")
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    STAGES[args.stage](args.out)


if __name__ == '__main__':
    main()
