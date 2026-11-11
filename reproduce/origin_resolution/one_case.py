"""One case of origin resolution from its saved predictions: the card of the summary-statistics model on Jiang et al.'s
fold 4 (calibration on the nested model's calibration lines, cost attribution, the held-out readings), compared with
its expected files.

The card's stages run into ``--out`` (``cards``, each in its own process), reading the data root, and every
expected file is compared with the new one: CSV byte for byte, JSON without wall time and memory and with the
``--out`` folder's path taken out of its strings (the attribution record names the calibration record by it; the
expected record is stored without it; every data path is relative to the root). Where a CSV differs,
the number of differing cells and the largest relative difference among numbers are printed: on another CPU the last
digits of a value can differ (the four-part split of WHERE rests on SVDs). The data the case reads are 18 files, 0.62 GB
(the six grids of its nested and full models, the fold's units, input sets, references and noise tables, its selection);
the expected files are kept beside this module (``one_case_expected/``), the package's own output under a data root.

    python -m reproduce.origin_resolution.one_case --data ROOT --out DIR
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from . import ROOT

__all__ = ["CASE", "EXPECTED", "VOLATILE", "comparable", "same", "main"]

CASE = ('--dataset', 'jiang', '--model', 'summary', '--fold', 'F4')
EXPECTED = Path(__file__).with_name('one_case_expected')   # the files the case must give
VOLATILE = ('wall_seconds', 'peak_rss_mb')   # a run's own wall time and memory


def comparable(value, out=''):
    """A JSON value without wall time and memory, with ``out`` taken out of its strings."""
    if isinstance(value, dict):
        return {k: comparable(v, out) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [comparable(v, out) for v in value]
    return value.replace(out, '') if isinstance(value, str) and out else value


def same(new, expected, out=''):
    """(equal, note) of a new file and its expected copy; ``out``: the new run's folder as its records spell it."""
    if expected.suffix == '.json':
        return comparable(json.loads(new.read_text()), out) == comparable(json.loads(expected.read_text())), ''
    if new.read_bytes() == expected.read_bytes():
        return True, ''
    rows = [list(csv.reader(open(p, newline=''))) for p in (new, expected)]
    cells = [(a, b) for x, y in zip(*rows) for a, b in zip(x, y) if a != b]
    worst = 0.
    for a, b in cells:
        try:
            worst = max(worst, abs(float(a) - float(b)) / max(abs(float(b)), 1e-300))
        except ValueError:
            worst = float('inf')   # a text cell differs
    shape = 'same shape' if [len(r) for r in rows[0]] == [len(r) for r in rows[1]] else 'another shape'
    return False, f'{len(cells)} cells differ, largest relative difference {worst:.1e}, {shape}'


def main(argv=None):
    parser = argparse.ArgumentParser(description='One case of origin resolution, compared with its expected files.')
    parser.add_argument('--data', type=Path, help='the data root (SCSAILOR_DATA); default: the environment')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    root, out = (args.data or ROOT), args.out.resolve()   # the stages run in the repository root
    root = root and root.resolve()
    env = {**os.environ, **({'SCSAILOR_DATA': str(root)} if root else {})}
    out.mkdir(parents=True, exist_ok=True)
    for stage in ('calibration', 'attribution'):
        subprocess.run([sys.executable, '-B', '-m', 'reproduce.origin_resolution.cards', *CASE, '--stage', stage, '--out', str(out)],
                       cwd=Path(__file__).resolve().parents[2], check=True, env=env)
    results = {p.name: same(out / p.name, p, f'{out}/') for p in sorted(EXPECTED.iterdir())}
    for name, (equal, note) in results.items():
        print('equal ' if equal else 'DIFFERS', name, note)
    sys.exit(0 if all(equal for equal, _ in results.values()) else 1)


if __name__ == '__main__':
    main()
