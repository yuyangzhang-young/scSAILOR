"""Every table of the memorization layer on Tahoe-100M, computed on a CPU from the saved predictions under the data
root.

    python -m reproduce.memorization.tables --data DATA --out OUT [--compare ROOT]

The tables are written under ``OUT/memorization/tahoe-100m/`` with ``to_csv(**data.WRITE)``, in the order in which they
depend on each other (``tables``); a table computed from others reads them as written. ``--compare`` prints, for each
table, ``equal`` or ``DIFFERS`` against the table at the same path under ROOT, cell for cell, or ``not deposited``.
"""
import argparse
import csv
import os
import time
from pathlib import Path

__all__ = ["FOLDER", "tables", "compare", "main"]

FOLDER = Path('memorization/tahoe-100m')


def tables():
    """(file name, table) of every table, in dependency order."""
    from . import arms, growth_genes, metrics, ranking, selection, storage
    four_arms = arms.four_arms_table()
    yield 'FOUR_ARMS.csv', four_arms
    yield 'FOUR_ARMS_SEED_MEANS.csv', arms.seed_means_table(four_arms)
    values, gains = metrics.values_table(), metrics.gains_table()
    yield 'NINE_METRICS_VALUES.csv', values
    yield 'NINE_METRICS_GAINS.csv', gains
    yield 'NINE_METRICS_SUMMARY.csv', metrics.summary(gains)
    yield 'NINE_METRICS_ABSOLUTE.csv', metrics.absolute_values(four_arms, values)
    yield 'ARM_CHANGES.csv', metrics.arm_changes(four_arms)
    yield 'GROWTH_SELECTION.csv', selection.selection_table()
    yield 'GROWTH_GENE_CORRECTION_SEEDS.csv', growth_genes.readouts_table()
    yield 'GROWTH_GENE_CORRECTION_CHANGES.csv', growth_genes.changes_table()
    yield 'GROWTH_LINES.csv', ranking.growth_lines_table()
    growth = ranking.growth_ranking_table()
    yield 'GROWTH_RANKING.csv', growth
    pull = ranking.pull_table()
    yield 'PULL.csv', pull
    line_ranking = ranking.line_ranking_table(growth, pull)
    yield 'LINE_RANKING_TABLE.csv', line_ranking
    yield 'RANKING_FOLD_MEANS.csv', ranking.fold_means(line_ranking)
    yield 'STORAGE.csv', storage.storage_table()
    yield 'TRAINING_LENGTH.csv', storage.training_length_table()


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.reader(handle))


def compare(out, root, names):
    """Per table ``names`` written under ``out``: 'equal' or 'DIFFERS' against the table at the same path under
    ``root``, cell for cell, or 'not deposited'."""
    lines = []
    for name in names:
        path = (FOLDER / name).as_posix()
        if not (root / FOLDER / name).is_file():
            lines.append(f'not deposited  {path}')
            continue
        same = read_rows(root / FOLDER / name) == read_rows(out / FOLDER / name)
        lines.append(f"{'equal' if same else 'DIFFERS'}  {path}")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the memorization layer's tables on Tahoe-100M from the saved "
                                                 "predictions.")
    parser.add_argument('--data', type=Path, required=True, help='the data root')
    parser.add_argument('--out', type=Path, required=True, help='where the tables are written, under ' + str(FOLDER))
    parser.add_argument('--compare', type=Path, help='a root holding the saved tables at the same paths')
    args = parser.parse_args(argv)
    # Set before NumPy and the data root's readers load. The paper's tables were computed with four BLAS threads; with
    # another number, sums can be taken in another order and the last digits differ.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ['SCSAILOR_DATA'] = str(args.data.resolve())
    from .data import WRITE

    (args.out / FOLDER).mkdir(parents=True, exist_ok=True)
    start, names = time.time(), []
    for name, table in tables():
        table.to_csv(args.out / FOLDER / name, **WRITE)
        names.append(name)
        print(f'{time.time() - start:5.0f} s  {FOLDER / name}', flush=True)
    if args.compare:
        print('\n'.join(compare(args.out, args.compare, names)))


if __name__ == '__main__':
    main()
