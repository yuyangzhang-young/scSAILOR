"""Every table of the information ladder (Results 1-2), from the saved scores and the fold data under the data root.

    python -m reproduce.ladder.tables --data DATA --out OUT [--compare ROOT]

The tables are written under ``OUT/ladder/`` with floats at full precision, in the order in which they depend on each
other (``tables``). ``--compare`` prints, for each table, ``equal`` or ``DIFFERS`` against
the table at the same path under ROOT, cell for cell, or ``not deposited`` (``reproduce.compare``).
"""
import argparse
import os
import time
from pathlib import Path

from ..compare import compare

__all__ = ["FOLDER", "tables", "main"]

FOLDER = Path('ladder')


def tables():
    """(file name, table) of every table, in dependency order."""
    from . import fits, intervals, ladder, plate, references, weighted_r2
    fit_table = fits.fit_table()
    yield 'FIT_TABLE.csv', fit_table
    points = ladder.summarize(fit_table)
    yield 'LADDER.csv', points
    yield 'PARSE_PBMC_DONORS.csv', ladder.parse_donors_table(fit_table)
    yield 'FIT_EXCLUSIONS.csv', ladder.exclusions_table(fit_table)
    yield 'FIT_SENSITIVITY.csv', ladder.sensitivity_table(fit_table)
    yield 'NO_CHANGE.csv', ladder.no_change_table(points)
    refs = references.references_table()
    yield 'REFERENCES.csv', refs
    yield 'MODELS_VS_REFERENCES.csv', references.models_table(points, refs)
    yield 'INTERVALS.csv', intervals.intervals_table()
    plate_rows = plate.plate_table()
    yield 'PLATE_AWARE_BASELINE.csv', plate_rows
    yield 'SUPPLEMENTARY_TABLE_7.csv', plate.supp_table_7(points, plate_rows)
    yield 'TAHOE_PLATE_COMPARISON.csv', plate.comparison_table(fit_table, plate_rows)
    yield 'WEIGHTED_R2.csv', weighted_r2.plotted_table(fit_table)
    yield 'WEIGHTED_R2_REFERENCES.csv', weighted_r2.references_table()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the information ladder's tables from the saved scores.")
    parser.add_argument('--data', type=Path, required=True, help='the data root')
    parser.add_argument('--out', type=Path, required=True, help='where the tables are written, under ' + str(FOLDER))
    parser.add_argument('--compare', type=Path, help='a root holding the saved tables at the same paths')
    args = parser.parse_args(argv)
    # Set before NumPy and the data root's readers load: the sums of the paper's tables were taken with four BLAS
    # threads, and with another number the last digits can differ.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ['SCSAILOR_DATA'] = str(args.data.resolve())

    (args.out / FOLDER).mkdir(parents=True, exist_ok=True)
    start, names = time.time(), []
    for name, table in tables():
        table.to_csv(args.out / FOLDER / name, index=False)
        names.append(name)
        print(f'{time.time() - start:5.0f} s  {FOLDER / name}', flush=True)
    if args.compare:
        print('\n'.join(compare(args.out, args.compare, [FOLDER / name for name in names])))


if __name__ == '__main__':
    main()
