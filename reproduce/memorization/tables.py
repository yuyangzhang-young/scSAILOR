"""Every table of the memorization layer, computed on a CPU from the saved predictions under the data root.

    python -m reproduce.memorization.tables --data DATA --out OUT [--compare ROOT] [--screens SCREEN ...]

Each screen of ``SCREENS`` (all by default) writes its tables under ``OUT/memorization/<screen>/`` (``folder``) with
``to_csv(**data.WRITE)``, in the order in which they depend on each other (``tahoe_tables`` then
``retraining_tables``, ``parse.tables``, ``parse_interferon.tables`` then ``parse_retraining.tables``,
``genetic.tables``); a table computed from others reads them as written. ``--compare``
prints, for each table, ``equal`` or ``DIFFERS`` against the table at the same path under ROOT, cell for cell, or
``not deposited``.
"""
import argparse
import itertools
import os
import time
from pathlib import Path

from ..compare import compare

__all__ = ["SCREENS", "folder", "tahoe_tables", "retraining_tables", "screen_tables", "main"]

SCREENS = ('tahoe-100m', 'parse-pbmc', 'genetic-screens')


def folder(screen):
    """Where a screen's tables are written, under the output root."""
    return Path('memorization') / screen



def tahoe_tables():
    """(file name, table) of every Tahoe-100M table, in dependency order."""
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


def retraining_tables():
    """(file name, table) of every Tahoe-100M table read from the retraining runs (``fits``)."""
    from . import interventions, known_answers, published_tables, retrieval, two_phase
    yield 'TARGET_INTERVENTIONS.csv', interventions.interventions_table()
    yield 'TRAINING_PULL.csv', interventions.training_pull_table()
    yield 'RETRIEVAL.csv', retrieval.retrieval_table()
    yield 'TWO_PHASE.csv', two_phase.comparison_table()
    yield 'TWO_PHASE_OUTPUT_ENERGY.csv', two_phase.output_energy_table()
    yield 'KNOWN_ANSWERS.csv', known_answers.known_answers_table()
    yield 'PUBLISHED_TARGET_INTERVENTIONS.csv', published_tables.interventions_table()
    yield 'PUBLISHED_TWO_PHASE.csv', published_tables.two_phase_table()
    yield 'PUBLISHED_TWO_PHASE_NINE_METRICS_GAINS.csv', published_tables.two_phase_metrics_table()
    yield 'PUBLISHED_RETRIEVAL.csv', published_tables.retrieval_table()


def screen_tables(screen):
    """(file name, table) of every table of one screen, in dependency order."""
    from . import genetic, parse, parse_interferon, parse_retraining
    parts = {'tahoe-100m': [tahoe_tables, retraining_tables],
             'parse-pbmc': [parse.tables, parse_interferon.tables, parse_retraining.tables],
             'genetic-screens': [genetic.tables]}[screen]
    return itertools.chain.from_iterable(part() for part in parts)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the memorization layer's tables from the saved predictions.")
    parser.add_argument('--data', type=Path, required=True, help='the data root')
    parser.add_argument('--out', type=Path, required=True,
                        help='where the tables are written, under memorization/<screen>')
    parser.add_argument('--compare', type=Path, help='a root holding the saved tables at the same paths')
    parser.add_argument('--screens', nargs='+', choices=SCREENS, default=SCREENS, help='the screens to compute (all)')
    args = parser.parse_args(argv)
    # Set before NumPy and the data root's readers load. The paper's tables were computed with four BLAS threads; with
    # another number, sums can be taken in another order and the last digits differ.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '4'
    os.environ['SCSAILOR_DATA'] = str(args.data.resolve())
    from .data import WRITE

    start = time.time()
    for screen in args.screens:
        where, names = folder(screen), []
        (args.out / where).mkdir(parents=True, exist_ok=True)
        for name, table in screen_tables(screen):
            table.to_csv(args.out / where / name, **WRITE)
            names.append(name)
            print(f'{time.time() - start:5.0f} s  {where / name}', flush=True)
        if args.compare:
            print('\n'.join(compare(args.out, args.compare, [where / name for name in names])), flush=True)


if __name__ == '__main__':
    main()
