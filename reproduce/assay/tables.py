"""Every table of the synthetic-answer assay in the paper (Results 3, Fig. 3, Supplementary Note S6), computed on a CPU
from the saved per-fit scores and the deposited instances under the data root.

    python -m reproduce.assay.tables --data DATA --out OUT [--compare ROOT]

The tables are written under ``OUT/synthetic-answer-assay/`` at full precision, in the order of ``tables``; the
constructed predictions, the attainable references and the carrier check, which read the instances, come last.
``--compare`` prints, for each table, ``equal`` or ``DIFFERS`` against the table at the same path under ROOT, cell for
cell, or ``not deposited``.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from ..compare import compare
from . import checks, constructed, fold_selection, recovery, references, scores, study, whole_profile

__all__ = ["FOLDER", "CONSTRUCTED_FOLDS", "REFERENCE_INSTANCES", "CHECKED_INSTANCE", "tables", "attainable_references",
           "main"]

FOLDER = Path("synthetic-answer-assay")                                                      # the tables, written
CONSTRUCTED_FOLDS = ("F1", "F2", "F3")                                                        # Tahoe-100M, Fig. 3d
REFERENCE_INSTANCES = (("tahoe", "F1"), ("tahoe", "F2"), ("tahoe", "F3"), ("parse", "G1"))   # Supplementary Table 9
CHECKED_INSTANCE = ("tahoe", "F1")                                                           # the carrier check


def tables(root):
    """(file name, table) of every table."""
    fits = recovery.fits_table(scores.fit_scores(root))
    yield "FITS.csv", fits
    folds = recovery.by_fold(fits)
    yield "RECOVERY_BY_FOLD.csv", folds
    yield "RECOVERY.csv", recovery.fold_means(folds)
    yield "ISOLATION.csv", recovery.isolation(fits)
    profile = whole_profile.scores_table(root, fits)
    yield "WHOLE_PROFILE_SCORES.csv", profile
    yield "WHOLE_PROFILE_CORRELATIONS.csv", whole_profile.correlations(profile)
    yield "POSITIVE_CONTROL.csv", recovery.positive_control(root)
    yield "GENETIC_FOLD_SELECTION.csv", fold_selection.fold_selection(root)
    cases = pd.concat([constructed.constructed_table(study.load_instance(root, "tahoe", fold), fold)
                       for fold in CONSTRUCTED_FOLDS], ignore_index=True)
    yield "CONSTRUCTED_PREDICTIONS.csv", cases
    yield "CONSTRUCTED_PREDICTIONS_FIG3D.csv", constructed.summary(cases)
    yield "ATTAINABLE_REFERENCES.csv", attainable_references(root)
    yield "CARRIER_CHECK.csv", checks.carrier_check(study.load_instance(root, *CHECKED_INSTANCE))


def attainable_references(root):
    """The decoded-cue reference on each instance of ``REFERENCE_INSTANCES``."""
    parts = []
    for dataset, fold in REFERENCE_INSTANCES:
        part = references.decoded_cue_reference(study.load_instance(root, dataset, fold))
        part.insert(0, "fold", fold)
        part.insert(0, "screen", scores.SCREENS[study.DATASETS[dataset]])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the synthetic-answer assay's tables from the saved per-fit "
                                                 "scores and the instances.")
    parser.add_argument("--data", type=Path, required=True, help="the data root")
    parser.add_argument("--out", type=Path, required=True, help="where the tables are written, under " + str(FOLDER))
    parser.add_argument("--compare", type=Path, help="a root holding the saved tables at the same paths")
    args = parser.parse_args(argv)
    (args.out / FOLDER).mkdir(parents=True, exist_ok=True)
    start, names = time.time(), []
    for name, table in tables(args.data):
        table.to_csv(args.out / FOLDER / name, index=False)
        names.append(name)
        print(f"{time.time() - start:5.0f} s  {FOLDER / name}", flush=True)
    if args.compare:
        print("\n".join(compare(args.out, args.compare, [FOLDER / name for name in names])))


if __name__ == "__main__":
    main()
