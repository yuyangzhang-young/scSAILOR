"""Recovery of the planted responses: the per-fit table, its seed and fold means (Fig. 3c), the isolation checks
and the sampled-label positive control.

* ``fits_table``: every fit and planted state (``scores.fit_scores``) with its status: an attention-model fit is
  excluded as a training collapse when, for every non-null planted state, its predicted energy is below 1% of the
  true energy. On Replogle–Nadig each collapsed fit has a fit of the same seed retrained at 0.3 times the standard
  peak learning rate, deposited as a separate fit (``learning_rate`` ``0.3x standard``) and kept like any other.
  ``cue_in_input`` says whether the input at the fit's level carries the planted cue (recovery is 0 by
  construction where it does not).
* ``by_fold``: per screen, fold, model, level and non-null planted state, the mean over the kept seeds.
* ``fold_means``: the mean of those seed means over the folds, each fold weighing equally (Fig. 3c).
* ``isolation``: per model, over the kept fits, the largest predicted energy of a non-null state where the cue is
  absent from the input, and the null states' leakage (``scsailor.assay.capability.leakage_ratio``) where it is
  present.
* ``positive_control``: recovery of the summary-statistics model trained on exact or sampled labels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from scsailor.assay.capability import leakage_ratio
from scsailor.assay.design import visible

from .scores import LEVELS, MODELS, describe, held_out

__all__ = ["COLLAPSE", "COLLAPSED", "FIT", "POSITIVE_CONTROL", "fits_table", "by_fold", "fold_means", "isolation",
           "positive_control"]

COLLAPSE = .01            # an attention-model fit below this predicted/true energy on every non-null state collapsed
COLLAPSED = "excluded: training collapse"
FIT = ["screen", "fold", "model", "seed", "level", "learning_rate"]                 # the columns that name a fit
CELL = ["screen", "model", "level", "planted_state", "cue", "response", "cue_in_input"]
LEVEL_CODES = {name: code for code, name in LEVELS.items()}
POSITIVE_CONTROL = Path("synthetic-answer-positive-control")   # <labels>/SCORES.json, labels "exact" or "sampled"


def fits_table(fits):
    """``fits`` with ``fit_status`` (kept, or excluded as a training collapse) and ``cue_in_input``."""
    non_null = fits[fits.response != "null"]
    largest = (non_null.predicted_energy / non_null.true_energy).groupby([non_null[k] for k in FIT]).max()
    attention = largest.index.get_level_values("model") == MODELS["attention_model"]
    collapsed = set(largest[attention & (largest < COLLAPSE)].index)
    status = [COLLAPSED if key in collapsed else "kept" for key in fits[FIT].itertuples(index=False, name=None)]
    table = fits.copy()
    table.insert(len(FIT), "fit_status", status)
    table.insert(table.columns.get_loc("response") + 1, "cue_in_input",
                 [visible(a, LEVEL_CODES[level]) for a, level in zip(table.planted_state, table.level)])
    return table


def by_fold(table):
    """Seed means of recovery over the kept fits, per screen, fold, model, level and non-null planted state."""
    kept = table[(table.fit_status == "kept") & (table.response != "null")]
    out = kept.groupby(["screen", "fold", "model", "level", "planted_state", "cue", "response", "cue_in_input"],
                       sort=False).recovery.agg(n_seeds="size", recovery="mean")
    return out.reset_index()


def fold_means(folds):
    """The mean over folds of the seed means, each fold weighing equally (Fig. 3c), with the folds averaged."""
    return folds.groupby(CELL, sort=False).agg(n_folds=("fold", "size"), folds=("fold", ";".join),
                                               recovery=("recovery", "mean")).reset_index()


def isolation(table):
    """Per model, over the kept fits of every screen: the largest predicted energy of a non-null state at a level
    whose input lacks its cue (and its largest ratio to the true energy), and the largest and median leakage ratio of
    the null states whose cue is in the input."""
    kept = table[table.fit_status == "kept"]
    absent = kept[(kept.response != "null") & ~kept.cue_in_input]
    energies = kept.set_index(FIT + ["cue", "response"]).true_energy
    null = kept[(kept.response == "null") & kept.cue_in_input]
    keys = null[FIT + ["cue"]].itertuples(index=False, name=None)
    ratios = null.assign(ratio=[leakage_ratio(A, energies[k + ("shared",)], energies[k + ("selective",)])
                                for A, k in zip(null.predicted_energy, keys)])
    out = pd.DataFrame({
        "n_fits": kept.drop_duplicates(FIT).groupby("model").size(),
        "largest_size_without_cue": absent.groupby("model").predicted_energy.max(),
        "largest_ratio_without_cue": (absent.predicted_energy / absent.true_energy).groupby(absent.model).max(),
        "largest_null_ratio": ratios.groupby("model").ratio.max(),
        "median_null_ratio": ratios.groupby("model").ratio.median()})
    return out.rename_axis("model").reset_index()


def positive_control(root):
    """Recovery of the summary-statistics model at Mean on Tahoe-100M fold 1 (seed 31002), trained on the exact
    full-pool targets or on the means of 256 sampled synthetic treated cells, for the non-null shift states."""
    return pd.DataFrame([{"labels": labels, **describe(row["planted_state"]), "recovery": row["recovery"]}
                         for labels in ("exact", "sampled")
                         # planted states 0-3, the non-null shift tests
                         for row in held_out(Path(root) / POSITIVE_CONTROL / labels / "SCORES.json")[:4]])
