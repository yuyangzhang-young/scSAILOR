"""Recovery against the whole-profile scores on Tahoe-100M (Fig. 3b, Supplementary Note S6.5).

Each Tahoe-100M fit's five whole-profile scores on the synthetic task were computed from its predictions and saved
per fold, in ``<root>/synthetic-answer-scores/tahoe-100m/<fold>/WHOLE_PROFILE_SCORES_*.csv`` (the built models' and
the published models' tables, with their saved columns: ``total.mse``, ``systema.pearson_ref_train``,
``mz.r_gamma``, ``dbd.r2w_delta``, ``systema.centroid_accuracy``).

* ``scores_table``: one row per kept Tahoe-100M fit: the five scores and its recovery summed over the five shared
  and over the five selective planted states (every cue, whether or not the input carries it).
* ``correlations``: Spearman correlations (average ranks for ties; total MSE sign-reversed) between the seven, within
  each fold over its fits, then the mean of the folds: over all six models together, and within each model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .scores import LEVELS, MODELS, SCORES_FOLDER, read_saved

__all__ = ["PROFILE_COLUMNS", "LABELS", "ALL", "scores_table", "correlations"]

SAVED = {"total.mse": "total_mse", "systema.pearson_ref_train": "pearson_delta", "mz.r_gamma": "interaction_alignment",
         "dbd.r2w_delta": "weighted_r2", "systema.centroid_accuracy": "centroid_accuracy"}
PROFILE_COLUMNS = (*SAVED.values(), "shared_recovery", "selective_recovery")
LABELS = ("total MSE (sign reversed)", "PearsonΔ", "interaction alignment", "adapted weighted R²", "centroid accuracy",
          "shared recovery", "selective recovery")
ALL = "all six models"
TAHOE_FIT = ["fold", "model", "seed", "level"]      # the columns that name a Tahoe-100M fit


def scores_table(root, fits):
    """The kept Tahoe-100M fits (``fits``: rows of ``recovery.fits_table``) with their whole-profile scores: one row per
    fit present in both the saved scores and ``fits`` (on the paper's data, every fit: 90 per fold)."""
    saved = []
    for path in sorted((Path(root) / SCORES_FOLDER / "tahoe-100m").glob("*/WHOLE_PROFILE_SCORES_*.csv")):
        table = read_saved(path)
        saved.append(pd.DataFrame({"fold": path.parent.name, "model": [MODELS[m] for m in table.model],
                                   "seed": table.seed, "level": table.level.map(LEVELS),
                                   **{name: table[column] for column, name in SAVED.items()}}))
    kept = fits[(fits.screen == "Tahoe-100M") & (fits.fit_status == "kept")]
    sums = {f"{response}_recovery": kept[kept.response == response].groupby(TAHOE_FIT).recovery.sum()
            for response in ("shared", "selective")}
    return pd.concat(saved).join(pd.DataFrame(sums), on=TAHOE_FIT, how="inner").reset_index(drop=True)


def _matrix(table):
    """Spearman matrix of ``PROFILE_COLUMNS`` per fold (total MSE sign-reversed), and its fold mean."""
    per_fold = {}
    for fold, group in table.groupby("fold"):
        values = group[list(PROFILE_COLUMNS)].to_numpy(dtype=float)
        values[:, 0] *= -1                                       # total MSE, sign-reversed
        per_fold[fold] = np.corrcoef(rankdata(values, axis=0), rowvar=False)
    return np.mean(list(per_fold.values()), axis=0), per_fold


def correlations(table):
    """Each pair of the seven scores once (lower triangle in ``PROFILE_COLUMNS`` order): the fold-mean Spearman
    correlation and each fold's, over all six models and within each model."""
    rows = []
    for models, group in [(ALL, table)] + list(table.groupby("model")):
        mean, per_fold = _matrix(group)
        for i in range(len(PROFILE_COLUMNS)):
            for j in range(i):
                rows.append({"models": models, "score_row": LABELS[i], "score_column": LABELS[j],
                             "spearman": mean[i, j], **{f"spearman_{f}": m[i, j] for f, m in per_fold.items()}})
    return pd.DataFrame(rows)
