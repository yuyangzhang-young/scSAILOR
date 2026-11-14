"""The saved per-fit scores of the paper's synthetic-answer fits, read from the data root.

The synthetic predictions (about 3 TB) are not deposited; each fit's scores are. Under
``<root>/synthetic-answer-scores/<screen>/<fold>/`` (Parse PBMC: donor groups G1-G3 as folds):

* ``<model>_<seed>_<level>[_lr03]/SCORES.json``: one fit's scores as ``scsailor.assay.score`` writes them (or the
  same scoring in the published models' packages). Its held-out side (``HELDOUT``) holds the 25 planted states' E, M,
  A and K averaged with equal weight over every held-out context (``score.pool_contexts``). ``_lr03`` marks a fit
  retrained at 0.3 times the standard peak learning rate, kept beside the collapsed fit of the same seed.
* ``POOLED_SCORES_BUILT_MODELS.csv`` (folds 1-3) and ``POOLED_SCORES_PREPR.csv`` (folds 2-3), on Tahoe-100M: the same
  pooled rows of the built models and of PrePR-CT, whose score files are not deposited. They keep their saved
  columns: the attention model is ``state`` and PrePR-CT ``PrePR-CT``; ``scope`` ``F1`` and ``support`` ``heldout``
  are the held-out rows.

``fit_scores`` returns one row per fit and planted state. Recovery and projection slope are computed from the pooled
E, M and K (Methods), the same way for every source.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from scsailor.assay.design import STATISTICS
from scsailor.assay.instance import HELDOUT

from .study import SAVED_NAMES

__all__ = ["SCORES_FOLDER", "SCREENS", "MODELS", "LEVELS", "CUES", "RESPONSES", "LEARNING_RATES", "SAVED_MODELS",
           "describe", "held_out", "read_saved", "fit_scores"]

SCORES_FOLDER = Path("synthetic-answer-scores")
SCREENS = {"tahoe-100m": "Tahoe-100M", "gse306429": "GSE306429", "parse-pbmc": "Parse PBMC",
           "replogle-nadig": "Replogle–Nadig", "jiang": "Jiang et al."}
MODELS = {"attention_model": "attention model", "deepsets": "DeepSets model", "summary": "summary-statistics model",
          "state": "State", "cpa": "CPA", "prepr": "PrePR-CT"}
LEVELS = {"L1": "Mean", "L2": "+Var", "L3": "+Dist", "L3d": "+DepCp", "L4": "Cells"}   # L0 (depth only) is not read
CUES = {"mean": "shift", "variance": "spread", "skewness": "skew", "depth_coupling": "depth trend",
        "gene_joint": "covariation"}
RESPONSES = ("shared", "selective", "weak shared", "weak selective", "null")             # planted state a % 5
LEARNING_RATES = {None: "standard", "lr03": "0.3x standard"}   # peak learning rate; run suffix
# Model codes in the saved pooled and whole-profile tables (there ``state`` is the attention model; in the score
# files' folder names it is State).
SAVED_MODELS = {**SAVED_NAMES, "state_pub": "state", "PrePR-CT": "prepr"}
RUN = re.compile(r"(?P<model>.+)_(?P<seed>\d+)_(?P<level>L[0-4]d?)(?:_(?P<rate>lr03))?")


def describe(a):
    """Planted state ``a``'s cue and response."""
    return {"planted_state": a, "cue": CUES[STATISTICS[a // 5][0]], "response": RESPONSES[a % 5]}


def _ratios(E, M, A, K):
    """A pooled row's energies with recovery 1 - M/E and projection slope K/(2E), None where E = 0 (null states)."""
    return {"true_energy": E, "squared_error": M, "predicted_energy": A,
            "recovery": 1 - M / E if E > 0 else None, "projection_slope": K / (2 * E) if E > 0 else None}


def held_out(path):
    """The 25 planted states of a saved SCORES.json, held out (``_ratios`` of its pooled rows)."""
    rows = json.loads(Path(path).read_text())[HELDOUT]["modules"]
    return [{"planted_state": row["module"], **_ratios(row["E"], row["M"], row["A"], row["K"])} for row in rows]


def read_saved(path):
    """A saved table read with exact floats: its rows of levels L1-L4, with the public model codes."""
    table = pd.read_csv(path, float_precision="round_trip")
    table = table[table.level.isin(list(LEVELS))]
    return table.assign(model=[SAVED_MODELS.get(m, m) for m in table.model])


def _fit(screen, fold, model, seed, level, rate, states):
    return [{"screen": SCREENS[screen], "fold": fold, "model": MODELS[model], "seed": int(seed),
             "level": LEVELS[level], "learning_rate": LEARNING_RATES[rate], **describe(row["planted_state"]), **row}
            for row in states]


def _pooled_table(path, screen, fold):
    """The held-out rows of a saved pooled table, one fit per model, seed and level."""
    table = read_saved(path)
    side = table["scope"] == HELDOUT if "scope" in table else table["support"] == "heldout"
    table = table[side].rename(columns={"module_id": "module"})
    out = []
    for (model, seed, level), group in table.groupby(["model", "seed", "level"], sort=False):
        states = [{"planted_state": int(r.module), **_ratios(r.E, r.M, r.A, r.K)}
                  for r in group.sort_values("module").itertuples()]
        out += _fit(screen, fold, model, seed, level, None, states)
    return out


def fit_scores(root):
    """One row per fit and planted state, held out: screen, fold, model, seed, level, learning_rate, planted_state,
    cue, response, the pooled true_energy (E), squared_error (M) and predicted_energy (A), recovery and
    projection_slope."""
    rows = []
    for screen_dir in sorted((Path(root) / SCORES_FOLDER).iterdir()):
        for fold_dir in sorted(screen_dir.iterdir()):
            for path in sorted(fold_dir.glob("*/SCORES.json")):
                run = RUN.fullmatch(path.parent.name)
                rows += _fit(screen_dir.name, fold_dir.name, run["model"], run["seed"], run["level"], run["rate"],
                             held_out(path))
            for path in sorted(fold_dir.glob("POOLED_SCORES_*.csv")):
                rows += _pooled_table(path, screen_dir.name, fold_dir.name)
    return pd.DataFrame(rows)
