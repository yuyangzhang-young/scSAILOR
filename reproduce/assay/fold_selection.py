"""The genetic-screen folds of the synthetic-answer assay (Supplementary Note S6.3): on Replogle–Nadig and Jiang et al.
the assay uses the held-out lines with the largest and the smallest fraction of above-noise perturbation-specific
signal explained by the average training response on the natural task.

Per fold, from ``<root>/genetic-screens-held-out-references/<screen>/<fold>/``: ``REFERENCE_SCORES.csv`` holds the
held-out contrast error of predicting no change (the input set's mean; saved
``heldout_input_only_raw_difference_error_mse``) and of the baseline, the input set's mean plus the average training
response (saved ``heldout_anchor_raw_difference_error_mse``); ``NOISE_FLOOR.csv`` holds the contrast error of the
measured responses' own sampling noise (saved ``difference_error_mse``). The fraction, in %, is
100 x (no change - baseline) / (no change - noise floor). The table's columns keep the saved word ``anchor`` for the
baseline.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .scores import SCREENS

__all__ = ["REFERENCES_FOLDER", "HELD_OUT_LINES", "fold_selection"]

REFERENCES_FOLDER = Path("genetic-screens-held-out-references")
HELD_OUT_LINES = {   # each fold's held-out line (Supplementary Data 1, folds)
    "replogle-nadig": {"F1": "K562", "F2": "RPE1", "F3": "HepG2", "F4": "Jurkat"},
    "jiang": {"F1": "A549", "F2": "BXPC3", "F3": "HAP1", "F4": "HT29", "F5": "K562", "F6": "MCF7"},
}


def fold_selection(root):
    """One row per genetic-screen fold: the held-out line, the contrast errors of no change (``no_change_error``) and of
    the baseline (``anchor_error``), the noise floor, the above-noise signal, the fraction the baseline explains
    (``anchor_share_percent``) and, per screen, the folds with the largest and the smallest fraction."""
    rows = []
    for screen_dir in sorted((Path(root) / REFERENCES_FOLDER).iterdir()):
        screen = screen_dir.name
        for folder in sorted(screen_dir.iterdir()):
            fold = folder.name
            reference = pd.read_csv(folder / "REFERENCE_SCORES.csv", float_precision="round_trip").iloc[0]
            noise = pd.read_csv(folder / "NOISE_FLOOR.csv", float_precision="round_trip").difference_error_mse.iloc[0]
            no_change, baseline = (reference.heldout_input_only_raw_difference_error_mse,
                                   reference.heldout_anchor_raw_difference_error_mse)
            rows.append({"screen": SCREENS[screen], "fold": fold, "held_out_line": HELD_OUT_LINES[screen][fold],
                         "no_change_error": no_change, "anchor_error": baseline, "noise_floor": noise,
                         "above_noise_signal": no_change - noise,
                         "anchor_share_percent": 100 * (no_change - baseline) / (no_change - noise)})
    table = pd.DataFrame(rows)
    share = table.groupby("screen").anchor_share_percent
    table["selected"] = ""
    table.loc[table.anchor_share_percent == share.transform("max"), "selected"] = "largest share"
    table.loc[table.anchor_share_percent == share.transform("min"), "selected"] = "smallest share"
    return table
