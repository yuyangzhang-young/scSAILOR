"""Origin resolution's +Var cases on Tahoe-100M folds 1-3 and their calibration toward the design-aware baseline.

The cases (``PLUS_VAR``): the summary-statistics, DeepSets and attention models and State, three training seeds each,
with their held-out predictions as origin resolution deposited them (``plus_var``). On a fold's calibration-supported
units (``data.calibration_units``), each seed's prediction is calibrated toward the design-aware baseline
``Th = b + kappa (A - b)`` (``net_cost.design_baseline``) by the two-part calibration (``menu.two_part``), with origin
resolution's amounts (``net-cost/AMOUNTS.json``): the case's ``kappa`` (lambda_r), the slope of truth - b on A - b on
the calibration lines truncated to 0-1, and the seed's amounts (lambda_1, lambda_2) of the context-level and the
perturbation-level correction, chosen on the calibration lines (``case``). ``Case.arms`` gives a seed's arms
(``readers.fold_arms``) with its calibrated prediction.
"""
import json
from functools import cache
from typing import NamedTuple

import numpy as np

from scsailor.resolution import menu, net_cost

from ..origin_resolution import ROOT, SEEDS
from ..origin_resolution.net_cost import read_side
from ..origin_resolution.slopes import load_case
from . import data, readers

__all__ = ["PLUS_VAR", "AMOUNTS", "plus_var", "amounts", "Case", "case"]

PLUS_VAR = {'summary': 'summary', 'deepsets': 'deepsets', 'attention_model': 'attention',
            'state': 'official_state'}   # code name -> origin resolution's name of the case
AMOUNTS = ROOT / 'net-cost/AMOUNTS.json'


def plus_var(fold, model):
    """Origin resolution's +Var case of ``model`` on the fold's held-out units: 'p' (seed -> prediction) and the
    design-aware baseline's input-set mean 'b' and plate-aware anchor 'A'."""
    return read_side('tahoe', load_case('tahoe', fold, PLUS_VAR[model], outer=True), 'held-out', cell=None)


@cache
def amounts():
    """Origin resolution's amounts, per case ('tahoe|<fold>|<name>'): 'kappa' and per seed 'original_design'
    (the amounts lambda_1, lambda_2 of the context-level and the perturbation-level correction)."""
    return json.loads(AMOUNTS.read_text())


class Case(NamedTuple):
    fold: str
    predictions: dict   # seed -> the prediction on every held-out unit
    baseline: np.ndarray   # the design-aware baseline on the calibration-supported units
    calibrated: dict   # seed -> the calibrated prediction on the calibration-supported units

    def arms(self, seed):
        """The seed's arms on the calibration-supported units, 'calibrated' and 'calibrated + readers' among them."""
        keep, _, _ = data.calibration_units(self.fold)
        return readers.fold_arms(self.fold, self.predictions[seed][keep], self.calibrated[seed])


@cache
def case(fold, model):
    """The +Var case of ``model`` on the fold, with each seed's calibrated prediction (``Case``)."""
    held = plus_var(fold, model)
    keep, u, _ = data.calibration_units(fold)
    saved = amounts()[f'tahoe|{fold}|{PLUS_VAR[model]}']
    Th = net_cost.design_baseline(held, saved['kappa'])[keep]
    calibrated = {s: menu.two_part(Th, held['p'][s][keep], *saved['seeds'][str(s)]['original_design'], u)
                  for s in SEEDS}
    return Case(fold, held['p'], Th, calibrated)
