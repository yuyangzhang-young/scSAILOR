"""Two-phase retraining on Tahoe-100M folds 1-3: the retrained networks with the readers against the original networks
with the same readers.

The two-phase fits (``fits.tahoe('two-phase', model, seed, fold)``; the summary-statistics, DeepSets and attention
models at Mean, seeds 31002-31004) trained on the measured targets for the first half of their updates, then on targets
from which the control sampling difference and the source context response were subtracted. Each is read as its raw
held-out prediction ``P`` with its context-level correction replaced by the readers' offset, ``P - proj_ctx(P - A) +
offset``, ``A`` the fit's own saved anchor (the readers arm, ``readers.arm_predictions``). The original is the model's
ladder fit of the same seed (``data.ladder``) with the same readers and the fold's anchor (``readers.fold_arms``).

``comparison_table``: per fold, model and seed, both total errors (predictions clipped at zero) in % of the
design-aware baseline's total MSE on all held-out units, and their difference with origin resolution's 95% paired
interval over held-out lines (``comparison.bootstrap``); on the calibration-supported units
(``data.calibration_units``), the two-phase prediction with the readers and the calibrated original
(``calibration.case``, the +Var case of the same model and seed), in % of that case's design-aware baseline. The seed
mean is the mean of the seeds' values; its difference's interval resamples the held-out lines with the seeds' per-line
differences averaged within each line (``comparison.pool_seeds``).

``output_energy_table``: the energy of each network's drug differences, its within-context deviations from its anchor
(``output_energy``), for the two-phase network and the original.
"""
import numpy as np
import pandas as pd

from scsailor.resolution import comparison, grids

from ..origin_resolution import SEEDS
from . import arms, calibration, data, fits, readers
from .data import SEED_MEAN
from .storage import BUILT

__all__ = ["two_phase_fit", "output_energy", "seed_row", "seed_mean_row", "comparison_rows", "case_rows",
           "comparison_table", "output_energy_table"]


def two_phase_fit(fold, model, seed):
    """A two-phase fit's raw prediction and its own anchor on the fold's held-out units."""
    path, u = fits.tahoe('two-phase', model, seed, fold), data.fold_data(fold)['u']
    return grids.load(path, 'outer', u), grids.load(path, 'outer', u, member='anchor_raw')


def output_energy(P, A, D):
    """The energy (``D.energy``, x 1e3) of a network's drug differences: its deviation from the anchor ``A`` minus that
    deviation's context mean, unclipped."""
    X = P - A
    return 1e3 * D.energy(X - D.project_ctx(X))


def seed_row(two_phase, original, D, truth, base):
    """One seed's comparison on the units of ``D``: each prediction's total error (clipped at zero) in % of ``base``,
    and their difference with its 95% line interval. Returns the row and the per-line difference (line -> difference)
    that the seed mean pools."""
    e2, e1 = arms.per_line_total(two_phase, D, truth), arms.per_line_total(original, D, truth)
    per_line = dict(zip(D.line_names, e2 - e1))
    iv = comparison.bootstrap(per_line, base)
    return {'two_phase_readers_pct_of_design_aware': 100 * e2.sum() / base,
            'original_readers_pct_of_design_aware': 100 * e1.sum() / base,
            'delta_points': iv['estimate_pct'], 'delta_low': iv['interval_pct'][0],
            'delta_high': iv['interval_pct'][1]}, per_line


def seed_mean_row(rows, per_line, base):
    """The seed mean of the seeds' rows: the mean of each value, and the difference's interval from the seeds' per-line
    differences averaged within each line (``per_line``: one line -> difference map per seed)."""
    iv = comparison.bootstrap(*comparison.pool_seeds([(p, base) for p in per_line]))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return {**mean, 'delta_points': iv['estimate_pct'], 'delta_low': iv['interval_pct'][0],
            'delta_high': iv['interval_pct'][1]}


def comparison_rows(info, runs, fold, case):
    """The rows of one comparison on a fold (``runs``: seed -> (two-phase prediction with the readers, original with
    the readers) on the fold's held-out units; ``case``: the model's ``calibration.case``): per seed ``seed_row`` and
    on the calibration-supported units the calibrated original and the two-phase prediction with the readers, in % of
    that case's design-aware baseline; then the seed mean (``seed_mean_row``). Each row starts with ``info``."""
    d, D = data.fold_data(fold), data.design(fold)
    truth = d['u']['truth']
    base = arms.per_line_total(d['b'] + d['kappa'] * d['r_plate_unit'], D, truth).sum()
    keep, uk, Dk = data.calibration_units(fold)
    base_k = arms.per_line_total(case.baseline, Dk, uk['truth']).sum()
    rows, per_line = [], []
    for s, (Q, F) in runs.items():
        row, lines = seed_row(Q, F, D, truth, base)
        row['supported_calibrated_original_pct_of_design_aware'] = (
            100 * arms.per_line_total(case.calibrated[s], Dk, uk['truth']).sum() / base_k)
        row['supported_two_phase_readers_pct_of_design_aware'] = (
            100 * arms.per_line_total(Q[keep], Dk, uk['truth']).sum() / base_k)
        rows.append(row)
        per_line.append(lines)
    return ([{**info, 'seed': s, **r} for s, r in zip(runs, rows)]
            + [{**info, 'seed': SEED_MEAN, **seed_mean_row(rows, per_line, base)}])


def case_rows(fold, model):
    """One fold and built model's rows (``comparison_rows``): the two-phase fit and the ladder fit of each seed, both
    with the readers."""
    D, offsets = data.design(fold), readers.fit(fold)
    runs = {}
    for s in SEEDS:
        P, A = two_phase_fit(fold, model, s)
        runs[s] = (readers.arm_predictions(P, A, offsets, D)['readers'],
                   readers.fold_arms(fold, data.ladder(fold, model, s))['readers'])
    return comparison_rows({'fold': fold, 'model': data.MODELS[model]}, runs, fold, calibration.case(fold, model))


def comparison_table():
    """One row per fold, model and seed (and the seed mean): the two-phase prediction and the original, both with the
    readers, in % of the design-aware baseline's total MSE on all held-out units
    ('two_phase_readers_pct_of_design_aware', 'original_readers_pct_of_design_aware'), their difference with its 95%
    line interval ('delta_points', 'delta_low', 'delta_high'), and on the calibration-supported units the calibrated
    original and the two-phase prediction with the readers ('supported_*'). Written with ``to_csv(**data.WRITE)``."""
    return pd.DataFrame([r for fold in data.FOLDS for model in BUILT for r in case_rows(fold, model)])


def output_energy_table():
    """One row per fold, model and seed (and the seed mean): the output energy (``output_energy``) of the two-phase
    network ('two_phase_energy_e3', with its own anchor) and of the original ('original_energy_e3', with the fold's
    anchor), and their ratio; the seed mean's ratio is that of the seed means. Written with ``to_csv(**data.WRITE)``."""
    rows = []
    for fold in data.FOLDS:
        d, D = data.fold_data(fold), data.design(fold)
        for model in BUILT:
            seeds = [(output_energy(*two_phase_fit(fold, model, s), D),
                      output_energy(data.ladder(fold, model, s), d['A0'], D)) for s in SEEDS]
            for s, (e2, e1) in [*zip(SEEDS, seeds), (SEED_MEAN, tuple(np.mean(seeds, axis=0)))]:
                rows.append({'fold': fold, 'model': data.MODELS[model], 'seed': s, 'two_phase_energy_e3': float(e2),
                             'original_energy_e3': float(e1), 'ratio': float(e2 / e1)})
    return pd.DataFrame(rows)
