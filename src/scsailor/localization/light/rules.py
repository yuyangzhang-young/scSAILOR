"""The full version's reading rules on tables of runs, read on the validation panel.

The recovery level, the margin values and the arm names are imported from ``scsailor.localization.readings``.
``cells`` computes the margins of ``readings.margins`` on a whole table at once, and ``carried`` makes the choice of
``readings.carried`` for one case. The exit for a case the original model already recovers is read here on the
validation panel (``readings.carried`` reads it on the held-out panel).

A cell table has one row per run and target: the run's identity (model, fold, half, seed, case, cut, arm, rate),
``Q`` (validation recovery of the target) and ``margins`` (whether the validation margins hold).
"""
import numpy as np
import pandas as pd

from ..readings import ERROR_FACTOR, FRESH, LEARNED_MARGIN, RECOVERY_LEVEL as LEVEL, REFERENCE_ARM

MAP_SEEDS = (31002, 31003)
TARGETS = ('1', '0', '5')     # selective shift, shared shift, shared spread
JUNCTION = {'summary': 'projections', 'deepsets': 'phi', 'attention_model': 'head', 'state': 'head',
            'cpa': 'decoder', 'prepr': 'prediction'}


def cells(runs):
    """Per run and target: Q and whether the validation margins hold, as ``readings.margins`` gives them run by run."""
    out = []
    for target in TARGETS:
        t = runs.copy()
        kept = (t['validation_raw'] <= ERROR_FACTOR * t['parent_validation_raw']).to_numpy()
        synthetic = (t['half'] == 'synthetic').to_numpy()
        for bit in (b for b in TARGETS if b != target):
            before = t[f'parent_validation_Q{bit}'].to_numpy()
            need = synthetic & ((bit == '0') | (before >= .5))
            kept &= ~need | (t[f'validation_Q{bit}'].to_numpy() >= before - LEARNED_MARGIN)
        t['target'] = target
        t['Q'] = t[f'validation_Q{target}']
        t['margins'] = kept
        out.append(t)
    return pd.concat(out, ignore_index=True)


def solved_at_start(runs):
    """(dataset, model, fold, half, target) -> True when both map seeds' original models reach LEVEL on the validation
    panel before any training."""
    parents = runs[(runs['case'] == 'parent') & runs['seed'].isin(MAP_SEEDS)]
    parents = parents.drop_duplicates(['dataset', 'model', 'fold', 'half', 'seed'])
    out = {}
    for key, rows in parents.groupby(['dataset', 'model', 'fold', 'half']):
        for target in TARGETS:
            out[key + (target,)] = bool((rows[f'parent_validation_Q{target}'] >= LEVEL).all()) and set(rows['seed']) == set(MAP_SEEDS)
    return out


def carried(menu, model, solved, seeds=MAP_SEEDS, junction=None):
    """The full version's choice on one case: among cells behind a cut that reach LEVEL within the margins in every
    seed of ``seeds``, the one with the best mean Q; otherwise the fixed intervention (junction, a, x1). ``junction``
    names the junction cut of an architecture outside JUNCTION."""
    if solved:
        return 'solved at the start', ''
    menu = menu[(menu['cut'] != 'input') & ~menu['arm'].isin([FRESH, REFERENCE_ARM]) & menu['seed'].isin(seeds)]
    best, best_q = None, -np.inf
    for (cut, arm, rate), rows in menu.groupby(['cut', 'arm', 'rate'], sort=False):
        if set(rows['seed']) >= set(seeds) and ((rows['Q'] >= LEVEL) & rows['margins']).all():
            q = rows['Q'].mean()
            if q > best_q:
                best, best_q = f'{cut}/{arm}/x{rate:g}', q
    if best is None:
        return 'diagnostic intervention', f'{junction or JUNCTION[model]}/a/x1'
    return 'qualified repair', best
