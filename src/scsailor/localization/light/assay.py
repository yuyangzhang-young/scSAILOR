"""The light synthetic-answer assay panel and its readings against the full assay.

Light panel: the levels Mean (L1), +Var (L2) and Cells (L4); all five cues; one seed (31002); all 26 replay
configurations at those levels, with the assay itself unchanged. One more fit, at +Var with seed 31003, is trained
only: it is the synthetic-task model of the localization check and needs its start reading on the validation panel.
Left out of the full assay: depth only (L0), +Dist (L3), +DepCp (L3d) and seeds 31003 and 31004 of the read-out.

Planted state (``module``) m of the 25: cue m // 5 (shift, spread, skew, depth trend, covariation), response m % 5
(shared, selective, weak shared, weak selective, null; the column ``role``). A cue is visible from its first level on:
shift L1, spread L2, skew L3, depth trend L3d, covariation L4. The entries read are the shared and selective tests
where the cue is visible; the null test is read as leakage, A_null over the mean E of the cue's shared and selective
tests.

A module table has the columns of COLUMNS, one row per fit and module: ``scsailor.assay.capability``'s rows of one
support, with ``module_id`` as ``module`` and the fold added.
"""
import numpy as np
import pandas as pd

LEVELS = ('L0', 'L1', 'L2', 'L3', 'L3d', 'L4')
LIGHT_LEVELS = ('L1', 'L2', 'L4')
LIGHT_SEED = 31002
CHECK_PARENT = ('L2', 31003)
FULL_SEEDS = (31002, 31003, 31004)
CUES = ('shift', 'spread', 'skew', 'depth trend', 'covariation')
ROLES = ('shared', 'selective', 'weak shared', 'weak selective', 'null')
FIRST = {'shift': 'L1', 'spread': 'L2', 'skew': 'L3', 'depth trend': 'L3d', 'covariation': 'L4'}
COLUMNS = ['model', 'fold', 'level', 'seed', 'module', 'E', 'A', 'Q']


def annotate(t):
    """Keep the full assay's seeds and levels; add cue, role and visibility."""
    t = t[t.seed.isin(FULL_SEEDS) & t.level.isin(LEVELS)].copy()
    t['module'] = t.module.astype(int)
    t['cue'] = t.module.map(lambda m: CUES[m // 5])
    t['role'] = t.module.map(lambda m: ROLES[m % 5])
    t['visible'] = [LEVELS.index(l) >= LEVELS.index(FIRST[c]) for l, c in zip(t.level, t.cue)]
    return t


def light_against_full(t):
    """The light panel (seed 31002 at the light levels) against the full assay, on annotated module rows.

    Returns ENTRIES (each strong visible entry at a light level, with seed 31002 minus the three-seed and the other two
    seeds' mean), SUMMARY (per model), LOST (per model x fold x cue x role: the full assay's best level against the light
    panel's) and NULL (leakage per seed at the light levels)."""
    seeds = list(FULL_SEEDS)
    wide = t.pivot_table(index=['model', 'fold', 'level', 'module', 'cue', 'role', 'visible'], columns='seed', values='Q').reset_index()
    wide['mean3'] = wide[seeds].mean(axis=1)
    wide['other2'] = wide[seeds[1:]].mean(axis=1)
    strong = wide[wide.role.isin(['shared', 'selective']) & wide.visible]
    entries = strong[strong.level.isin(LIGHT_LEVELS)].copy()
    entries['diff_vs_three'] = entries[LIGHT_SEED] - entries.mean3
    entries['diff_vs_other_two'] = entries[LIGHT_SEED] - entries.other2
    summary = entries.groupby('model').agg(
        entries=('module', 'size'), mean_abs_diff_vs_three=('diff_vs_three', lambda s: s.abs().mean()),
        max_abs_diff_vs_three=('diff_vs_three', lambda s: s.abs().max()),
        mean_abs_diff_vs_other_two=('diff_vs_other_two', lambda s: s.abs().mean())).reset_index()
    lost = []
    for key, g in strong.groupby(['model', 'fold', 'cue', 'role']):
        best = g.loc[g.mean3.idxmax()]
        lg = g[g.level.isin(LIGHT_LEVELS)]
        onset = g[g.level == FIRST[key[2]]]
        lost.append(dict(zip(['model', 'fold', 'cue', 'role'], key), full_max_Q=best.mean3, full_max_level=best.level,
                         light_max_Q=lg[LIGHT_SEED].max() if not lg.empty else np.nan,
                         light_max_level=lg.loc[lg[LIGHT_SEED].idxmax()].level if not lg.empty else '',
                         full_Q_at_onset=onset.mean3.iloc[0] if not onset.empty else np.nan))
    lost = pd.DataFrame(lost)
    lost['max_difference_light_minus_full'] = lost.light_max_Q - lost.full_max_Q
    ae = t.pivot_table(index=['model', 'fold', 'level', 'cue', 'seed'], columns='role', values=['A', 'E']).reset_index()
    ae.columns = ['_'.join([c for c in col if c]) for col in ae.columns]
    ae['leakage'] = ae['A_null'] / ((ae['E_shared'] + ae['E_selective']) / 2)
    null = ae[ae.level.isin(LIGHT_LEVELS)].pivot_table(index=['model', 'fold', 'level', 'cue'], columns='seed', values='leakage').reset_index()
    null['mean3'] = null[seeds].mean(axis=1)
    return entries, summary, lost, null
