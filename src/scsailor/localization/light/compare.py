"""The light version against the full one, on maps that hold both menus.

The full reading is ``rules.carried`` on the full menu with both map seeds; the light reading is
``selection.light_select`` on the light menu, at either primary seed and with the fixed window or the top cell only.
All of it reads the validation panel, except ``heldout_cases``: the held-out recovery of the cells already chosen.
The full reading's references include the from-scratch network and the light reading's do not, so the two
descriptive readings differ where only the from-scratch network recovers.
"""
import numpy as np
import pandas as pd

from .maps import heldout_q
from .menus import WINDOW, in_menu
from .rules import MAP_SEEDS, carried, solved_at_start
from .selection import CASE, OTHER_SEED, TARGET_NAMES, behaviour, cell_q, light_select, shared_fresh

WINDOWS = (WINDOW, 0.0)   # the fixed window, and top only as the sensitivity
GROUP = ['dataset', 'model', 'primary_seed', 'window']


def site(cell):
    return '/'.join(cell.split('/')[:2]) if cell else ''


def cases(runs, cells):
    """CASES (one row per case x primary seed x window) and CHECK_RUNS (the checked cells, deduplicated over targets)."""
    start = solved_at_start(runs)
    fresh = shared_fresh(cells)
    rows, checks = [], []
    for key, menu in cells[cells.case == 'parent'].groupby(CASE):
        if key not in start:
            continue
        model = key[1]
        shared = fresh.get((key[0], key[1], key[2], key[4]))
        full = carried(menu, model, start[key])
        full_q = [cell_q(menu, full[1], s)[0] for s in MAP_SEEDS] if full[0] == 'qualified repair' else [np.nan, np.nan]
        for s in OTHER_SEED:
            for window in WINDOWS:
                kind, cell, checked = light_select(menu, model, start[key], s, window)
                light_q = [cell_q(menu, cell, x)[0] for x in MAP_SEEDS] if kind == 'qualified repair' else [np.nan, np.nan]
                rows.append(dict(zip(CASE, key)) | {
                    'target_name': TARGET_NAMES[key[4]], 'primary_seed': s, 'window': window,
                    'full_kind': full[0], 'full_cell': full[1], 'light_kind': kind, 'light_cell': cell,
                    'full_qualified': full[0] == 'qualified repair', 'light_qualified': kind == 'qualified repair',
                    'same_cell': kind == full[0] and cell == full[1],
                    'same_site': kind == full[0] and site(cell) == site(full[1]),
                    'full_mean_Q': np.nanmean(full_q) if full[0] == 'qualified repair' else np.nan,
                    'light_mean_Q': np.nanmean(light_q) if kind == 'qualified repair' else np.nan,
                    'checked': len(checked),
                    'behaviour_full': behaviour(menu, model, start[key], full[0], full[1], MAP_SEEDS, restrict=False, fresh=shared),
                    'behaviour_light': behaviour(menu, model, start[key], kind, cell, (s,), restrict=True, fresh=shared)})
                checks += [dict(dataset=key[0], model=model, fold=key[2], half=key[3], primary_seed=s, window=window, cell=c)
                           for c in checked]
    out = pd.DataFrame(rows)
    out['Q_difference_light_minus_full'] = out.light_mean_Q - out.full_mean_Q
    return out, pd.DataFrame(checks).drop_duplicates()


def summary(out, checks):
    summ = out.groupby(GROUP).apply(lambda g: pd.Series({
        'cases': len(g), 'full_qualified': int(g.full_qualified.sum()),
        'light_still_qualified': int((g.full_qualified & g.light_qualified).sum()),
        'same_cell': int((g.full_qualified & g.same_cell).sum()), 'same_site': int((g.full_qualified & g.same_site).sum()),
        'light_qualified_where_full_not': int((~g.full_qualified & g.light_qualified).sum()),
        'unconfirmed': int((g.light_kind == 'unconfirmed').sum()),
        'check_failed': int((g.light_kind == 'diagnostic intervention (check failed)').sum()),
        'max_abs_Q_difference': (float(np.nanmax(np.abs(g.Q_difference_light_minus_full)))
                                 if g.Q_difference_light_minus_full.notna().any() else np.nan),
        'behaviour_same': int((g.behaviour_full == g.behaviour_light).sum())}), include_groups=False).reset_index()
    n = checks.groupby(GROUP).size().rename('check_runs').reset_index()
    return summ.merge(n, on=GROUP, how='left').fillna({'check_runs': 0})


def behaviour_table(out):
    fixed = out[(out.primary_seed == MAP_SEEDS[0]) & (out.window == WINDOW)]
    return fixed[CASE + ['target_name', 'behaviour_full', 'behaviour_light']]


def map_runs(runs):
    """Per dataset x model x fold: the light menu's primary-seed runs present in a full map, and the full map's runs."""
    rr = runs[(runs.case == 'parent') & (runs.seed == MAP_SEEDS[0])]
    rr = rr.drop_duplicates(['dataset', 'model', 'fold', 'half', 'cut', 'arm', 'rate'])
    rr = rr[np.array([in_menu(m, c, a, r) for m, c, a, r in zip(rr.model, rr.cut, rr.arm, rr.rate)], dtype=bool)]
    full = runs[runs.case == 'parent'].groupby(['dataset', 'model', 'fold']).size().rename('full_map_runs_as_run')
    light = rr.groupby(['dataset', 'model', 'fold']).size().rename('light_map_runs_in_archive')
    return pd.concat([light, full], axis=1).reset_index()


def heldout_cases(out, runs):
    """HELDOUT_CASES: held-out recovery of each case's full and light cell, per map seed and as their mean."""
    index = {}
    for r in runs[(runs.case == 'parent') & (runs.status == 'complete')].itertuples():
        index.setdefault((r.dataset, r.model, r.fold, r.half, r.cut, r.arm, r.rate, r.seed), r.run)

    def q(r, cell, seed):
        if not cell:
            return np.nan
        cut, arm, rate = cell.split('/')
        run = index.get((r.dataset, r.model, r.fold, r.half, cut, arm, float(rate[1:]), seed))
        return np.nan if run is None else heldout_q(run, r.target)

    rows = []
    for r in out.itertuples():
        row = {k: getattr(r, k) for k in CASE + ['target_name', 'primary_seed', 'window', 'full_kind', 'full_cell',
                                                 'light_kind', 'light_cell', 'same_cell']}
        for who, cell in (('full', r.full_cell), ('light', r.light_cell)):
            values = [q(r, cell, s) for s in MAP_SEEDS]
            row[f'{who}_heldout_Q_{MAP_SEEDS[0]}'], row[f'{who}_heldout_Q_{MAP_SEEDS[1]}'] = values
            row[f'{who}_heldout_Q_mean'] = np.nanmean(values) if not all(np.isnan(values)) else np.nan
        row['heldout_Q_difference_light_minus_full'] = row['light_heldout_Q_mean'] - row['full_heldout_Q_mean']
        rows.append(row)
    return pd.DataFrame(rows)


def heldout_summary(h):
    q = h[h.full_kind == 'qualified repair']
    return q.groupby(GROUP).agg(
        qualified_cases=('same_cell', 'size'), same_cell=('same_cell', 'sum'),
        full_heldout_Q_mean=('full_heldout_Q_mean', 'mean'), light_heldout_Q_mean=('light_heldout_Q_mean', 'mean'),
        max_abs_difference=('heldout_Q_difference_light_minus_full', lambda s: s.abs().max())).reset_index()


def natural_effect(scores, model, full_lines):
    """Natural-task effect of each scored cell on the real task, from its score table: the change minus the original
    model, in % of the anchor's total MSE on the same lines, held-out and training. ``full_lines`` is the full original
    model's line count; original models with fewer lines are the nested ones."""
    s = scores[scores.model == model]
    rows = []
    for (name, cut, arm, rate, seed, n_lines), g in s.groupby(['name', 'cut', 'arm', 'rate_multiple', 'seed', 'n_lines']):
        p, r = g[g.stage == 'parent'].iloc[0], g[g.stage == 'repair'].iloc[0]
        row = dict(cell=f'{cut}/{arm}/x{rate:g}', seed=seed, parent='full' if n_lines == full_lines else 'nested', name=name)
        for side in ('heldout', 'train'):
            base = p[f'{side}_anchor_total_mse']
            for part, key in (('total', 'total_mse'), ('common', 'common_error_mse'), ('contrast', 'difference_error_mse')):
                row[f'{side}_{part}_change_points'] = 100 * (r[f'{side}_model_{key}'] - p[f'{side}_model_{key}']) / base
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['cell', 'parent', 'seed'])
