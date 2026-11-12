"""The light selection and the descriptive reading, per case.

A case is model x fold x half x target; its cell table is ``rules.cells`` restricted to the case (map runs only,
``case == 'parent'``). Outcomes of ``light_select``:
  'solved at the start'                      both map seeds' original models reach LEVEL before training;
  'diagnostic intervention'                  no candidate at the primary seed: the fixed intervention is handed on;
  'unconfirmed'                              a checked cell's second-seed run has not finished: nothing is handed on;
  'diagnostic intervention (check failed)'   every checked cell fails at the second seed (a diverged run fails);
  'qualified repair'                         the confirmed cell with the best two-seed mean Q is handed on.
"""
import numpy as np
import pandas as pd

from .menus import DIAGNOSTIC, PRIMARY_SEED, WINDOW, in_menu
from .rules import FRESH, LEVEL, MAP_SEEDS, REFERENCE_ARM, solved_at_start

CASE = ['dataset', 'model', 'fold', 'half', 'target']
TARGET_NAMES = {'1': 'selective mean', '0': 'common mean', '5': 'common variance'}
OTHER_SEED = {MAP_SEEDS[0]: MAP_SEEDS[1], MAP_SEEDS[1]: MAP_SEEDS[0]}


def name(cut, arm, rate):
    return f'{cut}/{arm}/x{float(rate):g}'


def cell_q(menu, cell, seed):
    """(Q, margins) of one cell at one seed; (nan, None) when the run has not finished; (nan, False) when it diverged."""
    cut, arm, rate = cell.split('/')
    r = menu[(menu.cut == cut) & (menu.arm == arm) & (menu.rate == float(rate[1:])) & (menu.seed == seed)]
    return (np.nan, None) if r.empty else (float(r.Q.iloc[0]), bool(r.margins.iloc[0]))


def light_part(menu, model):
    return menu[np.array([in_menu(model, c, a, r) for c, a, r in zip(menu.cut, menu.arm, menu.rate)], dtype=bool)]


def light_select(menu, model, solved, primary=PRIMARY_SEED, window=WINDOW):
    """The light selection on one case. Candidates are cells behind a cut that reach LEVEL within the margins at the
    primary seed; those within ``window`` of the best are checked at the other map seed, and the confirmed one with the
    best two-seed mean Q is handed on (ties: the lower rate, then cut and arm). Returns (kind, cell, checked cells)."""
    if solved:
        return 'solved at the start', '', []
    light = light_part(menu, model)
    behind = light[(light.cut != 'input') & ~light.arm.isin([FRESH, REFERENCE_ARM])]
    cand = behind[(behind.seed == primary) & (behind.Q >= LEVEL) & behind.margins]
    if cand.empty:
        return 'diagnostic intervention', DIAGNOSTIC[model], []
    check = cand[cand.Q >= cand.Q.max() - window - 1e-12]
    confirmed, missing, checked = [], 0, []
    for r in check.itertuples():
        cell = name(r.cut, r.arm, r.rate)
        checked.append(cell)
        q_o, ok_o = cell_q(behind, cell, OTHER_SEED[primary])
        if ok_o is None:
            missing += 1
        elif q_o >= LEVEL and ok_o:
            confirmed.append((-(r.Q + q_o) / 2, float(r.rate), r.cut, r.arm, cell))
    if missing:
        return 'unconfirmed', '', checked
    if not confirmed:
        return 'diagnostic intervention (check failed)', DIAGNOSTIC[model], checked
    return 'qualified repair', sorted(confirmed)[0][-1], checked


def shared_fresh(cells):
    """The from-scratch network does not depend on the original model, so it runs once, from the synthetic-task model's
    map, and serves both halves: its rows per (dataset, model, fold, target)."""
    f = cells[(cells.case == 'parent') & (cells.half == 'synthetic') & (cells.cut == 'input') & (cells.arm == FRESH)]
    return {key: rows for key, rows in f.groupby(['dataset', 'model', 'fold', 'target'])}


def behaviour(menu, model, solved, kind, cell, seeds, restrict, fresh=None):
    """The descriptive reading of one case. ``seeds`` are the seeds read for the parts (the light version: its primary
    seed); the references are read in both map seeds: the whole network in the case's own half and, unless
    ``restrict``, the from-scratch network (``fresh``, from shared_fresh) from the synthetic half. ``restrict`` reads
    the light menu only, which has no from-scratch network. An unconfirmed case is read as waiting for its check, not as
    a negative. No case is forced into a class: 'not recovered within the menu' leaves slow and not learnable apart,
    because the light version has no training-longer runs."""
    if solved:
        return 'solved at the start'
    if kind == 'unconfirmed':
        return 'waiting for the check'
    rows = menu if fresh is None else pd.concat([menu[menu.arm != FRESH], fresh])
    rows = light_part(rows, model) if restrict else rows
    part = rows[rows.seed.isin(seeds)]
    ref = rows[rows.seed.isin(MAP_SEEDS)]
    if kind == 'qualified repair':
        cut, arm, _ = cell.split('/')
        fine = part[(part.cut == cut) & (part.arm == 'a')]
        if arm != 'a' and not fine.empty and fine.Q.max() < LEVEL:
            return 'a reset recovers where fine-tuning at the same cut does not'
        return 'a part recovers'
    whole = ref[(ref.cut == 'input') & ref.arm.isin(['a', FRESH])]
    if whole.empty:
        return 'reference not in archive'
    if (whole.Q >= LEVEL).any():
        return 'no part recovers; the whole or fresh network does'
    return 'not recovered within the menu (time axis not read here)'


def light_cases(runs, cells, primary=PRIMARY_SEED, window=WINDOW):
    """The light reading of every case in one or more maps: ``runs`` from maps.read_map, ``cells`` = rules.cells(runs)."""
    start = solved_at_start(runs)
    fresh = shared_fresh(cells)
    rows = []
    for key, menu in cells[cells.case == 'parent'].groupby(CASE):
        if key not in start:
            continue
        kind, cell, checked = light_select(menu, key[1], start[key], primary, window)
        rows.append(dict(zip(CASE, key)) | {
            'target_name': TARGET_NAMES[key[4]], 'primary_seed': primary, 'window': window, 'kind': kind, 'cell': cell,
            'checked': ';'.join(checked),
            'behaviour': behaviour(menu, key[1], start[key], kind, cell, (primary,), restrict=True,
                                   fresh=fresh.get((key[0], key[1], key[2], key[4])))})
    return pd.DataFrame(rows)


def handoff(cases):
    """Per dataset x model x fold: the cells handed on, merged and deduplicated. Unconfirmed cases are listed apart,
    because they hand on nothing until their check runs finish."""
    carried = cases[cases.kind.isin(['qualified repair', 'diagnostic intervention', 'diagnostic intervention (check failed)'])]
    out = carried.groupby(['dataset', 'model', 'fold']).cell.agg(lambda s: ';'.join(sorted(set(s)))).rename('cells')
    waiting = cases[cases.kind == 'unconfirmed'].groupby(['dataset', 'model', 'fold']).size().rename('unconfirmed_cases')
    out = pd.concat([out, waiting], axis=1).fillna({'cells': '', 'unconfirmed_cases': 0})
    return out.astype({'unconfirmed_cases': int}).reset_index()
