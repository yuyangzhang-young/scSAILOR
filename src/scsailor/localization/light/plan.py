"""Run lists for a light case.

The rows carry the identity fields of a run (name, stage, model, half, fold, seed, case, cut, arm, rate multiple); the
code that submits them adds the original model, the inputs and the output folder. State's rows carry its original
model's own 5,000-update schedule and a read-out every 500 updates and no case field; CPA's and PrePR-CT's carry the
fold in the name. Each fit that origin resolution needs is listed with the status 'to register: reused or new', to be
marked as an existing fit or a new one before it is run.
"""
import pandas as pd

from . import menus
from .assay import CHECK_PARENT, LIGHT_LEVELS, LIGHT_SEED
from .selection import OTHER_SEED

OR_SEEDS = (31002, 31003, 31004)   # origin resolution's three seeds; 31004 is used only there


STATE_SCHEDULE, STATE_EVAL_EVERY = 5000, 500


def row(model, half, fold, seed, cut, arm, rate, stage='map'):
    rate = float(rate)
    if model == 'state':
        return dict(name=f'{stage}-state-{half}-{seed}-{cut}-{arm}-x{rate:g}', stage=stage, model=model, half=half,
                    fold=fold, seed=seed, cut=cut, arm=arm, rate_multiple=rate, schedule=STATE_SCHEDULE,
                    eval_every=STATE_EVAL_EVERY)
    tag = f'{half}-{fold}' if model in ('cpa', 'prepr') else half
    return dict(name=f'{stage}-{model}-{tag}-{seed}-{cut}-{arm}-x{rate:g}', stage=stage, model=model, half=half,
                fold=fold, seed=seed, case='parent', cut=cut, arm=arm, rate_multiple=rate)


def map_rows(model, fold):
    """The light map of one model and fold: the primary seed's menu and the references at both map seeds."""
    return [row(model, half, fold, seed, cut, arm, rate) for half, seed, cut, arm, rate in menus.runs(model)]


def check_rows(cases, runs):
    """The second-seed runs the check still needs, deduplicated over targets: the checked cells of ``cases``
    (selection.light_cases) whose run at the other map seed has not finished (``runs`` from maps.read_map)."""
    maps = runs[runs.case == 'parent']   # map runs only: a planted-fault run of the same cell is not a check
    done = set(zip(maps.dataset, maps.model, maps.fold, maps.half, maps.seed, maps.cut, maps.arm, maps.rate))
    out = {}
    for r in cases.itertuples():
        seed = OTHER_SEED[r.primary_seed]
        for cell in filter(None, r.checked.split(';')):
            cut, arm, rate = cell.split('/')
            key = (r.dataset, r.model, r.fold, r.half, seed, cut, arm, float(rate[1:]))
            if key not in done:
                out.setdefault(key, row(r.model, r.half, r.fold, seed, cut, arm, key[7]) | {'dataset': r.dataset})
    return list(out.values())


def assay_fits(model, fold):
    """The light assay of one model and fold: the three light levels at seed 31002 with the full replay, and the
    training-only +Var fit at seed 31003 (with its start reading on the validation panel)."""
    fits = [dict(model=model, fold=fold, level=level, seed=LIGHT_SEED, replay='all 26 configurations') for level in LIGHT_LEVELS]
    return fits + [dict(model=model, fold=fold, level=CHECK_PARENT[0], seed=CHECK_PARENT[1], replay='none (training only)')]


def origin_needs(dataset, model, fold, cells, runs):
    """The fits origin resolution needs for one screen, model and fold, given its handoff ``cells`` (the cells handed
    on, merged and deduplicated; selection.handoff) and the finished map runs ``runs`` (maps.read_map).

    Natural-task chain: three nested original models; per cell, three seeds on the full and the nested original model;
    continued training at equal dose, three seeds on both chains, shared by cells whose dose and read-out match.
    Cards (origin resolution's readings of one change on a fold; three seeds, as in the full version): each cell's map
    run in both halves at every seed. A light map has seed
    31003 only where the check ran, so missing runs at any of the three seeds are listed with their run names; and the
    seed-31004 synthetic +Var original model (training only, with its start reading). The full natural original models
    at the three seeds and the seed-31003 synthetic +Var model (the light assay's check model) are taken to exist."""
    need = [dict(item='nested parent', cell='', chain='nested', seed=s, name='') for s in OR_SEEDS]
    for cell in cells:
        need += [dict(item='natural-task run of the cell', cell=cell, chain=chain, seed=s, name='')
                 for chain in ('full', 'nested') for s in OR_SEEDS]
    if cells:
        need += [dict(item='ordinary continuation at equal dose', cell='', chain=chain, seed=s, name='')
                 for chain in ('full', 'nested') for s in OR_SEEDS]
        maps = runs[(runs.dataset == dataset) & (runs.model == model) & (runs.fold == fold) & (runs.case == 'parent')]
        have = set(zip(maps.half, maps.seed, maps.cut, maps.arm, maps.rate))
        for cell in cells:
            cut, arm, rate = cell.split('/')
            need += [dict(item='map run for the cards', cell=cell, chain=half, seed=s,
                          name=row(model, half, fold, s, cut, arm, float(rate[1:]))['name'])
                     for half in ('synthetic', 'natural') for s in OR_SEEDS if (half, s, cut, arm, float(rate[1:])) not in have]
        need.append(dict(item='synthetic +Var parent for the cards (training only)', cell='', chain='synthetic',
                         seed=OR_SEEDS[2], name=''))
    return pd.DataFrame(need).assign(dataset=dataset, model=model, fold=fold, status='to register: reused or new')
