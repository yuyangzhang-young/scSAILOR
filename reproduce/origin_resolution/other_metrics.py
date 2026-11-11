"""The matched comparisons' predictions on metrics other than squared error (``metrics``), from saved grids.

Stages, writing into ``--out``; the saved tables were read with one thread per process (``OMP_NUM_THREADS=1``):

- ``tahoe``: ``METRIC_READINGS_TAHOE.csv``, every Tahoe-100M change of the matched comparisons, per seed, prediction and
  metric, per line then over lines: no change, the baseline, the design-aware baseline, the original model as trained
  and calibrated toward each baseline, the second training run (the seed averaged with the next one) as trained and
  calibrated, the change at full strength, kept, and kept and calibrated; from ``AMOUNTS.json`` (``net_cost --stage
  calibration``) and ``PLATE_FREE_SEED.csv`` (``net_cost --stage plate-free-seed``) in ``--out``;
- ``genetic``: ``METRIC_READINGS_GENETIC.csv``, the same on Replogle-Nadig and Jiang et al., the target gene excluded;
- ``other-screens``: ``METRIC_READINGS_OTHER_SCREENS.csv``, the original, the change at full strength and the second
  training run on the changes the matched comparisons do not cover (GSE306429, Parse PBMC, the attention model and
  State on the genetic screens), from the cost readings' grids and ``RESTORED_LIST.csv`` in ``--out``, rows in
  that list's order;
- ``by-change``: ``BY_CHANGE.csv``, every change at full strength and the second training run against the original on
  every metric, seed means, signed so that positive is better (squared error in points of the original's total).

    OMP_NUM_THREADS=1 python -m reproduce.origin_resolution.other_metrics --stage tahoe --out DIR --workers 5
"""
import argparse
import ast
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scsailor.resolution import grids, menu, metrics, net_cost
from scsailor.resolution.stages import bag_of
from scsailor.resolution.scoring import identities
from . import ROOT, SEEDS, runs, tahoe, write_csv
from .cost_readings import units as cost_units
from .net_cost import all_changes, key, read_side
from .slopes import BINDINGS, load_case

__all__ = ["references", "predictions", "tahoe_stage", "genetic_stage", "other_screens_stage", "by_change_stage", "main"]

TRUTH = ROOT / 'genetic-truth-statistics/truth'   # TRUTH_STATS_<screen>_<fold>.npz: each unit's target gene
COST_ROOTS = {'official_state': ROOT / 'state-other-screens', 'attention': ROOT / 'attention-model-other-screens',
              'attention_jiang': ROOT / 'attention-model-jiang', 'built': ROOT / 'built-models-other-screens'}
RUN_MODEL = {'summary': 'summary', 'deepsets': 'deepsets', 'attention': 'state', 'official_state': 'official_state'}
METRICS = ('mse', 'weighted_r2', 'weighted_r2_contrast', 'pearson_delta', 'spearman_delta', 'direction_top50', 'cosine_pds',
           'systema_pearson_delta', 'systema_pearson_delta_observed')


def references(sd, case, bank, r_bar, target=None):
    """The reference of ``metrics`` for the side's units; ``target`` the target gene per unit on genetic screens."""
    u = sd['u']
    ybar = metrics.context_mean(u['truth'], u['weight'], u['context'])
    d = u['truth'] - ybar
    lines = identities(u)
    names = np.unique(lines)
    C = bag_of(case['bags'][bank], u)[1]
    return {'C': C, 'R': C + r_bar, 'ybar': ybar, 'd': d, 'gene_weight': metrics.effect_weights(d, target),
            'index': np.searchsorted(names, lines), 'n': len(names), 'target': target}


def predictions(sd, s, model_amounts, kept, second_baseline):
    u, a, b, p, two = sd['u'], sd['a'], sd['b'], sd['p'][s], menu.two_part
    T = net_cost.design_baseline(sd, model_amounts['kappa'])
    m, k = model_amounts['seeds'][str(s)], kept[str(s)]
    seed = (p + sd['p'][net_cost.partner(sd, s)]) / 2
    X = p + k['alpha'] * (sd['R'][s] - p)
    return {'no change': b, 'baseline': a, 'design baseline': T, 'original': p,
            'original, baseline': two(a, p, *m['original_field'], u), 'original, design': two(T, p, *m['original_design'], u),
            'seed': seed, 'seed, baseline': two(a, seed, *second_baseline, u), 'seed, design': two(T, seed, *m['second_design'], u),
            'change': sd['R'][s], 'kept': X, 'kept, baseline': two(a, X, *k['action_field'], u), 'kept, design': two(T, X, *k['action_design'], u)}


def mean_response(constants, train_targets):
    """The training perturbations' mean input response."""
    with np.load(constants) as z:
        r = z['r_input'].astype(np.float64)
    with np.load(train_targets) as z:
        return r[np.unique(z['p_code'])].mean(0)


def targets(screen, fold, u):
    """The target gene's column per unit, -1 for none (the weighted-R² truth tables)."""
    if screen not in ('jiang', 'replogle'):
        return np.full(len(u['truth']), -1)
    with np.load(TRUTH / f'TRUTH_STATS_{screen}_{fold}.npz') as z:
        look = {(str(k), int(p)): int(t) for k, p, t in zip(z['key'], z['p_code'], z['target'])}
    return np.asarray([look[(str(c), int(p))] for c, p in zip(u['context'], u['p_code'])])


def card_lines(sd, ref, saved, second, values):
    return {s: {name: metrics.per_line(values(x, sd['u'], ref), sd['u'], ref)
                for name, x in predictions(sd, s, saved['model'], saved['kept'], second[s]).items()} for s in SEEDS}


def read_tahoe(job):
    c, saved, second = job
    case = load_case(c['screen'], c['fold'], c['model'], True, c['cell'])
    sd = read_side(c['screen'], case, 'held-out', c['cell'])
    f = tahoe.files(c['fold'])
    ref = references(sd, case, 'heldout', mean_response(f['constants'], f['train_targets']))
    print(key(c), flush=True)
    return key(c), card_lines(sd, ref, saved, {s: second[(c['fold'], c['model'], s)] for s in SEEDS}, metrics.unit_values)


def read_genetic(job):
    c, saved = job
    case = load_case(c['screen'], c['fold'], c['model'], True, c['cell'])
    sd = read_side(c['screen'], case, 'held-out', c['cell'])
    f = BINDINGS[c['screen']].files(c['fold'])
    ref = references(sd, case, 'heldout', mean_response(f['constants'], f['train']), targets(c['screen'], c['fold'], sd['u']))
    cal = read_side(c['screen'], load_case(c['screen'], c['fold'], c['model'], False), 'calibration', None)
    second = {s: menu.pick_two(cal['a'], (cal['p'][s] + cal['p'][net_cost.partner(cal, s)]) / 2, cal['u'])[0] for s in SEEDS}
    print(key(c), flush=True)
    return key(c), card_lines(sd, ref, saved, second, metrics.target_excluded_values)


def readings(cards, found, screen_column):
    rows = []
    for c in cards:
        lines = found[key(c)]
        for s in SEEDS:
            rows += [{'card': key(c), **({'screen': c['screen']} if screen_column else {}), 'fold': c['fold'], 'model': c['model'],
                      'kind': c['kind'], 'seed': s, 'prediction': name, 'metric': metric, 'value': metrics.point(pair)}
                     for name, values in lines[s].items() for metric, pair in values.items()]
    return rows


def plain(c):
    return {k: c[k] for k in ('screen', 'fold', 'model', 'cell', 'kind', 'targets')}


def tahoe_stage(out, workers):
    out = Path(out)
    amounts = json.loads((out / 'AMOUNTS.json').read_text())
    second = {(r['fold'], r['model'], int(r['seed'])): ast.literal_eval(r['second_amounts']) for r in csv.DictReader(open(out / 'PLATE_FREE_SEED.csv'))}
    cards = [plain(c) for c in all_changes() if c['screen'] == 'tahoe']
    with Pool(workers) as pool:
        found = dict(pool.imap_unordered(read_tahoe, [(c, amounts[key(c)], second) for c in cards], chunksize=1))
    write_csv(out / 'METRIC_READINGS_TAHOE.csv', readings(cards, found, False), mode='w')


def genetic_stage(out, workers):
    out = Path(out)
    amounts = json.loads((out / 'AMOUNTS.json').read_text())
    cards = [plain(c) for c in all_changes() if c['screen'] != 'tahoe']
    with Pool(workers) as pool:
        found = dict(pool.imap_unordered(read_genetic, [(c, amounts[key(c)]) for c in cards], chunksize=1))
    write_csv(out / 'METRIC_READINGS_GENETIC.csv', readings(cards, found, True), mode='w')


def cost_root(screen, model):
    if model == 'official_state':
        return COST_ROOTS['official_state']
    if model == 'attention':
        return COST_ROOTS['attention_jiang' if screen == 'jiang' else 'attention']
    return COST_ROOTS['built']


def read_other(row):
    screen, fold, model, cell = row['screen'], row['fold'], row['model'], row['cell']
    _, held = cost_units(screen, fold)
    bags = load_case(screen, fold, 'summary', True)['bags']['heldout']
    found = {k: v for k, v in runs.find_runs(cost_root(screen, model) / screen / fold / 'natural').items() if k[0] == RUN_MODEL[model] and k[4] == cell}
    seeds = sorted(k[2] for k in found)
    loaded = {s: runs.read_run(next(v[0] for k, v in found.items() if k[2] == s), RUN_MODEL[model]) for s in seeds}
    p = {s: grids.load(loaded[s]['parent'], 'outer', held, 'raw') for s in seeds}
    R = {s: grids.load(loaded[s]['end'], 'outer', held, 'raw') for s in seeds}
    if screen == 'parse':
        r_bar = mean_response(ROOT / f'parse-pbmc-inputs/inputs/{fold}/training_constants.npz',
                              ROOT / f'training-arrays-parse-pbmc-tahoe-100m/training-arrays/parse_outer_{fold}/train_targets.npz')
    else:
        f = BINDINGS[screen].files(fold)
        r_bar = mean_response(f['constants'], f['train'])
    ref = references({'u': held}, {'bags': {'heldout': bags}}, 'heldout', r_bar, targets(screen, fold, held))
    out = []
    for i, s in enumerate(seeds):
        partner = seeds[(i + 1) % len(seeds)]
        for name, x in {'original': p[s], 'change': R[s], 'seed': (p[s] + p[partner]) / 2}.items():
            values = metrics.per_line(metrics.target_excluded_values(x, held, ref), held, ref)
            out += [{'screen': screen, 'fold': fold, 'model': model, 'cell': cell, 'kind': row['kind'], 'counted': row['counted'], 'seed': s,
                     'prediction': name, 'metric': metric, 'value': metrics.point(pair)} for metric, pair in values.items()]
    print(screen, fold, model, cell, flush=True)
    return out


def other_screens_stage(out, workers):
    out = Path(out)
    matched = {key(c) for c in all_changes()}
    rows = [r for r in csv.DictReader(open(out / 'RESTORED_LIST.csv')) if f"{r['screen']}|{r['fold']}|{r['model']}|{r['cell']}" not in matched]
    with Pool(workers) as pool:
        found = [r for part in pool.map(read_other, rows, chunksize=1) for r in part]
    write_csv(out / 'METRIC_READINGS_OTHER_SCREENS.csv', found, mode='w')


def by_change_stage(out):
    out = Path(out)
    t = pd.read_csv(out / 'METRIC_READINGS_TAHOE.csv')
    t = t.assign(screen='tahoe', cell=t.card.str.split('|').str[3])
    g = pd.read_csv(out / 'METRIC_READINGS_GENETIC.csv')
    g = g.assign(cell=g.card.str.split('|').str[3])
    o = pd.read_csv(out / 'METRIC_READINGS_OTHER_SCREENS.csv')
    keep = ['screen', 'fold', 'model', 'cell', 'seed', 'prediction', 'metric', 'value']
    r = pd.concat([x[x.prediction.isin(['original', 'change', 'seed'])][keep] for x in (t, g, o)])
    r = r.rename(columns={'seed': 'run'}).pivot_table(index=['screen', 'fold', 'model', 'cell', 'run', 'metric'], columns='prediction', values='value').reset_index()
    sign = r.metric.map(lambda m: -1 if m == 'mse' else 1)
    scale = r.apply(lambda x: 100 / x['original'] if x['metric'] == 'mse' else 1., axis=1)
    r['change_gain'] = sign * scale * (r['change'] - r['original'])
    r['seed_gain'] = sign * scale * (r['seed'] - r['original'])
    r['change_minus_seed'] = r['change_gain'] - r['seed_gain']
    m = r.groupby(['screen', 'fold', 'model', 'cell', 'metric'])[['change_gain', 'seed_gain', 'change_minus_seed']].mean().reset_index()
    changes = pd.read_csv(out / 'RESTORED_LIST.csv')[['screen', 'fold', 'model', 'cell', 'kind', 'levels', 'cost_pts']]
    m.merge(changes, on=['screen', 'fold', 'model', 'cell'], how='left').to_csv(out / 'BY_CHANGE.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description='The matched comparisons on metrics other than squared error.')
    parser.add_argument('--stage', choices=('tahoe', 'genetic', 'other-screens', 'by-change'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args(argv)
    if args.stage == 'by-change':
        return by_change_stage(args.out)
    {'tahoe': tahoe_stage, 'genetic': genetic_stage, 'other-screens': other_screens_stage}[args.stage](args.out, args.workers)


if __name__ == '__main__':
    main()
