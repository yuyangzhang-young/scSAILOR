"""Fixed-weight probes on Tahoe-100M: the probes' tables from saved grids.

The grids are the probes' saved forward passes (the built models' and State's); the forward passes need the trained
weights and are not run here. Tables, each written once into ``--out``, the later ones reading the earlier ones there:

- ``swap-scores``: ``SWAP_SCORES.csv`` of the built models;
- ``swap-summary``: ``SWAP_SUMMARY.csv``, its fold means for the summary-statistics and DeepSets models' changes;
- ``state-scores``: State's ``STATE_SCORES.csv``, ``STATE_PAIRED.csv`` and ``STATE_FOLD_MEANS.csv``, on fold 1 or
  on folds 2-3 (``--folds``);
- ``scores``: per family (``--family``) ``<FAMILY>_SCORES.csv``, ``_GEOMETRY``, ``_PAIRED``, ``_LINE_CONTRIBUTIONS`` and
  ``_SCORE_MEANS``, with the matched shrink;
- ``unseen-mean``: the held-out line's mean, as ``<FAMILY>_S4_*.csv``, ``<FAMILY>_EXTENDED_*.csv`` and
  ``<FAMILY>_S4_VS_S2.csv``;
- ``geometry``: ``ERROR_GEOMETRY_*``: increment energy and best scale, per context, line and fold;
- ``fractions``: ``ADDED_COST_FRACTIONS.csv``, what each condition removes of the cost;
- ``control-scores``: ``CONTROL_USE.csv`` and ``CONTROL_USE_SEED_MEANS.csv`` per family (``built`` or ``official_state``);
- ``own-mean-value``: ``OWN_MEAN_VALUE_H.csv``, the own mean's value H on held-out lines under the training average and
  the other plate, from the three families' score means; the summary-statistics model's training-average values are
  the swap summary's.

Condition names: ``S1_random_<k>`` one of three fixed random directions of the same size (the grids name each by its
seed, ``RANDOM``), ``S2_other_line`` another training line's mean, ``S3_non_target`` a change of the variance,
``S4_unseen_natural`` a held-out line's mean, ``output_shrink`` the matched shrink.

    python -m reproduce.origin_resolution.probes --stage scores --family summary --out DIR
"""
import argparse
import json
import re
from functools import cache
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scsailor.resolution import grids, probes
from scsailor.resolution.scoring import identities, risk
from . import ROOT, SEEDS, STATE_F23, read_csv, tahoe, write_csv

__all__ = ["fold_sides", "state_predictions", "swap_scores", "swap_summary", "state_scores", "scores", "unseen_mean", "geometry", "fractions",
           "control_scores", "own_mean_value", "main"]

SWAPS = ROOT / 'statistic-swap'   # <run>/{parent,repaired}.npz, the built models' grids
SHAMS = ROOT / 'statistic-swap-shams'   # <run>/{parent,repaired}.npz, s4/<run>/...: the sham probes' grids
STATE_SWAP = ROOT / 'state-statistic-swap'   # seed<seed>/SWAP_<state>.npz
STATE_SWAP_F23 = ROOT / 'state-statistic-swap-folds-2-3'   # <fold>/seed<seed>/SWAP_<state>.npz
STATE_FIT = ROOT / 'state-predictions-fold-1/results'   # State's 28-line fits
STATE_RUNS = ROOT / 'information-localization-state/natural'   # its real-task runs on fold 1
RUN_ROOTS = [ROOT / 'information-localization-runs-fold-1/natural', ROOT / 'information-localization-runs-fold-2/natural',
             ROOT / 'information-localization-runs-fold-3/natural', ROOT / 'information-localization-attention-model/natural']
INPUT_TABLES = ROOT / 'statistic-swap-shams'   # the sham forward passes' input readings
SUMMARY_MODELS = ('deepsets', 'summary')   # the swap summary's models; the attention model's swap is read by ``scores``
SUMMARY_COLUMNS = {'dep_common': ('heldout', 'dependence_common'), 'dep_contrast': ('heldout', 'dependence_contrast'),
                   'H_heldout': ('heldout', 'H_total'), 'H_heldout_common': ('heldout', 'H_common'),
                   'H_heldout_contrast': ('heldout', 'H_contrast'), 'H_train': ('train', 'H_total')}   # name: (side, column)
CONTROL_GRIDS = ROOT / 'control-use-by-training-length'   # CONTROL_USE_<model>_<seed>_<level>_step<step>.npz
STATE_CONTROL_GRIDS = ROOT / 'state-control-use-by-training-length'   # st_anchored_train28_<level>_<seed>/CONTROL_USE_step<step>.npz
RESAMPLES = 2000
FAMILIES = {'summary': 'SUMMARY', 'state': 'ATTENTION', 'official_state': 'OFFICIAL_STATE'}
UNSEEN = 'S4_unseen_natural'
CONDITIONS = ('true', 'train_avg', 'other_plate')   # the forward passes' conditions of a control-mean swap
AGNOSTIC = ('train_avg', 'pooled')   # the control-use readings' bags that do not read the context
# the random directions: label -> the grids' member, named by the direction's seed
RANDOM = {'S1_random_1': 'S1_2026102501', 'S1_random_2': 'S1_2026102502', 'S1_random_3': 'S1_2026102503'}
MEAN_CONDITIONS = ('train_avg', 'other_plate', 'S2_other_line', UNSEEN, *RANDOM)   # they change the control mean
GRID_SIDE = {'heldout': 'outer', 'train': 'train'}   # a side's name in ``grids``
META = ['model', 'fold', 'cell', 'side', 'condition', 'part', 'object']


def write(path, table):
    write_csv(path, table, lineterminator='\n')


@cache
def fold_sides(fold):
    """side -> a fold's units, with the anchor (the input set's mean plus the transferred response), the clipped baseline,
    the three parts of its risk (``baseline_risk``), the input sets' means and the training-average profile."""
    f = tahoe.files(fold)
    with np.load(f['constants']) as constants:
        r, c_train = constants['r_input'].astype(np.float64), constants['c_train'].astype(np.float64)
    sides = {'heldout': tahoe.units(fold, 'outer'), 'train': tahoe.units(fold, 'train')}
    with h5py.File(f['inputs'], 'r') as bank:
        for side, u in sides.items():
            means = {c: bank[side][c]['x'][:].astype(np.float64).mean(0) for c in np.unique(u['context'])}
            anchor = np.stack([means[c] for c in u['context']]) + r[u['p_code']]
            clipped = np.maximum(anchor, 0)
            sides[side] = u | {'anchor': anchor, 'clipped_baseline': clipped, 'means': means, 'c_train': c_train,
                               'baseline': np.array([risk(clipped - u['truth'], u['context'], u['weight'])[k] for k in probes.KEYS])}
    return sides


def aligned_grid(path, sides, conditions):
    return {side: {c: grids.load(path, GRID_SIDE[side], u, c) for c in conditions} for side, u in sides.items()}


def built_runs(family=None):
    """(row of the run's record, run folder, swap-grid folder) of each change with swap grids. ``family``: 'summary' or
    'state' (the attention model), which leaves DeepSets out as the sham probes did."""
    for target in sorted(SWAPS.glob('natural-*')):
        if not ((target / 'parent.npz').exists() and (target / 'repaired.npz').exists()):
            continue
        run = next(root / target.name for root in RUN_ROOTS if (root / target.name).exists())
        row = json.loads((run / 'RUN_RECORD.json').read_text())['row']
        if family is None or row['model'] == family:
            yield row, run, target


def run_meta(row):
    return {'model': {'state': 'attention'}.get(row['model'], row['model']), 'fold': row['fold'], 'seed': int(row['seed']),
            'cell': f"{row['cut']}/{row['arm']}/x{float(row['rate_multiple']):g}"}


def state_predictions(true_path, swap_path, sides):
    """State under the three conditions: 'true' from the run's saved predictions, in either layout."""
    return {side: {'true': grids.load(true_path, GRID_SIDE[side], u), **{c: grids.load(swap_path, GRID_SIDE[side], u, c) for c in CONDITIONS[1:]}}
            for side, u in sides.items()}


def state_files(fold, seed):
    """(state, cell, its predictions, its swap grids) of State's original model and changes on a fold."""
    if fold == 'F1':
        yield 'parent', 'parent', STATE_FIT / f'st_anchored_train28_L2_{seed}/PREDICTIONS_step5000.npz', STATE_SWAP / f'seed{seed}/SWAP_parent.npz'
        for cell in ('junction-a-x1', 'head-a-x1'):
            yield ('repaired', cell.replace('-', '/'), STATE_RUNS / f'natural-officialstate-{seed}-n28-{cell}/predictions-step05000.npz',
                   STATE_SWAP / f'seed{seed}/SWAP_{cell}.npz')
    else:
        run = STATE_F23 / fold / 'natural' / f'natural-{fold}-officialstate-{seed}-n28-junction-f-x1'
        swaps = STATE_SWAP_F23 / fold / f'seed{seed}'
        yield 'parent', 'parent', run / 'predictions-parent.npz', swaps / 'SWAP_parent.npz'
        yield 'repaired', 'junction/f/x1', run / 'predictions-step05000.npz', swaps / 'SWAP_junction-f-x1.npz'


def swap_scores(out):
    """L, dependence and H of every built model's change and original model under the three conditions."""
    rows = []
    for row, _, target in built_runs():
        sides = fold_sides(row['fold'])
        for state in ('parent', 'repaired'):
            preds = aligned_grid(target / f'{state}.npz', sides, CONDITIONS)
            for side, u in sides.items():
                for condition, values in probes.condition_scores(preds[side], u, u['clipped_baseline']).items():
                    rows.append({**run_meta(row), 'state': state, 'side': side, 'condition': condition, **values})
    write(Path(out) / 'SWAP_SCORES.csv', rows)


def swap_summary(out):
    """Fold means of ``SWAP_SCORES.csv`` in ``out``, per change of the summary-statistics and DeepSets models: on
    held-out lines the excess under each condition (the change's mean L minus the original model's), and under the
    training average the dependence and H of both, on held-out lines and H on training lines."""
    groups = {}
    for r in read_csv(Path(out) / 'SWAP_SCORES.csv'):
        if r['model'] in SUMMARY_MODELS:
            groups.setdefault((r['model'], r['cell'], r['fold']), {}).setdefault((r['state'], r['side'], r['condition']), []).append(r)
    mean = lambda rows, column: float(np.mean([float(r[column]) for r in rows]))
    table = []
    for (model, cell, fold), g in sorted(groups.items()):
        row = {'model': model, 'cell': cell, 'fold': fold}
        for condition in CONDITIONS:
            row[f'excess_heldout_{condition}'] = (mean(g['repaired', 'heldout', condition], 'L_total')
                                                  - mean(g['parent', 'heldout', condition], 'L_total'))
        for state in ('parent', 'repaired'):
            for name, (side, column) in SUMMARY_COLUMNS.items():
                row[f'{name}_{state}'] = mean(g[state, side, 'train_avg'], column)
        table.append(row)
    write(Path(out) / 'SWAP_SUMMARY.csv', table)


def state_scores(folds, out):
    """State's scores, paired excess cost and fold means on ``folds``. Folds 2-3 add the two levels of L to the
    pairs, and name the fold in the means."""
    later = folds != ('F1',)
    rows = []
    for fold in folds:
        sides = fold_sides(fold)
        for seed in SEEDS:
            for state, cell, true_path, swap_path in state_files(fold, seed):
                preds = state_predictions(true_path, swap_path, sides)
                for side, u in sides.items():
                    for condition, values in probes.condition_scores(preds[side], u, u['clipped_baseline']).items():
                        rows.append({'model': 'official_state', 'fold': fold, 'seed': seed, 'cell': cell, 'state': state, 'side': side,
                                     'condition': condition, **values})
    write(Path(out) / 'STATE_SCORES.csv', rows)

    def find(row, state, condition):
        return next(r for r in rows if (r['fold'], r['seed'], r['side'], r['state'], r['condition']) == (row['fold'], row['seed'],
                                                            row['side'], state, condition)
                    and (state == 'parent' or r['cell'] == row['cell']))
    ids = ('model', 'fold', 'seed', 'cell', 'side', 'condition')
    paired = [{k: row[k] for k in ids} | probes.excess_columns(row, find(row, 'parent', row['condition']), find(row, 'repaired', 'true'),
                                                            find(row, 'parent', 'true'), later)
              for row in rows if row['state'] == 'repaired']
    write(Path(out) / 'STATE_PAIRED.csv', paired)
    means = []
    for fold in folds:
        for cell in sorted({r['cell'] for r in paired if r['fold'] == fold}):
            for side in ('heldout', 'train'):
                for condition in CONDITIONS:
                    group = [r for r in paired if (r['fold'], r['cell'], r['side'], r['condition']) == (fold, cell, side, condition)]
                    means.append(({'fold': fold} if later else {}) | dict(cell=cell, side=side, condition=condition,
                                                                          **{k: float(np.mean([r[k] for r in group])) for k in group[0] if k not in ids}))
    write(Path(out) / 'STATE_FOLD_MEANS.csv', means)


@cache
def mean_diagnostics(fold, saved=True):
    """(side, context, condition) -> the input's readings: distances from the training average and, with ``saved``, what
    the forward passes saved about their carriers."""
    result = {}
    for side, d in fold_sides(fold).items():
        for context, own in d['means'].items():
            line, plate = context.split('|')
            other = f"{line}|{'plate14' if plate == 'plate6' else 'plate6'}"
            denominator = np.linalg.norm(own - d['c_train'])
            for condition, mean in (('true', own), ('train_avg', d['c_train']), ('other_plate', d['means'][other])):
                result[(side, context, condition)] = dict(distance_to_train=float(np.linalg.norm(mean - d['c_train'])),
                                                          distance_to_train_ratio=float(np.linalg.norm(mean - d['c_train']) / denominator),
                                                          mean_change_norm=float(np.linalg.norm(mean - own)),
                                                          mean_change_ratio=float(np.linalg.norm(mean - own) / denominator))
    if saved:
        for path in (INPUT_TABLES / f'INPUT_DIAGNOSTICS_{fold}.csv', INPUT_TABLES / f'INPUT_REFERENCE_DIAGNOSTICS_{fold}.csv'):
            if path.exists():
                for row in read_csv(path):   # the true and the other saved conditions keep their exact saved input construction
                    result[(row['side'], row['context'], row['condition'])] = {k: float(v) for k, v in row.items()
                                                                               if k not in ('fold', 'side', 'context', 'condition',
                                                            'donor') and v != ''}
    return result


def unseen_diagnostics(fold):
    """``mean_diagnostics`` with the readings of the held-out line's mean."""
    rows = pd.read_csv(INPUT_TABLES / f'S4_INPUT_DIAGNOSTICS_{fold}.csv').to_dict('records')
    return mean_diagnostics(fold) | {(r['side'], r['context'], UNSEEN): {k: v for k,
                                                            v in r.items() if k not in ('fold', 'side', 'context', 'condition')} for r in rows}


def write_means(frame, path):
    """The mean over seeds of every numeric column."""
    keys = ['model', 'fold', 'cell', 'state', 'side', 'condition']
    frame.groupby(keys, sort=True).mean(numeric_only=True).drop(columns=['seed']).reset_index().to_csv(path, index=False)


def with_reference_readings(out):
    """State's tables with the realized input readings attached, without scoring again."""
    out = Path(out)
    reference = pd.read_csv(INPUT_TABLES / 'INPUT_REFERENCE_DIAGNOSTICS_F1.csv').set_index(['side', 'context', 'condition'])
    columns = [c for c in reference.columns if c != 'fold']
    geom = pd.read_csv(out / 'OFFICIAL_STATE_GEOMETRY.csv')
    for index, row in geom.iterrows():
        d = reference.loc[(row.side, row.context, 'true' if row.condition == 'output_shrink' else row.condition)]
        for c in columns:
            geom.loc[index, c] = d[c]
    geom.to_csv(out / 'OFFICIAL_STATE_GEOMETRY.csv', index=False)
    table = pd.read_csv(out / 'OFFICIAL_STATE_SCORES.csv')
    keys = ['model', 'fold', 'seed', 'cell', 'state', 'side', 'condition']
    grouped = {key: group for key, group in geom.groupby(keys)}
    for index, row in table.iterrows():
        group = grouped[tuple(row[k] for k in keys)]
        for c in columns:
            table.loc[index, c] = (group[c] * group.support_weight).sum() / group.support_weight.sum()
    table.to_csv(out / 'OFFICIAL_STATE_SCORES.csv', index=False)
    write_means(table, out / 'OFFICIAL_STATE_SCORE_MEANS.csv')


def with_anchors(sides, run, state):
    """The sides with the state's saved anchors."""
    path = run / ('predictions-parent.npz' if state == 'parent' else 'predictions-step40000.npz')
    return {side: d | {'anchor': grids.load(path, GRID_SIDE[side], d, 'anchor_raw')} for side, d in sides.items()}


def built_grids(run, target, state, sides, conditions=CONDITIONS, shams=''):
    """A built model state's grids: the swaps and, where saved, the sham conditions under ``SHAMS / shams``."""
    found = aligned_grid(target / f'{state}.npz', sides, conditions)
    path = SHAMS / shams / run.name / f'{state}.npz'
    with np.load(path) as z:
        names = [c for c in z.files if c.startswith('S')]
    extra = aligned_grid(path, sides, names)
    label = {member: name for name, member in RANDOM.items()}
    for side in sides:
        found[side].update({label.get(c, c): grid for c, grid in extra[side].items()})
    return found


def s1_mean_rows(table, keys):
    """Readouts, not predictions, averaged over the fixed random directions: adds the 'S1_mean' rows."""
    frame = pd.DataFrame(table)
    random = frame[frame.condition.str.startswith('S1_')]
    if len(random):
        averaged = random.groupby(keys, sort=True).mean(numeric_only=True).reset_index()
        averaged['condition'] = 'S1_mean'
        table.extend(averaged.to_dict('records'))


def scores(family, out):
    """Every probe condition of one family, with the matched shrink."""
    all_scores, geometry_rows, pairs = [], [], []
    if family == 'official_state':
        sides = fold_sides('F1')
        diagnostics = mean_diagnostics('F1', saved=False)   # its input readings are attached afterwards
        for seed in SEEDS:
            files = {cell: (path, swaps) for _, cell, path, swaps in state_files('F1', seed)}
            for cell in ('junction/a/x1', 'head/a/x1'):
                meta = dict(model='official_state', fold='F1', seed=seed, cell=cell)
                values = {}
                for state, label in (('parent', 'parent'), ('repaired', cell)):
                    rows, geo, lines = probes.score_model(meta, state, state_predictions(*files[label], sides), sides, diagnostics)
                    all_scores.extend(rows)
                    geometry_rows.extend(geo)
                    values[state] = lines
                pairs.append((meta, values, sides))
    else:
        for row, run, target in built_runs(family):
            sides = fold_sides(row['fold'])
            meta = run_meta(row)
            values = {}
            for state in ('parent', 'repaired'):
                rows, geo, lines = probes.score_model(meta, state, built_grids(run, target, state, sides), with_anchors(sides, run, state),
                                                      mean_diagnostics(row['fold']))
                all_scores.extend(rows)
                geometry_rows.extend(geo)
                values[state] = lines
            pairs.append((meta, values, sides))
    paired, lines = probes.paired_tables(pairs)
    s1_mean_rows(all_scores, ['model', 'fold', 'seed', 'cell', 'state', 'side'])
    s1_mean_rows(geometry_rows, ['model', 'fold', 'seed', 'cell', 'state', 'side', 'context'])
    prefix = FAMILIES[family]
    for label, rows in (('SCORES', all_scores), ('GEOMETRY', geometry_rows), ('PAIRED', paired), ('LINE_CONTRIBUTIONS', lines)):
        write(Path(out) / f'{prefix}_{label}.csv', rows)
    write_means(pd.DataFrame(all_scores), Path(out) / f'{prefix}_SCORE_MEANS.csv')
    if family == 'official_state':
        with_reference_readings(out)


def compare_donors(lines):
    """The held-out line's mean against a training line's, per line and seed, with intervals over lines."""
    selected = lines[(lines.metric == 'E') & lines.condition.isin([UNSEEN, 'S2_other_line'])]
    keys = ['model', 'fold', 'cell', 'side', 'part', 'seed', 'line']
    paired = selected.pivot(index=keys, columns='condition', values='contribution')
    paired['difference'] = paired[UNSEEN] - paired['S2_other_line']
    rows = []
    for key, g in paired.reset_index().groupby(keys[:5], sort=True):
        matrix = g.pivot(index='line', columns='seed', values='difference')
        n = len(matrix)
        draws = probes.line_draws(n)
        vectors = {str(seed): matrix[seed].to_numpy() for seed in matrix.columns}
        vectors['mean'] = matrix.mean(axis=1).to_numpy()
        for seed, vector in vectors.items():
            samples = vector[draws].sum(1)
            rows.append(dict(zip(keys[:5], key)) | dict(seed=seed, metric='E_S4_minus_S2', value=float(vector.sum()),
                                                        ci_low=float(np.percentile(samples, 2.5)),
                                                        ci_high=float(np.percentile(samples, 97.5)), n_lines=n))
    return pd.DataFrame(rows)


def unseen_mean(family, out):
    """The condition of a held-out line's mean, scored apart and joined to the family's tables."""
    out, prefix = Path(out), FAMILIES[family]
    table_scores, table_geometry, pairs = [], [], []
    for row, run, target in built_runs(family):
        sides = fold_sides(row['fold'])
        diagnostics = unseen_diagnostics(row['fold'])
        meta = run_meta(row)
        values = {}
        for state in ('parent', 'repaired'):
            found = built_grids(run, target, state, sides, ('true', 'train_avg'), 's4')
            rows, geo, lines = probes.score_model(meta, state, found, with_anchors(sides, run, state), diagnostics)
            table_scores.extend(r for r in rows if r['condition'] == UNSEEN)
            table_geometry.extend(r for r in geo if r['condition'] == UNSEEN)
            values[state] = lines
        pairs.append((meta, values, sides))
    paired, lines = probes.paired_tables(pairs)
    tables = (('SCORES', table_scores), ('GEOMETRY', table_geometry), ('PAIRED', [r for r in paired if r['condition'] == UNSEEN]),
              ('LINE_CONTRIBUTIONS', [r for r in lines if r['condition'] == UNSEEN]))
    for name, rows in tables:
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f'{prefix}_S4_{name}.csv', index=False)
        combined = pd.concat([pd.read_csv(out / f'{prefix}_{name}.csv'), frame], ignore_index=True)
        combined.to_csv(out / f'{prefix}_EXTENDED_{name}.csv', index=False)
        if name == 'SCORES':
            write_means(combined, out / f'{prefix}_EXTENDED_SCORE_MEANS.csv')
        if name == 'LINE_CONTRIBUTIONS':
            compare_donors(combined).to_csv(out / f'{prefix}_S4_VS_S2.csv', index=False)


def saved_table(out, prefix, label):
    path = Path(out) / f'{prefix}_EXTENDED_{label}.csv'
    return pd.read_csv(path if path.exists() else Path(out) / f'{prefix}_{label}.csv', dtype={'seed': str})


def built_geometry(run, target, state, sides):
    """A built model state's clipped grids with the matched shrink, and its clipped anchors."""
    found = aligned_grid(target / f'{state}.npz', sides, CONDITIONS)
    for shams in ('', 's4'):
        extra = built_grids(run, target, state, sides, (), shams)
        for side in sides:
            found[side].update(extra[side])
    anchors = {side: d['anchor'] for side, d in with_anchors(sides, run, state).items()}
    for side, d in sides.items():
        found[side]['output_shrink'], _ = probes.matched_shrink(found[side]['true'], found[side]['train_avg'], anchors[side], d['context'],
                                                            d['weight'])
        found[side] = {c: np.maximum(v, 0) for c, v in found[side].items()}
        anchors[side] = np.maximum(anchors[side], 0)
    return found, anchors


def state_anchor(seed, sides):
    path = STATE_FIT / f'st_anchored_train28_L2_{seed}/PREDICTIONS_step5000.npz'
    return {side: grids.load(path, GRID_SIDE[side], u, 'anchor') for side, u in sides.items()}


def weighted_moments(frame):
    return frame[probes.MOMENTS].to_numpy() * frame.support_weight.to_numpy()[:, None]


def interval(values):
    """The 95% interval of the finite values, and how many were not."""
    finite = np.isfinite(values)
    return (*(np.percentile(values[finite], [2.5, 97.5]) if finite.any() else (np.nan, np.nan)), int((~finite).sum()))


def aggregate_backgrounds(backgrounds):
    keys = META + ['seed', 'line']
    frame = backgrounds.copy()
    frame[probes.MOMENTS] = frame[probes.MOMENTS].mul(frame.support_weight, axis=0)
    lines = frame.groupby(keys, sort=True)[probes.MOMENTS + ['support_weight']].sum().reset_index()
    lines[probes.MOMENTS] = lines[probes.MOMENTS].div(lines.support_weight, axis=0)
    # Average permutation moments within a line and seed, retaining each permutation separately.
    random = lines[lines.condition.str.startswith('S1_')]
    if len(random):
        average = random.groupby([k for k in keys if k != 'condition'], sort=True)[probes.MOMENTS + ['support_weight']].mean().reset_index()
        average['condition'] = 'S1_mean'
        lines = pd.concat([lines, average], ignore_index=True)
    return lines


def fold_summaries(lines):
    rows = []
    for key, group in lines.groupby(META, sort=True):
        meta = dict(zip(META, key))
        by_seed = {seed: g.sort_values('line') for seed, g in group.groupby('seed')}
        averaged = group.groupby('line', sort=True)[probes.MOMENTS + ['support_weight']].mean().reset_index()
        per_seed = []
        for seed, g in list(by_seed.items()) + [('mean', averaged)]:
            n = len(g)
            weighted = weighted_moments(g)
            draw = probes.line_draws(n)
            values, samples = probes.derived(weighted.sum(0)), probes.derived(weighted[draw].sum(1))
            row = meta | dict(seed=seed, n_lines=n)
            for name in probes.STATS:
                row[name] = float(values[name])
                row[name + '_ci_low'], row[name + '_ci_high'], undefined = interval(samples[name])
                if name in ('cosine', 'alpha_star'):
                    row[name + '_undefined_bootstrap'] = undefined
            if seed != 'mean':
                per_seed.append(values)
            else:
                row['mean_seed_alpha_star'] = float(np.mean([v['alpha_star'] for v in per_seed]))
                row['mean_seed_cosine'] = float(np.mean([v['cosine'] for v in per_seed]))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(lines):
    rows = []
    pairs = [(UNSEEN, 'S2_other_line'), ('true', 'train_avg'), ('S1_mean', 'true'), (UNSEEN, 'true'), ('other_plate', 'true')]
    keys = ['model', 'fold', 'cell', 'side', 'part']
    for key, group in lines[lines.object == 'repair_increment'].groupby(keys, sort=True):
        for change, reference in pairs:
            if not {change, reference}.issubset(set(group.condition)):
                continue
            pair = group[group.condition.isin([change, reference])]
            seed_mean = pair.groupby(['condition', 'line'], sort=True)[probes.MOMENTS + ['support_weight']].mean().reset_index()
            for seed, g in list(pair.groupby('seed')) + [('mean', seed_mean)]:
                a = g[g.condition == change].sort_values('line')
                b = g[g.condition == reference].sort_values('line')
                if a.line.tolist() != b.line.tolist():
                    raise ValueError(f'{key} {change} {reference}: the conditions cover different lines')
                n = len(a)
                draws = probes.line_draws(n)

                def all_values(frame):
                    v = weighted_moments(frame)
                    return probes.derived(np.vstack([v.sum(0), v[draws].sum(1)]))
                for metric, values in probes.pair_contrast(all_values(a), all_values(b)).items():
                    lo, hi, undefined = interval(values[1:])
                    rows.append(dict(zip(keys, key)) | dict(seed=seed, condition=change, reference=reference, metric=metric,
                                                            value=values[0], ci_low=lo, ci_high=hi,
                                                            n_lines=n, undefined_bootstrap=undefined))
    return pd.DataFrame(rows)


def response_comparisons(lines):
    rows = []
    keys = ['model', 'fold', 'cell', 'side', 'condition', 'part']
    response = lines[lines.object.str.startswith('response_')]
    for key, group in response.groupby(keys, sort=True):
        average = group.groupby(['object', 'line'], sort=True)[probes.MOMENTS + ['support_weight']].mean().reset_index()
        for seed, g in list(group.groupby('seed')) + [('mean', average)]:
            vectors = {}
            for obj, frame in g.groupby('object'):
                frame = frame.sort_values('line')
                n = len(frame)
                v = weighted_moments(frame)
                vectors[obj] = probes.derived(np.vstack([v.sum(0), v[probes.line_draws(n)].sum(1)]))
            metrics = probes.response_metrics(*[vectors['response_' + state] for state in ('parent', 'repaired', 'excess')])
            for metric, values in metrics.items():
                lo, hi, undefined = interval(values[1:])
                rows.append(dict(zip(keys, key)) | dict(seed=seed, metric=metric, value=values[0], ci_low=lo, ci_high=hi, n_lines=n,
                                                        undefined_bootstrap=undefined))
    return pd.DataFrame(rows)


def geometry(out):
    """Increment energy, best scale and the responses to a changed mean: per context, line and fold, and their contrasts."""
    out = Path(out)
    rows = []

    def baselines(table, meta):
        selected = table[(table.fold == meta['fold']) & (table.cell == meta['cell']) & (table.seed == meta['seed'])]
        return {side: selected[selected.side == side].baseline_total.unique()[0] for side in ('heldout', 'train')}

    for family in ('summary', 'state'):
        table = saved_table(out, FAMILIES[family], 'SCORES')
        for row, run, target in built_runs(family):
            sides = fold_sides(row['fold'])
            meta = run_meta(row) | {'seed': str(row['seed'])}
            parent, anchors = built_geometry(run, target, 'parent', sides)
            found = {'parent': parent, 'repaired': built_geometry(run, target, 'repaired', sides)[0]}
            rows.extend(probes.pair_geometry(meta, found, anchors, sides, baselines(table, meta), MEAN_CONDITIONS))
    table, sides = saved_table(out, FAMILIES['official_state'], 'SCORES'), fold_sides('F1')
    for seed in SEEDS:
        anchors = state_anchor(seed, sides)
        files = {cell: (path, swaps) for _, cell, path, swaps in state_files('F1', seed)}
        for cell in ('junction/a/x1', 'head/a/x1'):
            meta = dict(model='official_state', fold='F1', seed=str(seed), cell=cell)
            found = {state: state_predictions(*files[label], sides) for state, label in (('parent', 'parent'), ('repaired', cell))}
            for values in found.values():
                for side in sides:
                    values[side] = {c: np.maximum(v, 0) for c, v in values[side].items()}
            rows.extend(probes.pair_geometry(meta, found, anchors, sides, baselines(table, meta), MEAN_CONDITIONS))
    backgrounds = pd.DataFrame(rows)
    backgrounds.to_csv(out / 'ERROR_GEOMETRY_BACKGROUNDS.csv.gz', index=False)
    lines = aggregate_backgrounds(backgrounds)
    for k, v in probes.derived(lines[probes.MOMENTS].to_numpy()).items():
        lines[k] = v
    lines.to_csv(out / 'ERROR_GEOMETRY_LINES.csv.gz', index=False)
    fold_summaries(lines).to_csv(out / 'ERROR_GEOMETRY_FOLDS.csv', index=False)
    paired_comparisons(lines).to_csv(out / 'ERROR_GEOMETRY_CONTRASTS.csv', index=False)
    response_comparisons(lines).to_csv(out / 'ERROR_GEOMETRY_RESPONSES.csv', index=False)


def fractions(out):
    """What each condition removes of the cost, signed, from the paired tables."""
    out = Path(out)
    tables, keys = [], ['model', 'fold', 'cell', 'side', 'part', 'seed']
    for prefix in FAMILIES.values():
        d = saved_table(out, prefix, 'PAIRED')
        wide = d.pivot(index=keys + ['condition'], columns='metric', values='value').reset_index()
        original = wide[wide.condition == 'true'][keys + ['E']].rename(columns={'E': 'E_true'})
        random = wide[wide.condition == 'S1_mean'][keys + ['E']].rename(columns={'E': 'E_random_direction_reference'})
        wide = wide.merge(original, on=keys, validate='many_to_one').merge(random, on=keys, how='left', validate='many_to_one')
        denominator = wide.E_true.to_numpy()
        wide['removed_fraction'] = probes.ratio(wide.Delta_E, denominator)
        wide['original_cost_sign'] = np.where(denominator > 0, 'positive', np.where(denominator < 0, 'negative', 'zero_undefined_ratio'))
        for metric in ('E', 'Delta_E', 'parent_change'):
            ci = d[d.metric == metric][keys + ['condition', 'ci_low', 'ci_high']].rename(columns={'ci_low': f'{metric}_ci_low',
                                                            'ci_high': f'{metric}_ci_high'})
            wide = wide.merge(ci, on=keys + ['condition'], validate='one_to_one')
        tables.append(wide)
    pd.concat(tables, ignore_index=True).to_csv(out / 'ADDED_COST_FRACTIONS.csv', index=False)


def control_files(family):
    """(model, seed, level, step, grid) of one family's control-use grids."""
    if family == 'built':
        for path in sorted(CONTROL_GRIDS.glob('CONTROL_USE_*_step*.npz')):
            model, seed, level, step = re.match(r'CONTROL_USE_(\w+?)_(\d+)_(L\d)_step(\d+)\.npz', path.name).groups()
            yield 'attention' if model == 'state' else model, int(seed), level, int(step), path
    else:
        for path in sorted(STATE_CONTROL_GRIDS.glob('st_anchored_train28_L*_*/CONTROL_USE_step*.npz')):
            level, seed = re.match(r'st_anchored_train28_(L\d)_(\d+)', path.parent.name).groups()
            yield 'official_state', int(seed), level, int(re.match(r'CONTROL_USE_step(\d+)\.npz', path.name).group(1)), path


def control_scores(family, out):
    """The value of a model's own controls per training length, on Tahoe fold 1, with intervals over lines."""
    sides = fold_sides('F1')
    draws = {}
    for i, (side, u) in enumerate(sides.items()):
        n = len(np.unique(identities(u)))
        draws[side] = np.random.default_rng(np.random.SeedSequence([20261022, i])).integers(0, n, size=(RESAMPLES, n))
    rows = []
    for model, seed, level, step, path in control_files(family):
        for side, u in sides.items():
            preds = {c: grids.load(path, GRID_SIDE[side], u, c) for c in ('own',) + AGNOSTIC}
            for condition, values in probes.own_value(preds, u, u['clipped_baseline'], draws[side], AGNOSTIC).items():
                rows.append({'model': model, 'seed': seed, 'level': level, 'step': step, 'side': side, 'condition': condition, **values})
    write(Path(out) / 'CONTROL_USE.csv', rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    ids = ('model', 'seed', 'level', 'step', 'side', 'condition')
    numeric = [f for f in fields if f not in ids and not f.endswith(('_low', '_high'))]
    groups = {}
    for row in rows:
        groups.setdefault((row['model'], row['level'], row['step'], row['side'], row['condition']), []).append(row)
    write(Path(out) / 'CONTROL_USE_SEED_MEANS.csv',
          [{'model': k[0], 'level': k[1], 'step': k[2], 'side': k[3], 'condition': k[4], 'seeds': len(v),
            **{f: float(np.mean([r[f] for r in v])) for f in numeric if f in v[0]}}
           for k, v in sorted(groups.items())])


def own_mean_value(out):
    """H on held-out lines, total, common and contrast, of the parent and the repaired model under the training-average
    and other-plate means, from ``<FAMILY>_EXTENDED_SCORE_MEANS.csv`` (summary-statistics, attention) and
    ``OFFICIAL_STATE_SCORE_MEANS.csv`` in ``out``; the summary-statistics model's training-average values from
    ``SWAP_SUMMARY.csv`` there."""
    out = Path(out)
    original = pd.read_csv(out / 'SWAP_SUMMARY.csv')
    rows = []
    for prefix in ('SUMMARY', 'ATTENTION', 'OFFICIAL_STATE'):
        stem = prefix + '_EXTENDED' if prefix != 'OFFICIAL_STATE' else prefix
        means = pd.read_csv(out / f'{stem}_SCORE_MEANS.csv')
        for r in means[(means.side == 'heldout') & means.condition.isin(['train_avg', 'other_plate'])].to_dict('records'):
            row = {k: r[k] for k in ('model', 'fold', 'cell', 'state', 'condition', 'H_total', 'H_common', 'H_contrast')}
            row['source'] = f'{stem}_SCORE_MEANS.csv'
            if prefix == 'SUMMARY' and r['condition'] == 'train_avg':
                [match] = original[(original.model == r['model']) & (original.fold == r['fold']) & (original.cell == r['cell'])].to_dict('records')
                for part, name in (('total', 'H_heldout'), ('common', 'H_heldout_common'), ('contrast', 'H_heldout_contrast')):
                    row['H_' + part] = match[name + '_' + r['state']]
                row['source'] = 'not-deposited:SWAP_SUMMARY.csv'
            elif prefix == 'ATTENTION':
                row['source'] += '; original: information-localization/statistic-swap/results/SWAP_SCORES.csv'
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'OWN_MEAN_VALUE_H.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fixed-weight probes on Tahoe-100M.')
    parser.add_argument('--stage', required=True,
                        choices=('swap-scores', 'swap-summary', 'state-scores', 'scores', 'unseen-mean', 'geometry', 'fractions',
                                 'control-scores', 'own-mean-value'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--family', help="scores, unseen-mean: summary, state (the attention model) or official_state; control-scores: built or official_state")
    parser.add_argument('--folds', nargs='+', default=['F1'], help='state-scores: F1, or F2 F3')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'state-scores':
        return state_scores(tuple(args.folds), args.out)
    if args.stage in ('scores', 'unseen-mean', 'control-scores'):
        return {'scores': scores, 'unseen-mean': unseen_mean, 'control-scores': control_scores}[args.stage](args.family, args.out)
    return {'swap-scores': swap_scores, 'swap-summary': swap_summary, 'geometry': geometry, 'fractions': fractions,
            'own-mean-value': own_mean_value}[args.stage](args.out)


if __name__ == '__main__':
    main()
