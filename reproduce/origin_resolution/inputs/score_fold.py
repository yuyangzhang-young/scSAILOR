"""Score every finished natural fit of one genetic fold: the built models (summary-statistics, DeepSets, attention),
State and CPA, plus the fold references and the target-excluded rescoring. Fits without their completion file are
skipped.

The natural task's risk (``project`` and ``risk``, the same code as in ``scsailor.natural.score``, so that the script
stands alone), predictions and anchors clipped at zero, the held-out truth from
<NATURAL>/<dataset>/truth/<fold>_HELDOUT_TRUTH.npz, training targets with their weights. Primary score: all 2,000 genes,
target included. Each model's ratio is against its own same-path anchor.

The folders are arguments: the dataset's natural-task bank (BANK: training-arrays/<fold>/, inputs/<fold>/), the built
models' fits and the held-out truth (NATURAL: <dataset>/truth/, <dataset>/<fold>/<model>_<seed>_<level>/), State's and
CPA's 28-line fits (STATE, CPA: <dataset>/<fold>/<fit>/) and the output folder (OUT). Writes to <OUT>/<dataset>/<fold>/:
  INHOUSE_SCORES.csv, STATE_SCORES.csv, CPA_SCORES.csv (one row per fit), REFERENCE_SCORES.csv,
  TARGET_EXCLUDED.csv (held-out, every fit and reference, full and target-removed), SUMMARY.csv (seed means and ratios).
Usage: python score_fold.py <replogle|jiang> <fold> BANK NATURAL STATE CPA OUT
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BANKS, NATURAL, STATE, CPA, OUT = {}, None, None, None, None   # set from the command line by ``configure``


def configure(dataset, bank, natural, state, cpa, out):
    global NATURAL, STATE, CPA, OUT
    BANKS[dataset] = Path(bank)
    NATURAL, STATE, CPA, OUT = Path(natural), Path(state), Path(cpa), Path(out)


def project(values, contexts, weights):  # as scsailor.natural.score
    out = np.empty_like(values)
    for context in np.unique(contexts):
        mask = contexts == context
        out[mask] = np.average(values[mask], axis=0, weights=weights[mask])
    return out


def risk(error, contexts, weights):  # as scsailor.natural.score
    common = project(error, contexts, weights)
    mse = lambda v: float(np.average(np.mean(v * v, axis=1), weights=weights))
    return {'total_mse': mse(error), 'common_error_mse': mse(common), 'difference_error_mse': mse(error - common)}


def masked_risk(error, contexts, weights, valid):
    error = np.where(valid, error, 0.)
    common = project(error, contexts, weights)
    n = valid.sum(axis=1)
    mse = lambda v: float(np.average((v * v * valid).sum(axis=1) / n, weights=weights))
    return {'total_mse': mse(error), 'common_error_mse': mse(common), 'difference_error_mse': mse(error - common)}


def sides(dataset, fold):
    bank = BANKS[dataset]
    with np.load(bank / 'training-arrays' / fold / 'train_targets.npz') as z:
        keys = z['context_keys'].astype(str)
        train = dict(truth=z['target'].astype(np.float64), context=keys[z['parent']], p_code=z['p_code'], weight=z['weight'])
    with np.load(NATURAL / dataset / 'truth' / f'{fold}_HELDOUT_TRUTH.npz') as z:
        held = dict(truth=z['truth'].astype(np.float64), context=z['context'].astype(str), p_code=z['p_code'], weight=z['weight'])
    return {'train': train, 'heldout': held}


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(dataset, fold):
    units = sides(dataset, fold)
    held = units['heldout']
    out = OUT / dataset / fold
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(BANKS[dataset] / 'inputs' / fold / 'TARGET_GENE_MASK.csv').set_index('p_code').target_coordinate
    valid = np.ones_like(held['truth'], dtype=bool)
    for i, c in enumerate(held['p_code']):
        if table[c] >= 0:
            valid[i, table[c]] = False
    masked = []

    def score_sides(get, names):
        row = {}
        for side, u in units.items():
            for name in names:
                pred = np.maximum(get(side, u, name), 0)
                for k, v in risk(pred - u['truth'], u['context'], u['weight']).items():
                    row[f'{side}_{name}_{k}'] = v
        return row

    def add_masked(label, pred):
        e = np.maximum(pred, 0) - held['truth']
        masked.append({'name': label} | {f'full_{k}': v for k, v in risk(e, held['context'], held['weight']).items()}
                      | {f'masked_{k}': v for k, v in masked_risk(e, held['context'], held['weight'], valid).items()})

    # References.
    with np.load(BANKS[dataset] / 'inputs' / fold / 'BASELINE_PREDICTIONS.npz') as z:
        index = {(s, int(p)): i for i, (s, p) in enumerate(zip(z['side_context'].astype(str), z['p_code']))}
        names = ['anchor_raw', 'input_only_raw', 'train_p_mean_raw', 'l0_anchor_raw']
        data = {n: z[n] for n in names}
    take = lambda side, u: [index[(f'{side}/{c}', int(p))] for c, p in zip(u['context'], u['p_code'])]
    write(out / 'REFERENCE_SCORES.csv', [score_sides(lambda s, u, n: data[n][take(s, u)], names)])
    for n in ('anchor_raw', 'input_only_raw'):
        add_masked(n, data[n][take('heldout', held)])

    # Built models' fits (summary-statistics, DeepSets, attention = 'state' in the fits' names).
    rows = []
    for fit in sorted((NATURAL / dataset / fold).glob('*_3100?_L*')):
        if not (fit / 'PREDICTIONS_READY.json').exists():
            continue
        model, seed, level = fit.name.split('_')
        with np.load(fit / 'PREDICTIONS.npz') as z:
            index = {(s, int(p)): i for i, (s, p) in enumerate(zip(z['side_context'].astype(str), z['p_code']))}
            data = {'model': z['raw'], 'anchor': z['anchor_raw']}
        rows.append({'model': 'attention' if model == 'state' else model, 'seed': seed, 'level': level}
                    | score_sides(lambda s, u, n: data[n][take(s, u)], ['model', 'anchor']))
        add_masked(fit.name, data['model'][take('heldout', held)])
    write(out / 'INHOUSE_SCORES.csv', rows)

    # Published models: PREDICTIONS_step<N>.npz with <side>_raw / <side>_anchor.
    for label, root, filename in (('state_official', STATE, 'STATE_SCORES.csv'), ('cpa', CPA, 'CPA_SCORES.csv')):
        rows = []
        for fit in sorted((root / dataset / fold).glob('*_train28_*')):
            done = fit / 'FIT_DONE.json'
            files = sorted(fit.glob('PREDICTIONS_step*.npz'), key=lambda p: int(p.stem.split('step')[1]))
            if not done.exists() or not files:
                continue
            level, seed = fit.name.split('_')[-2:]
            with np.load(files[-1]) as z:
                row = {'model': label, 'seed': seed, 'level': level, 'step': files[-1].stem.split('step')[1]}
                for side, u in units.items():
                    index = {(c, int(p)): i for i, (c, p) in enumerate(zip(z[f'{side}_context'].astype(str), z[f'{side}_p_code']))}
                    rows_side = [index[(c, int(p))] for c, p in zip(u['context'], u['p_code'])]
                    for name, key in (('model', f'{side}_raw'), ('anchor', f'{side}_anchor')):
                        for k, v in risk(np.maximum(z[key][rows_side], 0) - u['truth'], u['context'], u['weight']).items():
                            row[f'{side}_{name}_{k}'] = v
                    if side == 'heldout':
                        add_masked(fit.name, z['heldout_raw'][rows_side])
            rows.append(row)
        write(out / filename, rows)
    write(out / 'TARGET_EXCLUDED.csv', masked)

    # Seed means and ratios to the anchor.
    frames = [pd.read_csv(out / f) for f in ('INHOUSE_SCORES.csv', 'STATE_SCORES.csv', 'CPA_SCORES.csv') if (out / f).exists()]
    s = pd.concat(frames, ignore_index=True)
    for side in ('heldout', 'train'):
        s[f'{side}_ratio'] = s[f'{side}_model_total_mse'] / s[f'{side}_anchor_total_mse']
    g = s.groupby(['model', 'level'])
    cols = [f'heldout_model_{k}_mse' for k in ('total', 'common_error', 'difference_error')] + ['heldout_anchor_total_mse']
    summary = g[cols].mean().join(g.seed.count().rename('seeds')).join(g.heldout_ratio.agg(['mean', 'min', 'max']).add_prefix('heldout_ratio_')) \
        .join(g.train_ratio.mean().rename('train_ratio_mean')).reset_index()
    summary.to_csv(out / 'SUMMARY.csv', index=False)
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    dataset, fold, *folders = sys.argv[1:]
    configure(dataset, *folders)
    main(dataset, fold)
