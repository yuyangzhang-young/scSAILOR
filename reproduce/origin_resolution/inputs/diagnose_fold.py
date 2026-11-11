"""Two read-outs on the saved natural predictions of one genetic fold. No training and no new predictions.

1. Line-level correction (the Tahoe-100M reading of the context-level correction, here in one convention:
   predictions and anchors clipped at zero, as the natural task's risk does). A prediction is its same-path anchor plus a
   correction. Per context, p = weighted mean over units of (prediction - anchor) and e = weighted mean of
   (truth - anchor). With |v|^2 the unit-weighted mean of the gene-mean square, the model's common error is exactly
   |e|^2 + |p|^2 - 2 e.p, checked against the risk for every fit and side. Reported per side:
   p_size = |p|^2 (needs no held-out label), e_size = |e|^2 (the anchor's common error), alignment = e.p and
   best_scale = e.p / |p|^2.
2. Reliable held-out units (signal >= 2 x noise; tables in reliability/, truth only): the natural task's risk on that
   subset, weights renormalised within it, for every fit, its same-path anchor and the fold references (data anchor,
   no change).
   noise_floor_total is the subset's weighted mean of the per-unit noise (target gene excluded).

The folders are arguments, as for ``score_fold.py``, whose scoring functions it uses. Writes
<OUT>/<dataset>/<fold>/OFFSET_READINGS.csv and RELIABLE_SCORES.csv.
Usage: python diagnose_fold.py <replogle|jiang> <fold> BANK NATURAL STATE CPA OUT
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import score_fold as sf

HERE = Path(__file__).resolve().parent
OUT = None   # the output folder, from the command line


def gene_mean(v, weights):
    return float(np.average(np.mean(v, axis=1), weights=weights))


def correction(pred, anchor, truth, contexts, weights):
    pred, anchor = np.maximum(pred, 0), np.maximum(anchor, 0)
    p = sf.project(pred - anchor, contexts, weights)
    e = sf.project(truth - anchor, contexts, weights)
    pp, ee, ep = gene_mean(p * p, weights), gene_mean(e * e, weights), gene_mean(e * p, weights)
    common = sf.risk(pred - truth, contexts, weights)['common_error_mse']
    if not np.isclose(common, ee + pp - 2 * ep, rtol=1e-9, atol=1e-15):
        raise ValueError(f'common error {common} differs from |e|^2 + |p|^2 - 2e.p = {ee + pp - 2 * ep}')
    return {'p_size': pp, 'e_size': ee, 'alignment': ep, 'best_scale': ep / pp if pp > 0 else np.nan, 'common_error': common}


def fits(dataset, fold):
    """(label, model, seed, level, step, get) for every finished fit; get(side, units) -> (prediction, anchor)."""
    for fit in sorted((sf.NATURAL / dataset / fold).glob('*_3100?_L*')):
        if not (fit / 'PREDICTIONS_READY.json').exists():
            continue
        model, seed, level = fit.name.split('_')

        def get(side, u, fit=fit):
            with np.load(fit / 'PREDICTIONS.npz') as z:
                index = {(s, int(p)): i for i, (s, p) in enumerate(zip(z['side_context'].astype(str), z['p_code']))}
                rows = [index[(f'{side}/{c}', int(p))] for c, p in zip(u['context'], u['p_code'])]
                return z['raw'][rows], z['anchor_raw'][rows]
        yield fit.name, 'attention' if model == 'state' else model, seed, level, 40000, get
    for label, root in (('state_official', sf.STATE), ('cpa', sf.CPA)):
        for fit in sorted((root / dataset / fold).glob('*_train28_*')):
            files = sorted(fit.glob('PREDICTIONS_step*.npz'), key=lambda p: int(p.stem.split('step')[1]))
            if not (fit / 'FIT_DONE.json').exists() or not files:
                continue
            level, seed = fit.name.split('_')[-2:]

            def get(side, u, path=files[-1]):
                with np.load(path) as z:
                    index = {(c, int(p)): i for i, (c, p) in enumerate(zip(z[f'{side}_context'].astype(str), z[f'{side}_p_code']))}
                    rows = [index[(c, int(p))] for c, p in zip(u['context'], u['p_code'])]
                    return z[f'{side}_raw'][rows], z[f'{side}_anchor'][rows]
            yield fit.name, label, seed, level, int(files[-1].stem.split('step')[1]), get


def main(dataset, fold):
    units = sf.sides(dataset, fold)
    held = units['heldout']
    table = pd.read_csv(HERE / 'reliability' / f'{dataset}_{fold}.csv', keep_default_na=False)
    reliable = dict(zip(zip(table.context.astype(str), table.p_code.astype(int)), table.reliable.astype(bool)))
    noise = dict(zip(zip(table.context.astype(str), table.p_code.astype(int)), table.noise))
    keys = list(zip(held['context'], held['p_code'].astype(int)))
    if set(keys) != set(reliable):
        raise ValueError('reliability table and held-out truth list different units')
    mask = np.array([reliable[k] for k in keys])
    sub = {k: v[mask] for k, v in held.items()}
    base = {'n_units': int(mask.sum()), 'of_units': len(mask), 'weight_share': float(held['weight'][mask].sum() / held['weight'].sum()),
            'noise_floor_total': float(np.average([noise[k] for k, m in zip(keys, mask) if m], weights=sub['weight']))}

    def subset_risk(pred):
        return sf.risk(np.maximum(pred[mask], 0) - sub['truth'], sub['context'], sub['weight'])

    offsets, scores = [], []
    with np.load(sf.BANKS[dataset] / 'inputs' / fold / 'BASELINE_PREDICTIONS.npz') as z:
        index = {(s, int(p)): i for i, (s, p) in enumerate(zip(z['side_context'].astype(str), z['p_code']))}
        rows = [index[(f'heldout/{c}', int(p))] for c, p in zip(held['context'], held['p_code'])]
        for name in ('anchor_raw', 'input_only_raw'):
            scores.append({'name': name, 'model': 'reference', 'seed': '', 'level': '', 'predictor': name, **base, **subset_risk(z[name][rows])})
    for label, model, seed, level, step, get in fits(dataset, fold):
        ident = {'name': label, 'model': model, 'seed': seed, 'level': level}
        for side, u in units.items():
            pred, anchor = get(side, u)
            offsets.append({**ident, 'step': step, 'side': side, **correction(pred, anchor, u['truth'], u['context'], u['weight'])})
            if side == 'heldout':
                for which, values in (('model', pred), ('anchor', anchor)):
                    scores.append({**ident, 'predictor': which, **base, **subset_risk(values)})
        print(label, flush=True)
    out = OUT / dataset / fold
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(offsets).to_csv(out / 'OFFSET_READINGS.csv', index=False)
    pd.DataFrame(scores).to_csv(out / 'RELIABLE_SCORES.csv', index=False)
    print(f'{dataset} {fold}: {len(offsets)} offset rows, {len(scores)} reliable-subset rows; reliable {base}', flush=True)


if __name__ == '__main__':
    dataset, fold, *folders = sys.argv[1:]
    sf.configure(dataset, *folders)
    OUT = Path(folders[-1])
    main(dataset, fold)
