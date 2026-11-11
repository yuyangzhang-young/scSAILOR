"""Why the cell-line-level correction does not carry to new lines: training coverage and the target's reproducibility,
Tahoe-100M, from saved grids and input bags. The saved tables were read with three BLAS threads
(``OMP_NUM_THREADS=3``).

Geometry follows ``card.alignment`` on the raw scale: inner product <x, y> = sum_u w_u mean_g(x_ug y_ug) / sum w; the
common part is the per-context weighted mean (``scoring.project``), the contrast part the rest; c = truth - anchor,
h = prediction - anchor.

Two stages, writing into ``--out``:

- ``coverage``: ``COVERAGE.csv``, the summary-statistics model's correction on fold 1's 14 held-out lines (the units
  of the 92 drugs every subset trained on) against the number of training lines: the grouped-validation runs on 7, 14
  and 21 lines (L1, L3), the natural fits on 28 (L1-L3) and origin resolution's 21- and 28-line L2 parents; per part,
  the cosine, size and best scale with each set's own anchor and with the fixed 28-line anchor;
- ``ridge --fold F1``: ``RIDGE_COMMON_<fold>.json``, a reader of the per-context common correction trained on the
  training contexts only (features: the 256-cell input bag's mean, or mean and log variance; principal components
  and penalty chosen by leaving one training line out; plain or plate intercepts), scored on the held-out contexts
  in the same geometry beside an intercept-only reference; and the held-out target's variance components (plate
  level, reproducible line level, residual).

    OMP_NUM_THREADS=3 python -m reproduce.origin_resolution.coverage --stage ridge --fold F1 --out DIR
"""
import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from scsailor.resolution import grids
from scsailor.resolution.scoring import project, risk
from . import ROOT, SEEDS, tahoe

__all__ = ["geometry", "coverage_stage", "ridge_stage", "main"]

E4 = ROOT / 'training-coverage-runs/runs'   # summary_train<n>_<seed>_<level>_wd0/prediction_step40000/
NATURAL = ROOT / 'tahoe-100m-predictions-fold-1'   # summary_<seed>_<level>_step40000/PREDICTIONS.npz
KS = (0, 1, 2, 5, 10, 20)
PENALTIES = (1e-3, 1e-2, 1e-1, 1., 10., 100., 1e3, 1e4)


def ip(x, y, w):
    return float(np.sum(w[:, None] * x * y) / x.shape[1])


def geometry(h, c, u):
    """Per part (common, differential, total): |c|^2, |h|^2, <h,c>, cos, rho, lambda* = <h,c>/|h|^2, beta, sin^2,
    (rho - cos)^2, E(1)/|c|^2 and its identity gap."""
    w = u['weight'] / u['weight'].sum()
    split = lambda x: (lambda px: {'common': px, 'differential': x - px, 'total': x})(project(x, u['context'], u['weight']))
    H, C = split(h), split(c)
    out = {}
    for part in ('common', 'differential', 'total'):
        c2, h2, hc = ip(C[part], C[part], w), ip(H[part], H[part], w), ip(H[part], C[part], w)
        cos = hc / np.sqrt(c2 * h2) if h2 > 0 else float('nan')
        rho = np.sqrt(h2 / c2)
        e1 = ip(H[part] - C[part], H[part] - C[part], w)
        out[part] = {'c2': c2, 'h2': h2, 'hc': hc, 'cos': cos, 'rho': rho, 'lam_star': hc / h2 if h2 > 0 else float('nan'),
                     'beta': hc / c2, 'sin2': 1 - cos * cos, 'mag2': (rho - cos) ** 2, 'E1_over_c2': e1 / c2,
                     'identity_gap': e1 / c2 - ((1 - cos * cos) + (rho - cos) ** 2), 'beneficial_rho_lt_2cos': bool(rho < 2 * cos)}
    return out


def clipped_total(pred, u):
    return risk(np.maximum(pred, 0) - u['truth'], u['context'], u['weight'])['total_mse']


def coverage_stage(out):
    full = tahoe.units('F1', 'outer')
    first = np.load(E4 / 'summary_train7_31002_L1_wd0/prediction_step40000/heldout.npz')
    have = set(zip(first['context'].astype(str), first['p_code'].astype(int)))
    keep = np.asarray([(c, int(p)) in have for c, p in zip(full['context'], full['p_code'])])
    u = {k: v[keep] for k, v in full.items()}

    def e4(n, seed, level):
        z = np.load(E4 / f'summary_train{n}_{seed}_{level}_wd0/prediction_step40000/heldout.npz')
        idx = {(c, int(p)): i for i, (c, p) in enumerate(zip(z['context'].astype(str), z['p_code']))}
        rows = [idx[(c, int(p))] for c, p in zip(u['context'], u['p_code'])]
        return z['raw'][rows].astype(np.float64), z['anchor_raw'][rows].astype(np.float64)

    def natural(seed, level):
        z = np.load(NATURAL / f'summary_{seed}_{level}_step40000/PREDICTIONS.npz')
        return z['heldout_raw'][keep].astype(np.float64), z['heldout_anchor_raw'][keep].astype(np.float64)

    sets = [(f'E4 {level}', n, level, {s: e4(n, s, level) for s in SEEDS}) for level in ('L1', 'L3') for n in (7, 14, 21)]
    sets += [(f'natural {level}', 28, level, {s: natural(s, level) for s in SEEDS}) for level in ('L1', 'L2', 'L3')]
    g = tahoe.grid_paths('summary', 'F1', SEEDS)
    load = lambda path, member='raw': grids.load(path, 'outer', u, member)
    sets.append(('OR parent L2 (nested, OR line order)', 21, 'L2', {s: (load(g[s]['nested']), load(g[s]['nested'], 'anchor_raw')) for s in SEEDS}))
    sets.append(('OR parent L2 (full)', 28, 'L2', {s: (load(g[s]['full']), load(g[s]['full'], 'anchor_raw')) for s in SEEDS}))
    a28 = sets[-1][3][SEEDS[0]][1]
    rows = []
    for name, n, level, preds in sets:
        for anchor_kind in ('own anchor', 'fixed 28-line anchor'):
            gm = {s: geometry(p - (a if anchor_kind == 'own anchor' else a28), u['truth'] - (a if anchor_kind == 'own anchor' else a28), u)
                  for s, (p, a) in preds.items()}
            aa = preds[SEEDS[0]][1] if anchor_kind == 'own anchor' else a28
            same_anchor = all(np.array_equal(preds[s][1], preds[SEEDS[0]][1]) for s in SEEDS)
            gavg = geometry(np.mean([preds[s][0] for s in SEEDS], axis=0) - aa, u['truth'] - aa, u)
            totals = [clipped_total(preds[s][0], u) for s in SEEDS]
            for part in ('common', 'differential'):
                rows.append({'set': name, 'lines': n, 'level': level, 'anchor': anchor_kind, 'part': part,
                             'cos_mean': np.mean([gm[s][part]['cos'] for s in SEEDS]),
                             'cos_seeds': ' / '.join(f"{gm[s][part]['cos']:.3f}" for s in SEEDS),
                             'rho_mean': np.mean([gm[s][part]['rho'] for s in SEEDS]),
                             'lam_star_mean': np.mean([gm[s][part]['lam_star'] for s in SEEDS]),
                             'beta_mean': np.mean([gm[s][part]['beta'] for s in SEEDS]),
                             'E1_over_c2_mean': np.mean([gm[s][part]['E1_over_c2'] for s in SEEDS]),
                             'cos_seed_avg': gavg[part]['cos'], 'lam_star_seed_avg': gavg[part]['lam_star'],
                             'c2': gm[SEEDS[0]][part]['c2'], 'outer_clipped_total_mean': np.mean(totals),
                             'anchors_identical_across_seeds': same_anchor})
    with open(Path(out) / 'COVERAGE.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bag_stats(fold, side, contexts):
    out = {}
    with h5py.File(tahoe.files(fold)['inputs'], 'r') as h:
        for c in contexts:
            x = h[side][c]['x'][:].astype(np.float64)
            out[c] = (x.mean(axis=0), np.log1p(x.var(axis=0, ddof=1)), h[side][c]['C'][:].astype(np.float64))
    return out


def per_context(u, values):
    ctx = np.unique(u['context'])
    m = np.asarray([u['weight'][u['context'] == c].sum() for c in ctx])
    v = np.stack([np.average(values[u['context'] == c], axis=0, weights=u['weight'][u['context'] == c]) for c in ctx])
    return ctx, v, m / m.sum()


def wcos(A, B, m):
    return np.sum(m[:, None] * A * B) / np.sqrt(np.sum(m[:, None] * A * A) * np.sum(m[:, None] * B * B))


def context_geometry(H, C, m):
    c2, h2, hc = (np.sum(m[:, None] * X * Y) / X.shape[1] for X, Y in ((C, C), (H, H), (H, C)))
    e1 = np.sum(m[:, None] * (H - C) ** 2) / C.shape[1]
    return {'cos': hc / np.sqrt(c2 * h2) if h2 > 0 else float('nan'), 'rho': np.sqrt(h2 / c2), 'lam_star': hc / h2 if h2 > 0 else float('nan'),
            'E1_over_c2': e1 / c2, 'Ebest_over_c2': 1 - (hc * hc / (c2 * h2) if h2 > 0 else 0.)}


def fit_predict(Xtr, Ytr, Xte, k, pen, plate_tr=None, plate_te=None):
    """Standardise on training rows, project on the top-k training principal components, ridge with an unpenalised
    intercept (per plate when plate labels are given)."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    if plate_tr is None:
        Dtr, Dte = np.ones((len(Xtr), 1)), np.ones((len(Xte), 1))
    else:
        Dtr = np.stack([plate_tr == 0, plate_tr == 1], axis=1).astype(float)
        Dte = np.stack([plate_te == 0, plate_te == 1], axis=1).astype(float)
    if k > 0:
        _, _, vt = np.linalg.svd(Ztr, full_matrices=False)
        Ftr, Fte = Ztr @ vt[:k].T, Zte @ vt[:k].T
    else:
        Ftr, Fte = np.zeros((len(Xtr), 0)), np.zeros((len(Xte), 0))
    Ymean = np.linalg.lstsq(Dtr, Ytr, rcond=None)[0]
    Fmean = np.linalg.lstsq(Dtr, Ftr, rcond=None)[0] if k > 0 else None
    Yc = Ytr - Dtr @ Ymean
    if k > 0:
        Fc = Ftr - Dtr @ Fmean
        beta = np.linalg.solve(Fc.T @ Fc + pen * np.eye(k), Fc.T @ Yc)
        return Dte @ Ymean + (Fte - Dte @ Fmean) @ beta
    return Dte @ Ymean


def ridge_stage(out, fold):
    """Training targets: per context, the weighted mean of truth - bag mean - r, with r recomputed leaving the
    context's own line out; held-out targets: the common part of truth - anchor."""
    tr, held = tahoe.units(fold, 'train'), tahoe.units(fold, 'outer')
    trctx, outctx = np.unique(tr['context']), np.unique(held['context'])
    bs_tr, bs_out = bag_stats(fold, 'train', trctx), bag_stats(fold, 'heldout', outctx)
    r_stored = np.load(tahoe.files(fold)['constants'])['r_input'].astype(np.float64)
    change = tr['truth'] - np.stack([bs_tr[c][2] for c in tr['context']])
    P = r_stored.shape[0]
    lines_tr = np.asarray([c.split('|')[0] for c in tr['context']])
    c_unit = np.empty_like(tr['truth'])
    btr = np.stack([bs_tr[c][0] for c in tr['context']])
    for l in np.unique(lines_tr):
        keep = lines_tr != l
        r_l = np.stack([np.average(change[keep & (tr['p_code'] == p)], axis=0, weights=tr['weight'][keep & (tr['p_code'] == p)])
                        if np.any(keep & (tr['p_code'] == p)) else r_stored[p] for p in range(P)])
        m = lines_tr == l
        c_unit[m] = tr['truth'][m] - btr[m] - r_l[tr['p_code'][m]]
    ctx_tr, Ytr, mtr = per_context(tr, c_unit)
    a = grids.load(tahoe.grid_paths('summary', fold, SEEDS)[31002]['full'], 'outer', held, 'anchor_raw')
    ctx_out, Yout, mout = per_context(held, held['truth'] - a)
    feats = lambda bs, ctx, kind: np.stack([np.concatenate([bs[c][0], bs[c][1]]) if kind == 'mean+var' else bs[c][0] for c in ctx])
    plate = lambda ctx: np.asarray([0 if c.endswith('plate6') else 1 for c in ctx])
    line = lambda ctx: np.asarray([c.split('|')[0] for c in ctx])
    rows = []
    for kind in ('mean', 'mean+var'):
        Xtr, Xout = feats(bs_tr, ctx_tr, kind), feats(bs_out, ctx_out, kind)
        for plate_aware in (False, True):
            ptr, pout = (plate(ctx_tr), plate(ctx_out)) if plate_aware else (None, None)
            L = line(ctx_tr)
            cv = {}
            for k in KS:
                for pen in (PENALTIES if k > 0 else (0.,)):
                    err = 0.
                    for l in np.unique(L):
                        te = L == l
                        pred = fit_predict(Xtr[~te], Ytr[~te], Xtr[te], k, pen, None if ptr is None else ptr[~te], None if ptr is None else ptr[te])
                        err += np.sum(mtr[te][:, None] * (pred - Ytr[te]) ** 2)
                    cv[(k, pen)] = err
            k, pen = min(cv, key=cv.get)
            pred = fit_predict(Xtr, Ytr, Xout, k, pen, ptr, pout)
            g = context_geometry(pred, Yout, mout)
            pl_o = plate(ctx_out)
            centred = lambda X: X - np.stack([X[pl_o == q].mean(axis=0) for q in (0, 1)])[pl_o]
            g['cos_line_specific_centred'] = wcos(centred(pred), centred(Yout), mout)
            g0 = context_geometry(fit_predict(Xtr, Ytr, Xout, 0, 0., ptr, pout), Yout, mout)
            rows.append({'fold': fold, 'features': kind, 'plate_intercepts': plate_aware, 'k': k, 'penalty': pen,
                         'cv_error_best_over_intercept_only': cv[(k, pen)] / cv[(0, 0.)],
                         **{f'reader_{x}': v for x, v in g.items()}, **{f'intercept_only_{x}': v for x, v in g0.items()}})
    pl, ln = plate(ctx_out), line(ctx_out)
    lines = np.unique(ln)
    A6 = np.stack([Yout[(ln == l) & (pl == 0)][0] for l in lines])
    A14 = np.stack([Yout[(ln == l) & (pl == 1)][0] for l in lines])
    G = Yout.shape[1]
    tot = (np.sum(A6 ** 2) + np.sum(A14 ** 2)) / (2 * len(lines) * G)
    g6, g14 = A6.mean(axis=0), A14.mean(axis=0)
    C6, C14 = A6 - g6, A14 - g14
    s2 = np.sum(C6 * C14) / (len(lines) * G) * len(lines) / (len(lines) - 1)
    resid = (np.sum(C6 ** 2) + np.sum(C14 ** 2)) / (2 * len(lines) * G) * len(lines) / (len(lines) - 1) - s2
    glob = (np.sum(g6 ** 2) + np.sum(g14 ** 2)) / (2 * G) - resid / len(lines)
    glob_shared = np.sum(g6 * g14) / G
    comp = {'fold': fold, 'total_c2': tot, 'plate_level_share': glob / tot, 'plate_level_shared_by_both_plates_share': glob_shared / tot,
            'line_reproducible_share': s2 / tot, 'residual_share': resid / tot,
            'ceiling_line_reproducible_only': float(np.sqrt(max(s2, 0) / tot)),
            'ceiling_plate_blind (shared global + line)': float(np.sqrt(max(glob_shared + s2, 0) / tot)),
            'ceiling_plate_aware (plate-level + line)': float(np.sqrt(max(glob + s2, 0) / tot)),
            'cos_g6_g14': float(np.sum(g6 * g14) / np.sqrt(np.sum(g6 ** 2) * np.sum(g14 ** 2)))}
    (Path(out) / f'RIDGE_COMMON_{fold}.json').write_text(json.dumps({'readers': rows, 'components': comp}, indent=1, default=float) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training coverage and the target's reproducibility on Tahoe-100M.")
    parser.add_argument('--stage', choices=('coverage', 'ridge'), required=True)
    parser.add_argument('--fold', default='F1', choices=('F1', 'F2', 'F3'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    coverage_stage(args.out) if args.stage == 'coverage' else ridge_stage(args.out, args.fold)


if __name__ == '__main__':
    main()
