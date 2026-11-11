"""Jiang et al.: is the held-out line's context-level correction reproducible between halves of its cells? (CPU)

The models' context-level corrections on unseen Jiang lines are nearly orthogonal to the truth's (median cosine
0.03-0.08, ``diagnose_fold.py``). This asks whether that truth is itself reproducible: a high
split-half reliability reads as "barely aligned", a low one as "target not reproducible".

For each fold (one held-out line, five stimulus contexts) and each held-out unit, the QC-retained treated cells are
split at random into halves A and B (5 splits; the cells, normalization and panel of the held-out truth). The
context-level correction is e = the per-context weighted mean over units of (truth - anchor), as in the diagnostics.
Two zero points:
- bag:  anchor = clip(bag mean + r), the model's own anchor (both halves share its 256 control cells);
- rest: anchor = clip(C_rest + r), C_rest = the pool mean without the bag cells, (N C - 256 b) / (N - 256), which
        shares no cell with the bag (as in Tahoe-100M's split-half reference).
Reliability R = cosine over genes of e_A and e_B, per context and pooled with context masses:
- shared:   both halves against the bag anchor (bag noise counts as reproducible);
- disjoint: half A against the bag, half B against the rest (no cell in common; the main reading).
Each on all 2,000 genes and with the context's knocked-down target genes removed. ceiling = sqrt(2R / (1 + R)), the
Spearman-Brown full-cell reliability, is the cosine a predictor of the population correction could reach.
Groups: IFNB, IFNG and INS against TGFB and TNFA.

Entry checks: each unit's full-cell mean reproduces its saved truth, and bag mean + r reproduces the saved anchor.
The folders are arguments: the natural task's Jiang et al. bank (BANK: inputs/<fold>/, universe.csv,
folds/<fold>/panel.csv, qc/cells.csv.gz), the raw counts per stimulus (COUNTS: <stimulus>_counts.h5) and the held-out
truth (TRUTH: <fold>_HELDOUT_TRUTH.npz). Writes <out>/JIANG_SPLIT_HALF_CONTEXTS.csv and
<out>/JIANG_SPLIT_HALF_POOLED.csv.
Usage: python jiang_common_split_half.py BANK COUNTS TRUTH OUT
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

BANK = SRC = TRUTH = None   # set from the command line
FOLDS = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')
SPLITS, BAG = 5, 256
GROUPS = {'IFNB': 'IFN/INS', 'IFNG': 'IFN/INS', 'INS': 'IFN/INS', 'TGFB': 'TGFB/TNFA', 'TNFA': 'TGFB/TNFA'}


def per_context(values, contexts, weights):
    """context -> weighted mean over its units (the common part of the natural task's risk)."""
    return {c: np.average(values[contexts == c], axis=0, weights=weights[contexts == c]) for c in np.unique(contexts)}


def fold_inputs(fold):
    held = pd.read_csv(BANK/'inputs'/fold/'HELDOUT_UNITS.csv', keep_default_na=False)
    with np.load(TRUTH/f'{fold}_HELDOUT_TRUTH.npz') as z:
        if not np.array_equal(z['p_code'], held.p_code.to_numpy()):
            raise ValueError(f'{fold}: truth rows differ from HELDOUT_UNITS order')
        truth = z['truth'].astype(np.float64)
    with np.load(BANK/'inputs'/fold/'training_constants.npz') as z:
        r = z['r_input'].astype(np.float64)
    with np.load(BANK/'inputs'/fold/'BASELINE_PREDICTIONS.npz') as z:
        lookup = {(c, int(p)): i for i, (c, p) in enumerate(zip(z['side_context'].astype(str), z['p_code']))}
        saved = z['anchor_raw'][[lookup[(f'heldout/{c}', int(p))] for c, p in zip(held.context, held.p_code)]].astype(np.float64)
    pools = json.loads((BANK/'inputs'/fold/'INPUT_LIBRARY.json').read_text())['evaluation_bags']
    bag, rest = np.empty_like(truth), np.empty_like(truth)
    with h5py.File(BANK/'inputs'/fold/'evaluation_inputs.h5', 'r') as h:
        for c in held.context.unique():
            m = (held.context == c).to_numpy()
            b = h['heldout'][c]['x'][:].astype(np.float64).mean(0)
            n = pools[f'heldout/{c}']['input_pool_cells']
            c_rest = (n * h['heldout'][c]['C'][:].astype(np.float64) - BAG * b) / (n - BAG)
            bag[m] = b + r[held.p_code[m]]
            rest[m] = c_rest + r[held.p_code[m]]
    gap = float(np.abs(bag - saved).max())
    if gap > 1e-4:
        raise ValueError(f'{fold}: bag mean + r differs from the saved anchor by {gap}')
    mask = pd.read_csv(BANK/'inputs'/fold/'TARGET_GENE_MASK.csv').set_index('p_code').target_coordinate
    targets = {c: sorted({int(mask[p]) for p in held.p_code[held.context == c] if mask[p] >= 0}) for c in held.context.unique()}
    return dict(held=held, truth=truth, anchors={'bag': np.maximum(bag, 0), 'rest': np.maximum(rest, 0)},
                targets=targets, anchor_gap=gap)


def cosine(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    data = {fold: fold_inputs(fold) for fold in FOLDS}
    universe = pd.read_csv(BANK/'universe.csv').gene.to_numpy(dtype=str)
    genes = {fold: universe[pd.read_csv(BANK/'folds'/fold/'panel.csv').universe_index.to_numpy()] for fold in FOLDS}   # each fold its own panel
    cells = pd.read_csv(BANK/'qc/cells.csv.gz', keep_default_na=False, low_memory=False)
    cells = cells[cells.role.eq('treated') & cells.reason.eq('retained')]
    rng = np.random.default_rng(20261024)
    halves = {fold: {s: (np.empty_like(d['truth']), np.empty_like(d['truth'])) for s in range(SPLITS)} for fold, d in data.items()}
    truth_gap = 0.0
    for stim in sorted(GROUPS):
        with h5py.File(SRC/f'{stim}_counts.h5', 'r') as h:
            g = h['RNA/counts']
            x = sparse.csc_matrix((g['data'][:], g['indices'][:], g['indptr'][:]), shape=tuple(g['shape'][:])).T.tocsr()
            names, barcodes = g['genes'][:].astype(str), g['barcodes'][:].astype(str)
        depth = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
        col = {n: i for i, n in enumerate(names)}
        pos = {b: i for i, b in enumerate(barcodes)}
        mine = cells[cells.stimulus.eq(stim)]
        for fold, d in data.items():
            cols = np.array([col[n] for n in genes[fold]])
            for i, unit in enumerate(d['held'].itertuples()):
                if unit.stimulus != stim:
                    continue
                rows = np.array([pos[b] for b in mine.barcode[(mine.context == unit.context) & (mine.gene == unit.gene)]])
                if len(rows) != unit.n_after_cell:
                    raise ValueError(f'{unit.context} {unit.gene}: {len(rows)} cells, expected {unit.n_after_cell}')
                values = np.log1p(x[rows][:, cols].toarray().astype(np.float64) * (10000 / depth[rows][:, None]))
                truth_gap = max(truth_gap, float(np.abs(values.mean(0) - d['truth'][i]).max()))
                for s in range(SPLITS):
                    order = rng.permutation(len(rows))
                    a, b = order[:len(rows) // 2], order[len(rows) // 2:2 * (len(rows) // 2)]
                    halves[fold][s][0][i], halves[fold][s][1][i] = values[a].mean(0), values[b].mean(0)
        del x
        print('stimulus', stim, 'done; max |cell mean - truth| so far', truth_gap, flush=True)
    if truth_gap > 1e-5:
        raise ValueError(f'full-cell means differ from the saved truth by {truth_gap}')
    rows, pooled = [], []
    for fold, d in data.items():
        held = d['held']
        ctx, w = held.context.to_numpy(dtype=str), held.weight.to_numpy(dtype=np.float64)
        stim_of = dict(zip(held.context, held.stimulus))
        mass = {c: w[ctx == c].sum() for c in np.unique(ctx)}
        full = per_context(d['truth'] - d['anchors']['bag'], ctx, w)
        reads = {}   # (version, genes) -> context -> list over splits of (e_A, e_B)
        for s in range(SPLITS):
            A, B = halves[fold][s]
            eA = per_context(A - d['anchors']['bag'], ctx, w)
            eB = {'shared': per_context(B - d['anchors']['bag'], ctx, w), 'disjoint': per_context(B - d['anchors']['rest'], ctx, w)}
            for version, e2 in eB.items():
                for c in eA:
                    keep = np.ones(len(eA[c]), bool)
                    for which, k in (('all', keep), ('target_removed', np.isin(np.arange(len(keep)), d['targets'][c], invert=True))):
                        reads.setdefault((version, which), {}).setdefault(c, []).append((eA[c][k], e2[c][k]))
        for (version, which), by_context in reads.items():
            for c, pairs in by_context.items():
                R = float(np.mean([cosine(a, b) for a, b in pairs]))
                rows.append(dict(fold=fold, line=c.split('_')[0], context=c, stimulus=stim_of[c], group=GROUPS[stim_of[c]],
                                 version=version, genes=which, units=int((ctx == c).sum()),
                                 cells_median=float(np.median(held.n_after_cell[ctx == c])), R=R,
                                 ceiling=float(np.sqrt(max(2 * R / (1 + R), 0))) if R > -1 else np.nan,
                                 e_size_full=float(np.mean(full[c] ** 2)),
                                 e_size_half=float(np.mean([np.mean(a ** 2) for a, _ in pairs]))))
            for group in ('all', 'IFN/INS', 'TGFB/TNFA'):
                cs = [c for c in by_context if group == 'all' or GROUPS[stim_of[c]] == group]
                if not cs:
                    continue
                Rs = []
                for s in range(SPLITS):
                    num = sum(mass[c] * by_context[c][s][0] @ by_context[c][s][1] for c in cs)
                    aa = sum(mass[c] * by_context[c][s][0] @ by_context[c][s][0] for c in cs)
                    bb = sum(mass[c] * by_context[c][s][1] @ by_context[c][s][1] for c in cs)
                    Rs.append(num / np.sqrt(aa * bb))
                R = float(np.mean(Rs))
                pooled.append(dict(fold=fold, line=cs[0].split('_')[0], group=group, version=version, genes=which,
                                   contexts=len(cs), R=R, R_min_split=float(np.min(Rs)), R_max_split=float(np.max(Rs)),
                                   ceiling=float(np.sqrt(max(2 * R / (1 + R), 0))), anchor_gap=d['anchor_gap']))
    pd.DataFrame(rows).to_csv(out/'JIANG_SPLIT_HALF_CONTEXTS.csv', index=False)
    table = pd.DataFrame(pooled)
    table.to_csv(out/'JIANG_SPLIT_HALF_POOLED.csv', index=False)
    print(table[table.genes.eq('target_removed')].to_string(index=False), flush=True)
    print('max |cell mean - truth|', truth_gap, flush=True)


if __name__ == '__main__':
    BANK, SRC, TRUTH = (Path(p) for p in sys.argv[1:4])
    main(Path(sys.argv[4]))
