"""What the model sees: planted bits written into real control cells, reduced to one information level.

``carrier`` builds one model input from m = 256 control cells of one background:

1. ``plant`` sorts the cells by depth and overwrites genes 0-49 so that each of
   the 25 state bits sets one statistic of two genes (``design.STATISTICS``).
   Genes 50-1999 keep their real values.
2. ``level_atoms`` keeps only one level of information about the planted bag
   (``design.LEVEL_INFORMATION``).
3. Every atom is repeated twice to fill the model's 512 rows (float32).

All random choices come from the one generator passed in, in a fixed order, so
the two members of a flip pair, built with equally seeded generators, share
every draw. Values stay within [0, U], U = log(10001), as log1p(CP10k) does.
"""
from __future__ import annotations

import numpy as np

from .constants import N_CELLS, N_ROWS

U = np.log(10001.0)   # the largest log1p(CP10k) value

__all__ = ["depth_bins", "plant", "level_atoms", "carrier", "decode_bit"]


def depth_bins(sorted_depth):
    """Depth-quantile bin of each depth-sorted cell: ``m // 32`` bins (8 for 256 cells), ties kept together."""
    m = len(sorted_depth)
    k = max(1, m // 32)
    thresholds = np.unique(sorted_depth[np.ceil(np.arange(1, k) * m / k).astype(int) - 1])
    thresholds = thresholds[thresholds < sorted_depth[-1]]
    return np.searchsorted(thresholds, sorted_depth, side="left")


def plant(x, depth, state, rng, params):
    """Sort cells by depth and write the 25 bits of ``state`` (each +1 or -1) into genes 0-49 (float64)."""
    order = np.argsort(depth, kind="stable")
    y = np.asarray(x[order], dtype=np.float64)
    d = np.asarray(depth[order], dtype=np.int64)
    m = len(d)
    bins = depth_bins(d)
    for a, bit in enumerate(state):
        g, h = 2 * a, 2 * a + 1
        statistic = a // 5
        if statistic == 0:  # mean
            y[:, g] = 3 + .5 * bit
            y[:, h] = 3 - .5 * bit
        elif statistic == 1:  # variance: each gene is 3 - s in half the cells and 3 + s in the other half
            signs = np.tile(np.array([-1., 1.]), m // 2)
            u = signs[rng.permutation(m)]
            v = signs[rng.permutation(m)]
            y[:, g] = 3 + (.5 + params.alpha * bit) * u
            y[:, h] = 3 + (.5 - params.alpha * bit) * v
        elif statistic == 2:  # skewness: two four-value patterns with equal mean and variance
            values = np.array([1., 1., 1., 3.]) if bit == 1 else np.array([0., 2., 2., 2.])
            y[:, g] = 2 + .5 * np.tile(values, m // 4)[rng.permutation(m)]
            y[:, h] = 3
        elif statistic == 3:  # depth coupling: shallow half 2.5, deep half 3.5 (reversed for -1), shuffled within bins
            if len(np.unique(bins)) < 2:
                raise ValueError("Depth coupling needs at least two nonempty depth bins")
            values = np.concatenate((np.full(m // 2, 2.5), np.full(m - m // 2, 3.5)))
            if bit == -1:
                values = values[::-1]
            for b in np.unique(bins):
                ix = np.flatnonzero(bins == b)
                y[ix, g] = values[ix][rng.permutation(len(ix))]
            y[:, h] = 3
        else:  # gene pairing: within each bin both genes take one ramp, in the same or the opposite order
            for b in np.unique(bins):
                ix = np.flatnonzero(bins == b)
                if len(ix) < 2:
                    raise ValueError("Gene pairing needs at least two cells in every depth bin")
                ramp = np.linspace(2.5, 3.5, len(ix))
                pair = np.stack((ramp, ramp if bit == 1 else ramp[::-1]), axis=1)
                y[ix[:, None], [g, h]] = pair[rng.permutation(len(ix))]
    if not params.I_mean:  # no mean cue: the mean bits' genes are constant 3
        y[:, :10] = 3
    return y, d


def level_atoms(y, d, level, rng):
    """Atoms that keep only ``level``'s information about depth-sorted cells ``y`` (float64)."""
    m, genes = y.shape
    if level == "L0":
        return np.zeros_like(y)
    if level == "L1":
        return np.broadcast_to(y.mean(axis=0), y.shape).copy()
    if level == "L2":
        # Per gene: the most spread-out atoms with the bag's sum inside [0, U] (as many at U as fit,
        # one remainder, the rest 0), shrunk toward the mean until their variance is the bag's
        # (m - 1) / m * var(ddof=1). Columns are then shuffled independently.
        s = y.mean(axis=0)
        h = y.var(axis=0, ddof=1)
        t = m * s
        full = np.floor(t / U).astype(int)
        row = np.arange(m)[:, None]
        z = np.where(row < full[None, :], U, 0.0)
        z += np.where(row == full[None, :], (t - full * U)[None, :], 0.0)
        vmax = np.mean((z - s) ** 2, axis=0)
        v = (m - 1) * h / m
        ratio = np.divide(v, vmax, out=np.zeros_like(v), where=vmax > 0)
        c = s + np.sqrt(ratio) * (z - s)
        return np.take_along_axis(c, np.argsort(rng.random((m, genes)), axis=0), axis=0)
    if level in ("L3", "L3d"):
        # Each gene's values (within each depth bin for L3d) are shuffled across cells independently.
        c = np.empty_like(y)
        bins = np.zeros(m, dtype=np.int8) if level == "L3" else depth_bins(d)
        for b in np.unique(bins):
            ix = np.flatnonzero(bins == b)
            values = np.sort(y[ix], axis=0)
            c[ix] = np.take_along_axis(values, np.argsort(rng.random((len(ix), genes)), axis=0), axis=0)
        return c
    if level == "L4":
        return y
    raise ValueError(f"Unknown level {level}")


def carrier(x, depth, state, level, rng, params):
    """One model input: 512 float32 rows and their depths, from m control cells and a 25-bit state."""
    if N_ROWS % len(depth):
        raise ValueError(f"{len(depth)} cells do not divide the {N_ROWS} model rows")
    y, d = plant(x, depth, state, rng, params)
    atoms = level_atoms(y, d, level, rng)
    k = N_ROWS // len(d)
    return np.repeat(atoms, k, axis=0).astype(np.float32), np.repeat(d, k)


def decode_bit(rows, row_depth, a, m=N_CELLS):
    """Sign of bit ``a``'s planted statistic, read from one model input (its m distinct atoms)."""
    k = N_ROWS // m
    x = rows[::k].astype(np.float64)
    d = row_depth[::k]
    g, h = 2 * a, 2 * a + 1
    statistic = a // 5
    if statistic == 0:
        value = x[:, g].mean() - x[:, h].mean()
    elif statistic == 1:
        value = x[:, g].var(ddof=1) - x[:, h].var(ddof=1)
    elif statistic == 2:
        centered = x[:, g] - x[:, g].mean()
        value = (centered ** 3).mean()
    elif statistic == 3:
        value = np.cov(depth_bins(d).astype(np.float64), x[:, g], ddof=0)[0, 1]
    else:
        bins = depth_bins(d)
        value = 0.
        for b in np.unique(bins):
            v, w = x[bins == b, g], x[bins == b, h]
            value += np.mean((v - v.mean()) * (w - w.mean())) * len(v) / m
    return int(np.sign(value))
