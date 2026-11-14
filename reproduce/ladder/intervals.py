"""Paired bootstrap intervals for the change from Mean to Cells on Tahoe-100M (Supplementary Note S5.1, Supplementary
Table 6), from each fit's weighted error sums per held-out cell line (``natural-scores/held-out-unit-sums/``).

Per model and fold, each line's sums (total, common, contrast) are averaged over the three seeds at Mean and at Cells;
the fold baseline's sums are the same for every model. A change is 100 x (the component's sum at Cells minus at Mean)
over the baseline's total sum. The 14 held-out lines of each fold are drawn with replacement 10,000 times; the fold
mean resamples each fold independently (seeds spawned from 20261009) and averages the folds' changes; the 5th and 95th
percentiles give the 90% interval. Each fold's generator is built anew for every model, so all models are resampled
with the same draws.
"""
import numpy as np
import pandas as pd

from .fits import FOLDS, MODELS, unit_sums

__all__ = ["counts", "change", "fold_sums", "intervals_table"]

SEED, DRAWS = 20261009, 10000


def counts(n, rng, draws=DRAWS):
    """How often each of ``n`` held-out lines is drawn, per bootstrap draw (draws x lines)."""
    index = rng.integers(0, n, size=(draws, n))
    return np.stack([np.bincount(d, minlength=n) for d in index]).astype(np.float64)


def change(mean, cells, baseline, K):
    """The common and contrast change from Mean to Cells (2 x draws) under line counts ``K`` (draws x lines), from the
    (total, common, contrast) x lines sums at Mean and Cells and the baseline's."""
    m, c, b = mean @ K.T, cells @ K.T, baseline @ K.T
    return np.stack([100 * (c[i] - m[i]) / b[0] for i in (1, 2)])


def fold_sums(rows):
    """(fold, model) -> (Mean sums, Cells sums, baseline sums) from ``fits.unit_sums`` rows, each (total, common,
    contrast) x held-out lines in sorted line order: the model's sums (``kind`` 'model') averaged over seeds, and the
    fold baseline's; a built model's own anchor rows (``fit_anchor``) do not enter."""
    fits, baseline = {}, {}
    for r in rows:
        v = np.array([float(r[c]) for c in ('total', 'common', 'contrast')])
        if r['kind'] == 'baseline':
            baseline.setdefault(r['fold'], {})[r['unit']] = v
        elif r['kind'] == 'model' and r['level'] in ('L1', 'L4'):
            fits.setdefault((r['fold'], r['model'], r['level']), {}).setdefault(int(r['seed']), {})[r['unit']] = v
    out = {}
    for fold, model in sorted({key[:2] for key in fits}):
        mean, cells = fits[(fold, model, 'L1')], fits[(fold, model, 'L4')]
        lines = sorted(mean[min(mean)])
        sums = [np.mean([np.stack([by[u] for u in lines], axis=1) for _, by in sorted(level.items())], axis=0)
                for level in (mean, cells)]
        out[(fold, model)] = (*sums, np.stack([baseline[fold][u] for u in lines], axis=1))
    return out


def intervals_table():
    """Supplementary Table 6: per model and component, the fold mean's change from Mean to Cells and its 90%
    interval."""
    sums = fold_sums(unit_sums('Tahoe'))
    folds = FOLDS['Tahoe-100M']
    spawned = dict(zip(folds, np.random.SeedSequence(SEED).spawn(len(folds))))
    rows = []
    for model, name in MODELS.items():
        points, draws = [], []
        for fold in folds:
            mean, cells, baseline = sums[(fold, model)]
            points.append(change(mean, cells, baseline, np.ones((1, mean.shape[1]))))
            draws.append(change(mean, cells, baseline, counts(mean.shape[1], np.random.default_rng(spawned[fold]))))
        point, draws = np.mean(points, axis=0), np.mean(draws, axis=0)
        for i, part in enumerate(('common', 'contrast')):
            low, high = np.quantile(draws[i], [.05, .95])
            rows.append(dict(model=name, component=part, change_from_mean=float(point[i, 0]), low=low, high=high))
    return pd.DataFrame(rows)
