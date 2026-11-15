"""Target interventions on the built models (Tahoe-100M fold 1, Mean inputs, natural task): the summary-statistics,
DeepSets and attention models trained with the original recipe and with changed targets, read for storage, control-level
memory, pull, line ranking and total error; and the pull read on the training lines of the original recipe's fits.

Terms as in ``data``, ``storage`` and ``ranking``. Designs (``fits``):

* 'original': the original recipe; the same-machine refit of the model and seed when there is one (``fits.REFITS``),
  else the ladder's fit (``original``);
* 'input-relative': training targets minus each context's sampling difference (pool mean minus input-set mean);
* 'readable-only': training targets minus each training context's unreadable residual (the offset not explained by
  readers fitted on the other lines).

Per fit (``interventions_table``, three seeds each, and their means as written):

* storage on the training lines' growth genes (``train_growth_slope_unreadable``) and control-level memory at
  program level (``train_program_slope_sampling``): ``storage``'s regression of the correction on the readable part,
  the unreadable residual and the sampling difference;
* the pull on the held-out lines (``pull_slope``, ``ranking.Pull``), the line ranking (``lines_spearman``,
  ``ranking.Growth``) and the total error of the clipped prediction in % of the design-aware baseline's total
  (``total_points``), on every held-out unit.

**Pull on the training lines** (``training_pull_table``): per training context, the network's context-level correction
on the growth genes (the weighted mean over its drugs of the prediction minus the fit's own anchor), and the same of
the training targets, against the gap between the mean control-pool level of the plate's training contexts and the
context's own (``ranking.level_gap``); each read at program level with the pairs' mix (``selection.mix``) and
centred per plate. A model that reproduces its training targets has the targets' slope there, while on held-out lines
it pulls toward the training level.
"""
import numpy as np
import pandas as pd

from scsailor.resolution import grids

from ..origin_resolution import SEEDS
from . import arms, data, fits, ranking, readers, selection, storage

__all__ = ["DESIGNS", "FOLD", "COLUMNS", "original", "run_file", "with_seed_means", "reading", "interventions_table",
           "training_pull", "training_pull_table"]

FOLD = 'F1'
DESIGNS = ('original', 'input-relative', 'readable-only')
COLUMNS = ('train_growth_slope_unreadable', 'train_program_slope_sampling', 'pull_slope', 'lines_spearman',
           'total_points')


def original(model, seed):
    """The file of the original recipe's fit on fold 1's natural task: the same-machine refit when there is one, else
    the ladder's fit."""
    if seed in fits.REFITS[model]:
        return fits.tahoe('original', model, seed)
    return data.ladder_file(FOLD, model, seed)


def run_file(design, model, seed):
    """The file of one fit of the interventions table."""
    return original(model, seed) if design == 'original' else fits.tahoe(design, model, seed)


def with_seed_means(rows, keys, columns):
    """``rows`` (one per seed) with, after each group of ``keys``, a row 'seed mean': the mean of the seeds' values of
    ``columns`` as written."""
    written = data.as_written(rows)
    out = []
    for _, group in rows.groupby(keys, sort=False):
        means = written.loc[group.index, list(columns)].mean()
        out += [group, pd.DataFrame([{**group.iloc[0][keys].to_dict(), 'seed': data.SEED_MEAN, **means.to_dict()}])]
    return pd.concat(out, ignore_index=True)


def reading():
    """The reading of one fit: a function of a fit's file returning the values of ``COLUMNS``."""
    sides = storage.sides(FOLD)
    train, held = sides['training lines'], sides['held-out lines']
    G, mix = selection.mix(FOLD)
    growth, pull = ranking.growth_response(FOLD), ranking.Pull(FOLD, G, mix)
    d, D = data.fold_data(FOLD), data.design(FOLD)
    truth = d['u']['truth']
    base = arms.per_line_total(d['b'] + d['kappa'] * d['r_plate_unit'], D, truth).sum()

    def read(path):
        c = storage.correction(storage.side_prediction(path, train), train['A0'], train['u'])
        P = storage.side_prediction(path, held)
        return {'train_growth_slope_unreadable': storage.slopes(c, train['R'], train['E'], train['s'], G)[1],
                'train_program_slope_sampling': storage.program_slopes(c, train['R'], train['E'], train['s'], G,
                                                                       mix)[2],
                'pull_slope': pull(P)[0], 'lines_spearman': growth.ranking(P),
                'total_points': 100 * D.energy(np.maximum(P, 0) - truth) / base}

    return read


def interventions_table():
    """One row per model, design and seed, and the seed means (``with_seed_means``): the values of ``COLUMNS``.
    Written with ``to_csv(**data.WRITE)``."""
    read = reading()
    rows = [{'design': design, 'model': data.MODELS[m], 'seed': s, **read(run_file(design, m, s))}
            for m in storage.BUILT for design in DESIGNS for s in SEEDS]
    return with_seed_means(pd.DataFrame(rows), ['model', 'design'], COLUMNS)


def training_pull(correction, targets, level, contexts, mix):
    """The pull on training contexts (rows of each array; ``contexts`` their keys): the slope ('pull_slope') and
    correlation ('pull_r') of the correction (contexts x growth genes) on the gap of the contexts' control-pool level
    ``level`` to their plate's mean, both at program level (``mix``) and centred per plate; the same for the training
    targets' correction ``targets``; and the correlation of the two corrections ('network_targets_r')."""
    plates = data.plates(contexts)
    gap = ranking.level_gap(level, contexts, level, contexts)
    a, t, c = (readers.centre((X @ mix)[:, None], plates)[:, 0] for X in (correction, targets, gap))
    r = lambda x, y: float(np.corrcoef(x, y)[0, 1])   # noqa: E731
    return {'pull_slope': ranking.pull_slope(correction, gap, mix, plates), 'pull_r': r(a, c),
            'targets_pull_slope': ranking.pull_slope(targets, gap, mix, plates), 'targets_pull_r': r(t, c),
            'network_targets_r': r(a, t)}


def training_pull_table():
    """One row per same-machine refit of the original recipe (``fits.REFITS``): ``training_pull`` on fold 1's
    training lines, the correction taken against the fit's own anchor (its saved 'anchor_raw'). Written with
    ``to_csv(**data.WRITE)``."""
    d = data.fold_data(FOLD)
    u = d['train']['u']
    D = data.Design(u)
    contexts, first = np.unique(u['context'], return_index=True)
    G, mix = selection.mix(FOLD)
    # in C order, as in ``ranking.training_levels``
    level = np.ascontiguousarray(data.pool_means(d['bags']['train'], contexts)[:, G])

    def context_level(X):
        """Each training context's weighted mean over its units of X on the growth genes."""
        return D.project_ctx(X[:, G])[first]

    rows = []
    for m, seeds in fits.REFITS.items():
        for s in seeds:
            path = fits.tahoe('original', m, s)
            P, A = (grids.load(path, 'train', u, member) for member in ('raw', 'anchor_raw'))
            rows.append({'fold': FOLD, 'model': data.MODELS[m], 'seed': s,
                         **training_pull(context_level(P - A), context_level(u['truth'] - A), level, contexts, mix)})
    return pd.DataFrame(rows)
