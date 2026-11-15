"""Results 7 on Parse PBMC from the retrained fits of the summary-statistics model and DeepSets (Mean inputs, three
seeds, donor groups G1 and G3; ``fits.parse``): a planted interferon response carried from training donors to held-out
donors, and fits trained on input-relative targets. Terms as in ``parse`` and ``parse_interferon``; every reading is on
the interferon alpha program (``parse_interferon.ALPHA``) or the interferon genes (those of the alpha or the gamma
program), and the control level relative to training donors is ``parse_interferon.relative_level`` of the pool levels.

**Carried response** (``carrier_table``). The planted task (``planted.carrier``) changes the training targets of the
carriers, the training donors whose control cells have high interferon activity (G1: Donor3; G3: Donors 1, 4 and 10).
A fit's change is its clipped prediction minus the clipped prediction of the same-machine refit of the original recipe
with the same seed (the natural task), as a weighted mean over each context's units on the program; the seeds' changes
are averaged. The planted change of a training context is the same mean of the planted minus the original targets; the
planted size of a cell type is the mean planted change over its carrier contexts. ``carried``: on the carriers' own
contexts, the mean of change / planted change; on held-out contexts, the mean of change / the planted size of the
context's cell type, over the recipients (control level above ``parse_interferon.HIGH``) and over the other held-out
contexts.

**Input-relative fits** (``input_relative_table``). Each fit trained on input-relative targets is read beside the
same-machine refit of the original recipe with the same seed ('same-machine refit') and the original fit on the ladder
('original fit'). Per fit:

1. memory (``memory``): the context-level correction regressed on the readable part of the context's offset, the
   unreadable residual and the sampling difference (``storage``), each centred within cell type, on the interferon
   genes and on the other genes; on the training contexts the readable part comes from readers fitted without the
   context's donor (``readable_training``), on the held-out contexts from readers fitted on all eight training donors.
   On the training contexts also the same regression at program level (``storage.program_slopes``);
2. pull (``pull_slope``): the slope through the origin of the held-out contexts' correction on their control level,
   both centred within cell type; its Spearman correlation; and the same slope of the measured change (the target
   minus the anchor);
3. ratios: the mean over the interferon-high contexts of the group's resembling donors (``HIGH_DONORS``) of the
   correction over the control level, and of the measured change, against the input wells and against the reference
   wells (``parse_interferon.reference_wells``: the measured change plus the input-set level minus the reference
   cells' mean program score);
4. error (``error_parts``): total MSE and its shared and cytokine-specific parts, in % (points) of the same-machine
   refit's total MSE with the same seed.

Rows 'G1+G3': the Spearman correlation over the 24 held-out contexts of both groups between the correction and the
measured change against the reference wells, both centred within group x cell type; and G1's ratio minus G3's. The
table holds the seed means of the three seeds' rows (``seed_means``).

Parse PBMC's data and the tables computed from them are CC BY-NC 4.0.
"""
from functools import cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scsailor.resolution import grids

from ..origin_resolution import SEEDS, bag_means, training_units
from ..origin_resolution import parse as binding
from . import data, fits, parse, parse_interferon, planted, readers, storage
from .data import SEED_MEAN
from .parse_interferon import ALPHA, GAMMA, HIGH

__all__ = ["GROUPS", "MODELS", "FITS", "HIGH_DONORS", "carried", "readable_training", "memory", "pull_slope",
           "error_parts", "seed_means", "sides", "carrier_group", "carrier_table", "input_relative_table", "tables"]

GROUPS = ('G1', 'G3')
MODELS = ('summary', 'deepsets')
FITS = ('same-machine refit', 'original fit', 'input-relative')
HIGH_DONORS = {'G1': ('Donor1', 'Donor4', 'Donor10'), 'G3': ('Donor3',)}   # the held-out donors resembling the carriers
KEYS = ('group', 'model', 'fit', 'seed', 'seeds')


def centre(M, contexts):
    """``M`` (one row or value per context) minus its mean within cell type."""
    return readers.centre(M, parse.cell_types(contexts))


def carried(planted_change, change, planted, train_contexts, contexts, carriers, recipient):
    """The carried fractions (module docstring): rows 'set', 'donors', 'fraction' for the carriers' training contexts
    and the held-out recipients and other contexts. ``planted_change``: the fit's change on the training contexts;
    ``change``: on the held-out ``contexts``; ``planted``: the planted change of the training contexts; ``recipient``:
    which held-out contexts are recipients."""
    is_carrier = np.isin(parse.donors(train_contexts), sorted(carriers))
    types = parse.cell_types(train_contexts)
    size = {t: planted[is_carrier & (types == t)].mean() for t in np.unique(types)}
    size = np.array([size[t] for t in parse.cell_types(contexts)])
    rows = [{'set': 'carriers, training contexts', 'donors': ' '.join(sorted(carriers)),
             'fraction': float(np.mean(planted_change[is_carrier] / planted[is_carrier]))}]
    for name, sel in (('recipients, new', recipient), ('other new contexts', ~recipient)):
        rows.append({'set': name, 'donors': ' '.join(sorted(set(parse.donors(contexts[sel])))),
                     'fraction': float(np.mean(change[sel] / size[sel]))})
    return rows


def readable_training(Y, contexts, means):
    """The readable part of each training context's offset ``Y``: the gene-wise readers' offset of the context
    (``readers.GeneReaders``, input-set ``means``), the readers fitted on the other donors' contexts."""
    donor = parse.donors(contexts)
    R = np.zeros_like(Y)
    for d in np.unique(donor):
        out = donor == d
        R[out] = readers.GeneReaders(means[~out], contexts[~out], Y[~out]).predict(means[out], contexts[out])
    return R


def memory(c, R, E, s, mask):
    """The regression of the correction ``c`` on ``R``, ``E`` and ``s`` (``storage.slopes``), pooled over contexts and
    the genes of ``mask``: the slopes and R² (one minus the residual over the uncentred sum of squares)."""
    readable, unreadable, sampling = storage.slopes(c, R, E, s, mask)
    y = c[:, mask].ravel()
    residual = y - (readable * R[:, mask] + unreadable * E[:, mask] + sampling * s[:, mask]).ravel()
    return {'slope_unreadable': unreadable, 'slope_readable': readable, 'slope_sampling': sampling,
            'r2': 1 - residual @ residual / (y @ y)}


def pull_slope(level, correction, contexts):
    """The slope through the origin of ``correction`` on ``level`` (one value per context), both centred within cell
    type."""
    x, y = centre(level, contexts), centre(correction, contexts)
    return float((x * y).sum() / (x * x).sum())


def error_parts(P, units, index):
    """Per context (``index``: each unit's), the sums over its units of the weight times the gene mean of the squared
    shared error (the context's weighted mean of the clipped prediction's error) and of the cytokine-specific rest."""
    error = np.maximum(P, 0) - units['truth']
    _, shared = data.context_means(error, units['context'], units['weight'])
    shared = shared[index]
    return (np.bincount(index, weights=units['weight'] * (shared ** 2).mean(axis=1)),
            np.bincount(index, weights=units['weight'] * ((error - shared) ** 2).mean(axis=1)))


def seed_means(rows):
    """Per group, model and fit of ``rows`` (first-seen order) with more than one seed row: a row with seed 'seed
    mean', the number of seeds and the mean of each number of the seed rows."""
    out = []
    for key in dict.fromkeys(tuple(r.get(k) for k in KEYS[:3]) for r in rows):
        part = [r for r in rows if tuple(r.get(k) for k in KEYS[:3]) == key]
        if len(part) > 1:
            out.append({**dict(zip(KEYS[:3], key)), 'seed': SEED_MEAN, 'seeds': len(part),
                        **{k: float(np.mean([r[k] for r in part])) for k in part[0] if k not in KEYS}})
    return out


@cache
def sides(g):
    """Per side of a donor group ('train': the training units; 'heldout': ``parse.group``'s), its units (with
    ``anchor`` and context ``index``), contexts (sorted), the side of the saved predictions 'grid', and per context the
    readable part 'R', the unreadable residual 'E' and the sampling difference 's', each centred within cell type."""
    d, f = parse.group(g), binding.files(g)
    bags = bag_means(f['inputs'])
    with np.load(f['constants']) as z:
        r = z['r_input'].astype(np.float64)
    out = {}
    for side, units, grid in (('train', training_units(f['train'], f['constants']), 'train'),
                              ('heldout', d['held'], 'outer')):
        contexts, offset, sampling = storage.context_levels(units, data.pool_means(bags[side], units['context']),
                                                            data.input_means(bags[side], units['context']), r)
        b = data.input_means(bags[side], contexts)
        if side == 'train':
            index = np.searchsorted(contexts, units['context'])
            units = units | {'anchor': b[index] + r[units['p_code']], 'index': index}
            readable = readable_training(offset, contexts, b)
        else:
            readable = d['readers']
        out[side] = {'units': units, 'contexts': contexts, 'grid': grid, 'R': centre(readable, contexts),
                     'E': centre(offset - readable, contexts), 's': centre(sampling, contexts)}
    return out


def alpha(g):
    """The interferon alpha program's weights on the group's panel, the interferon genes, and each held-out context's
    control level relative to training donors on the program."""
    d = parse.group(g)
    weights = parse_interferon.group_programs(g)[1]
    wv = weights[ALPHA]
    level = parse_interferon.relative_level(d['C'] @ wv, d['contexts'], d['C_train'] @ wv, d['train_contexts'])
    return wv, (weights[ALPHA] > 0) | (weights[GAMMA] > 0), level


def context_change(X, units, wv):
    """Per context of the units (sorted), the weighted mean of ``X``'s rows on the program with weights ``wv``."""
    return data.context_means(X, units['context'], units['weight'])[1] @ wv


def carrier_group(g, task):
    """``carried`` of each model on one donor group, from the planted carrier task ``task`` (``planted.carrier``)."""
    x = sides(g)
    wv, _, level = alpha(g)
    train = x['train']['units']
    planted = context_change(np.asarray(task['target'], np.float64) - train['truth'], train, wv)
    rows = []
    for model in MODELS:
        per_seed = []
        for seed in SEEDS:
            planted_fit, refit = (fits.parse(g, 'original', model, seed, t) for t in ('carrier', 'natural'))
            per_seed.append([context_change(np.maximum(grids.load(planted_fit, side['grid'], side['units']), 0)
                                            - np.maximum(grids.load(refit, side['grid'], side['units']), 0),
                                            side['units'], wv) for side in (x['train'], x['heldout'])])
        planted_change, new_change = (np.mean(side, axis=0) for side in zip(*per_seed))
        rows += [{'group': g, 'model': data.MODELS[model], 'seed': SEED_MEAN, 'program': ALPHA, **row}
                 for row in carried(planted_change, new_change, planted, x['train']['contexts'],
                                    x['heldout']['contexts'], task['carriers'], level > HIGH)]
    return rows


def carrier_table():
    """One row per group, model and set (``carried``): the fraction of the planted interferon alpha change that the
    fits carry, the seed mean of three seeds, on the planted carrier task (``planted.carrier``)."""
    return pd.DataFrame([row for g in GROUPS for row in carrier_group(g, planted.carrier(g))])


def fit_file(g, fit, model, seed):
    """The predictions of one fit (``FITS``) on the natural task."""
    if fit == 'original fit':
        return parse.LADDER / g / f'{model}_{seed}_L1.npz'
    return fits.parse(g, 'original' if fit == 'same-machine refit' else 'input-relative', model, seed)


def input_relative_group(g, wells):
    """One group's seed rows of the input-relative table, and per (model, fit, seed) the centred correction, the
    centred measured change against the reference wells and the predicted ratio (for the 'G1+G3' rows)."""
    d, x = parse.group(g), sides(g)
    held, contexts = d['held'], d['contexts']
    genes, weights = parse_interferon.group_programs(g)
    wv, interferon, level = alpha(g)
    measured = context_change(held['truth'] - held['anchor'], held, wv)
    reference = np.array([s.mean() for s in parse_interferon.well_scores(wells, genes, wv, contexts)])
    against_wells = measured + d['b'] @ wv - reference
    high = (level > HIGH) & np.isin(parse.donors(contexts), HIGH_DONORS[g])
    fixed = {'measured_pull_slope': pull_slope(level, measured, contexts),
             'ratio_measured_input_wells': parse_interferon.ratio(measured, level, high),
             'ratio_measured_reference_wells': parse_interferon.ratio(against_wells, level, high),
             'high_contexts': int(high.sum())}
    rows, ranked = [], {}
    for model in MODELS:
        refit = {}
        for fit in FITS:
            for seed in SEEDS:
                path = fit_file(g, fit, model, seed)
                row = {'group': g, 'model': data.MODELS[model], 'fit': fit, 'seed': seed, 'seeds': 1}
                for name, side in x.items():
                    units = side['units']
                    c = centre(data.context_means(grids.load(path, side['grid'], units) - units['anchor'],
                                                  units['context'], units['weight'])[1], side['contexts'])
                    for genes_name, mask in (('ifn', interferon), ('other', ~interferon)):
                        row |= {f'{name}_{genes_name}_{k}': v
                                for k, v in memory(c, side['R'], side['E'], side['s'], mask).items()}
                    if name == 'train':
                        row |= dict(zip(('train_program_slope_readable', 'train_program_slope_unreadable',
                                         'train_program_slope_sampling'),
                                        storage.program_slopes(c, side['R'], side['E'], side['s'], slice(None), wv)))
                P = grids.load(path, 'outer', held)
                correction = context_change(P - held['anchor'], held, wv)
                row |= {'pull_slope': pull_slope(level, correction, contexts),
                        'pull_spearman': float(spearmanr(centre(level, contexts),
                                                         centre(correction, contexts)).statistic),
                        'ratio_predicted': parse_interferon.ratio(correction, level, high), **fixed}
                shared, specific = (part.sum() for part in error_parts(P, held, held['index']))
                total = shared + specific
                if fit == 'same-machine refit':
                    refit[seed] = (total, shared, specific)
                base = refit[seed]
                row |= {'total_error': total, 'shared_error': shared, 'specific_error': specific,
                        'pct_of_same_machine_refit': 100 * total / base[0],
                        'shared_points_vs_refit': 100 * (shared - base[1]) / base[0],
                        'specific_points_vs_refit': 100 * (specific - base[2]) / base[0]}
                rows.append(row)
                ranked[(data.MODELS[model], fit, seed)] = (centre(correction, contexts),
                                                            centre(against_wells, contexts), row['ratio_predicted'])
    return rows, ranked


def input_relative_table():
    """The seed-mean rows (``seed_means``) of the input-relative readings (module docstring) per group, model and
    fit, then the 'G1+G3' rows."""
    wells = parse_interferon.reference_wells()
    rows, ranked = [], {}
    for g in GROUPS:
        group_rows, ranked[g] = input_relative_group(g, wells)
        rows += group_rows
    for key in ranked[GROUPS[0]]:
        first, second = ranked['G1'][key], ranked['G3'][key]
        rows.append({'group': 'G1+G3', 'model': key[0], 'fit': key[1], 'seed': key[2],
                     'ranking_spearman_24': float(spearmanr(np.r_[first[0], second[0]],
                                                            np.r_[first[1], second[1]]).statistic),
                     'ratio_g1_minus_g3': first[2] - second[2]})
    return pd.DataFrame(seed_means(rows))


def tables():
    """(file name, table) of the retrained Parse PBMC fits' tables."""
    yield 'PLANTED_CARRIER.csv', carrier_table()
    yield 'INPUT_RELATIVE.csv', input_relative_table()
