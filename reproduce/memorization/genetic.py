"""Results 7 on the genetic screens, computed on a CPU from the saved +Var predictions of five models on ten
held-out-line folds: how far the networks store the training lines' responses, the growth-program pull, Tahoe-100M's
gene-wise readers applied unchanged, and the gene-wise shrink, on the nine metrics.

**Design.** Replogle-Nadig CRISPRi (folds 1-4, one of K562, RPE1, HepG2 and Jurkat held out, trained on the other three
lines) and Jiang et al. (folds 1-6, one of six lines held out, trained on the other five under five stimuli). A context
is a line (Replogle-Nadig) or a line under one stimulus (Jiang et al.); a unit one context and knockdown. ``b`` is the
mean of a context's input cells, ``C`` its control-pool mean and ``r`` the knockdown's average training response as the
models received it. A model's anchor is the baseline and the denominator of every ``mse``: ``b + r`` for the built
models; State and CPA store their own anchor (the same in both), which differs from ``b + r`` in a minority of entries.
Each training seed is scored alone; a seed mean is the mean of the seeds' values. Contexts are read in storage and
pull when they have at least ``MIN_UNITS`` scored knockdowns.

**Offsets and readers.** A training context's offset is the readers' target, as Tahoe-100M's readers define it: the
weighted mean over its units of the target minus the context's pool mean (from the pools the training targets were
built from) minus the knockdown's weighted mean change over every training unit (``training_offsets``). A held-out
context's offset subtracts ``r``, the average the models received, from the target minus its pool mean (stored with the
evaluation inputs). The readers are Tahoe-100M's gene-wise readers (``readers.GeneReaders``) unchanged, with no growth
reader and no lineage feature: per gene, ridge regression of the offset on the gene's own input-set mean, centred
within the centring group (the whole fold on Replogle-Nadig, the stimulus on Jiang et al.) in place of the plate, the
penalty chosen per gene by leaving one training line out.

**Storage** (``storage_slope``). A model's context-level correction (the context mean of prediction minus anchor) is
regressed without intercept on the context's offset and its sampling difference (``C`` minus ``b``), each centred over
the genes within its context, contexts and genes pooled per side (the training lines, the held-out line). Unlike
Tahoe-100M's storage (the slope on the unreadable part beside the readable part, centred per plate across contexts),
the slope here is on the whole offset: the readers have no out-of-line skill on these screens (``reader_r2``, the R² of
each training line's offset by readers fitted on the other lines, is below zero), so the offset cannot be split; and
one held-out line leaves too few contexts to centre across, so each context is centred over its genes.

**Pull** (``excess_pull``). On each Hallmark program with at least ``programs.MIN_GENES`` panel genes and on the growth
composite (``COMPOSITE``, the union of E2F targets, G2M checkpoint and mitotic spindle), the program means of a held-out
context's correction ``c``, its measured correction ``e`` (the context mean of target minus anchor) and its control
level relative to the training lines of its centring group (the input-set mean minus their mean); the excess pull is
``(e - c) sign(level)``, and the most-pulled program the single program with the largest.

**Predictions** (``arms``). Original; removal, the prediction minus its context-level correction; the gene-wise shrink,
the input-set mean plus each gene's scale (``gene_scales``) times the removal's change from it, the knockdown's target
gene keeping its removal value; removal + readers, the removal plus the readers' offset; no change, the input-set mean.
Nine metrics per unit (``UNIT_VALUES``), then the weighted mean over the held-out units; squared error in % of the
baseline's total MSE. The knockdown's target gene is kept in every metric and given weight zero only in the weighted
R²; this differs from the shared genetic-screen convention, ``scsailor.resolution.metrics.target_excluded_values``,
which also sets the target gene to the control value in the correlations, the top-50 direction and PDS.
"""
from functools import cache

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from scsailor.resolution import grids, metrics, scoring

from ..origin_resolution import ROOT, SEEDS, bag_means, heldout_units, training_units
from ..origin_resolution.other_metrics import mean_response, references
from . import data, programs, readers
from .data import SEED_MEAN
from .metrics import METRICS

__all__ = ["FOLDS", "NAMES", "MODELS", "INPUTS", "PREDICTIONS", "SYMBOLS", "MIN_UNITS", "COMPOSITE", "SHRINK",
           "UNIT_VALUES", "COMPARISONS", "line_of", "group_of", "context_rows", "training_offsets", "gene_scales",
           "arms", "storage_slope", "reader_r2", "program_masks", "excess_pull", "most_pulled", "improved", "read_fold",
           "storage_sides", "fold_data", "fit", "storage_table", "pull_table", "pull_across_lines", "values_table",
           "changes_table", "summary_table", "tables"]

FOLDS = {'replogle': ('F1', 'F2', 'F3', 'F4'), 'jiang': ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')}
NAMES = {'replogle': 'Replogle-Nadig', 'jiang': 'Jiang et al.'}   # the names the tables print
MODELS = ('summary', 'deepsets', 'attention_model', 'state', 'cpa')
# training-arrays/<fold>/ and inputs/<fold>/ of each screen
INPUTS = {'replogle': ROOT / 'replogle-nadig-inputs', 'jiang': ROOT / 'jiang-inputs'}
# <screen>/<fold>/<model>_<seed>_L2/PREDICTIONS.npz; <screen>/truth/<fold>_HELDOUT_TRUTH.npz
PREDICTIONS = ROOT / 'genetic-screens-predictions'
SYMBOLS = ROOT / 'replogle-nadig-gene-symbols/hgnc_replogle_panels.tsv'   # HGNC symbol per Ensembl identifier
ANCHOR = {'state': 'anchor', 'cpa': 'anchor'}   # the anchor's member in a model's prediction file (else 'anchor_raw')
MIN_UNITS = 5   # the scored knockdowns a context needs to be read in storage and pull
COMPOSITE = 'growth composite'
GROWTH_PROGRAMS = tuple(name.removeprefix('HALLMARK_') for name in programs.GROWTH)
SHRINK = 'gene-wise shrink'
# The nine metrics per unit. The target gene is kept in every metric (weight zero in the weighted R² only); the shared
# genetic-screen convention is metrics.target_excluded_values (module docstring)
UNIT_VALUES = metrics.unit_values
COMPARISONS = {f'{SHRINK} - original': (SHRINK, 'original'),
               'removal + readers - removal': ('removal + readers', 'removal')}
SIDES = {'train': 'training lines', 'held': 'held-out line'}
GENE_SETS = ('growth genes', 'other genes')


def line_of(screen, context):
    """The cell line of a context."""
    return context if screen == 'replogle' else context.split('_')[0]


def group_of(screen, context):
    """The centring group of a context: the whole fold (Replogle-Nadig) or its stimulus (Jiang et al.)."""
    return 'all' if screen == 'replogle' else context.split('_')[1]


def reader_keys(screen, contexts):
    """'line|group' keys, so that ``readers.GeneReaders`` reads the line and the centring group as Tahoe's line and
    plate."""
    return np.array([f'{line_of(screen, c)}|{group_of(screen, c)}' for c in contexts])


def context_rows(X, units, contexts):
    """The weighted mean of ``X`` over the units of each of ``contexts``."""
    found, means = data.context_means(X, units['context'], units['weight'])
    return means[np.searchsorted(found, contexts)]


def by_knockdown(change, units):
    """Per unit, the weighted mean of ``change`` over every unit of its knockdown."""
    codes, means = data.context_means(change, units['p_code'], units['weight'])
    return means[np.searchsorted(codes, units['p_code'])]


def training_offsets(train, pools):
    """The training contexts (sorted) and their offsets: per context, the weighted mean over its units of the change
    from the context's pool mean (``pools``: context -> mean) beyond the knockdown's weighted mean change over every
    training unit."""
    change = train['truth'] - np.stack([pools[c] for c in train['context']])
    return data.context_means(change - by_knockdown(change, train), train['context'], train['weight'])


def gene_scales(train, pools, line):
    """Per gene, the gene-wise shrink's scale: the weighted least-squares slope without intercept, over every training
    line's units (``line``: each unit's line), of the unit's change from its pool mean on the other lines' weighted mean
    change for the same knockdown, clipped to [0, 1]; units whose knockdown no other line has are left out."""
    change = train['truth'] - np.stack([pools[c] for c in train['context']])
    y, r, w = [], [], []
    for held in np.unique(line):
        rest = line != held
        codes, means = data.context_means(change[rest], train['p_code'][rest], train['weight'][rest])
        mine = np.flatnonzero(~rest & np.isin(train['p_code'], codes))
        y.append(change[mine])
        r.append(means[np.searchsorted(codes, train['p_code'][mine])])
        w.append(train['weight'][mine])
    y, r, w = np.concatenate(y), np.concatenate(r), np.concatenate(w)
    return np.clip(np.sum(w[:, None] * y * r, axis=0) / np.sum(w[:, None] * r * r, axis=0), 0, 1)


def arms(raw, anchor, control, units, scales, targets):
    """The original, the removal (``raw`` minus each unit's context mean of raw minus ``anchor``) and the gene-wise
    shrink (``control`` plus ``scales`` times the removal's change from it; at each ``targets`` (units, gene columns)
    the removal's value)."""
    removal = raw - context_rows(raw - anchor, units, units['context'])
    shrink = control + scales * (removal - control)
    shrink[targets] = removal[targets]
    return {'original': raw, 'removal': removal, SHRINK: shrink}


def within(M):
    """Each row (a context) minus its mean over the genes."""
    return M - M.mean(axis=1, keepdims=True)


def storage_slope(correction, offset, sampling):
    """The slope on the offset of the regression without intercept of the correction on the offset and the sampling
    difference (contexts x genes), each centred within its context, contexts and genes pooled."""
    X = np.stack([within(offset).ravel(), within(sampling).ravel()], axis=1)
    return float(np.linalg.lstsq(X, within(correction).ravel(), rcond=None)[0][0])


def reader_r2(offset, readable):
    """1 - the sum of squares of the unreadable residual (offset - readable part) over that of the offset, each centred
    within its context (contexts x genes)."""
    return float(1 - (within(offset - readable) ** 2).sum() / (within(offset) ** 2).sum())


def program_masks(symbols, sets):
    """Per Hallmark program of ``sets`` (name -> its gene symbols) with at least ``programs.MIN_GENES`` of ``symbols``,
    the genes it holds (named without the 'HALLMARK_' prefix); and the growth composite, the union of the growth
    programs present."""
    out = {}
    for name, members in sets.items():
        mask = np.isin(symbols, sorted(members))
        if mask.sum() >= programs.MIN_GENES:
            out[name.removeprefix('HALLMARK_')] = mask
    out[COMPOSITE] = np.any([out[k] for k in GROWTH_PROGRAMS if k in out], axis=0)
    return out


def excess_pull(c, e, level, masks):
    """Per program: (program mean of ``e`` - that of ``c``) times the sign of the program mean of ``level``."""
    return {name: (e[m].mean() - c[m].mean()) * np.sign(level[m].mean()) for name, m in masks.items()}


def most_pulled(pull):
    """The single program (not the growth composite) with the largest excess pull."""
    names = [n for n in pull if n != COMPOSITE]
    ranks = rankdata([-pull[n] for n in names])   # ties share a rank, and then no program is first
    return names[int(np.argmin(ranks))] if (ranks == 1).any() else ''


def improved(metric, changes):
    """Whether each change improves: lower squared error, higher for every other metric."""
    return (-changes if metric == 'mse' else changes) > 0


def read_fold(screen, fold):
    """One fold's files: the training and held-out units (``train``, ``held``), the panel's gene symbols (``genes``),
    the training contexts' pool means (``pools``: context -> mean, from the pools the training targets were built
    from), the input sets' and held-out pools' means (``bags``, as ``bag_means``), ``r`` and, per held-out unit, its
    knockdown's target gene column (``target``, -1 off the panel; ``TARGET_GENE_MASK.csv``, which agrees with the
    weighted-R² truth statistics where both exist), and the nine metrics' ``reference``."""
    base = INPUTS[screen]
    targets_file = base / 'training-arrays' / fold / 'train_targets.npz'
    constants = base / 'inputs' / fold / 'training_constants.npz'
    with np.load(targets_file) as z:
        genes, names = z['gene_ids'].astype(str), z['context_keys'].astype(str)
    with h5py.File(base / 'training-arrays' / fold / 'controls.h5', 'r') as h:
        pools = {names[int(k)]: h[k]['input/x'][:].mean(axis=0, dtype=np.float64) for k in h}
    with np.load(constants) as z:
        r = z['r_input'].astype(np.float64)
    if screen == 'replogle':
        table = pd.read_csv(SYMBOLS, sep='\t', dtype=str)
        symbol = dict(zip(table.ensembl_gene_id, table.symbol))
        genes = np.array([symbol.get(g, '') for g in genes])
    held = heldout_units(PREDICTIONS / screen / 'truth' / f'{fold}_HELDOUT_TRUTH.npz')
    mask_file = pd.read_csv(base / 'inputs' / fold / 'TARGET_GENE_MASK.csv')
    target = mask_file.set_index('p_code').target_coordinate.reindex(held['p_code']).to_numpy().astype(int)
    bags = bag_means(base / 'inputs' / fold / 'evaluation_inputs.h5')
    reference = references({'u': held}, {'bags': bags}, 'heldout', mean_response(constants, targets_file), target)
    return {'train': training_units(targets_file, constants), 'held': held, 'genes': genes, 'pools': pools,
            'bags': bags, 'r': r, 'target': target,
            'reference': reference | {'index': np.zeros(len(target), int), 'n': 1}}


def storage_sides(screen, f, masks):
    """The storage regression's sides and the readers, from ``read_fold``'s ``f``: ``storage``, side -> the contexts
    read (sorted) with their offsets and sampling differences; ``reader_r2``, (side, gene set) -> the readers'
    out-of-line R² (training lines: each line's readable part from readers fitted on the other lines; held-out line:
    from readers fitted on every training line); ``held_readers``, the readers' offset of each held-out context
    (sorted)."""
    train, held, bags = f['train'], f['held'], f['bags']
    contexts, Y = training_offsets(train, f['pools'])
    X = data.input_means(bags['train'], contexts)
    keys = reader_keys(screen, contexts)
    lines = np.array([line_of(screen, c) for c in contexts])
    readable = np.zeros_like(Y)
    for line in np.unique(lines):
        out = lines == line
        readable[out] = readers.GeneReaders(X[~out], keys[~out], Y[~out]).predict(X[out], keys[out])
    held_contexts = np.unique(held['context'])
    held_read = readers.GeneReaders(X, keys, Y).predict(data.input_means(bags['heldout'], held_contexts),
                                                        reader_keys(screen, held_contexts))
    change = held['truth'] - f['reference']['C'] - f['r'][held['p_code']]
    _, held_offsets = data.context_means(change, held['context'], held['weight'])
    held_sampling = data.pool_means(bags['heldout'], held_contexts) - data.input_means(bags['heldout'], held_contexts)
    sides = {'train': (contexts, Y, readable, np.stack([f['pools'][c] for c in contexts]) - X, train),
             'held': (held_contexts, held_offsets, held_read, held_sampling, held)}
    storage, r2 = {}, {}
    for side, (ctx, off, read, sampling, units) in sides.items():
        found, n = np.unique(units['context'], return_counts=True)
        keep = np.isin(ctx, found[n >= MIN_UNITS])
        storage[side] = {'contexts': ctx[keep], 'offset': off[keep], 'sampling': sampling[keep]}
        for genes_name, mask in zip(GENE_SETS, (masks[COMPOSITE], ~masks[COMPOSITE])):
            r2[(side, genes_name)] = reader_r2(off[keep][:, mask], read[keep][:, mask])
    return {'storage': storage, 'reader_r2': r2, 'held_readers': held_read}


@cache
def fold_data(screen, fold):
    """One fold's units and the quantities every fit's readings share, by key:

    * ``train``, ``held``: the training and held-out units; ``line``: the held-out line;
    * ``masks``: the programs on the panel (``program_masks``), the growth composite among them;
    * ``storage``, ``reader_r2``: as ``storage_sides``;
    * ``level``: held-out context -> its control level relative to the training lines of its centring group;
    * ``control``, ``readers``, ``scales``, ``targets``, ``reference``: per held-out unit its input-set mean and the
      readers' offset, the gene-wise shrink's scales, the target genes (units, columns) and the nine metrics'
      references; ``no_change``: the nine metrics of the input-set mean (``scored``)."""
    f = read_fold(screen, fold)
    train, held, bags, target = f['train'], f['held'], f['bags'], f['target']
    masks = program_masks(f['genes'], programs.gene_sets(programs.HALLMARK.read_text()))
    sides = storage_sides(screen, f, masks)
    held_contexts = np.unique(held['context'])
    training_contexts = np.unique(train['context'])
    level = {c: bags['heldout'][c][0] - np.mean([bags['train'][t][0] for t in training_contexts
                                                  if group_of(screen, t) == group_of(screen, c)], axis=0)
             for c in held_contexts}
    control = data.input_means(bags['heldout'], held['context'])
    with_target = np.flatnonzero(target >= 0)
    return {'train': train, 'held': held, 'line': line_of(screen, held_contexts[0]), 'masks': masks,
            'storage': sides['storage'], 'reader_r2': sides['reader_r2'], 'level': level, 'control': control,
            'readers': sides['held_readers'][np.searchsorted(held_contexts, held['context'])],
            'scales': gene_scales(train, f['pools'], np.array([line_of(screen, c) for c in train['context']])),
            'targets': (with_target, target[with_target]), 'reference': f['reference'],
            'no_change': scored(control, held, f['reference'])}


def scored(P, units, reference):
    """The nine metrics of a prediction (``UNIT_VALUES``), each the weighted mean over the units (one 'line',
    ``reference['n'] == 1``)."""
    found = metrics.per_line(UNIT_VALUES(P, units, reference), units, reference)
    return {m: metrics.point(pair) for m, pair in found.items()}


def side_slopes(d, side, units, raw, anchor):
    """(side, gene set) -> the storage slope of a prediction on one side's contexts (``units``: the side's units), and
    its corrections there."""
    s = d['storage'][side]
    c = context_rows(raw - anchor, units, s['contexts'])
    masks = (d['masks'][COMPOSITE], ~d['masks'][COMPOSITE])
    slopes = {(side, g): storage_slope(c[:, m], s['offset'][:, m], s['sampling'][:, m])
              for g, m in zip(GENE_SETS, masks)}
    return slopes, c


@cache
def fit(screen, fold, model, seed):
    """One fit's readings: ``storage``, (side, gene set) -> the storage slope; ``pull``, held-out context -> the growth
    composite's correction and measured correction and the most-pulled program; ``values``, prediction -> the nine
    metrics (squared error in % of the baseline's total MSE)."""
    d = fold_data(screen, fold)
    path = PREDICTIONS / screen / fold / f'{model}_{seed}_L2' / 'PREDICTIONS.npz'

    def load(side, units):
        return grids.load(path, side, units), grids.load(path, side, units, ANCHOR.get(model, 'anchor_raw'))

    storage, _ = side_slopes(d, 'train', d['train'], *load('train', d['train']))
    held = d['held']
    raw, anchor = load('outer', held)
    found, c = side_slopes(d, 'held', held, raw, anchor)
    storage |= found
    contexts, composite = d['storage']['held']['contexts'], d['masks'][COMPOSITE]
    e = context_rows(held['truth'] - anchor, held, contexts)
    pull = {k: {'growth_correction': ck[composite].mean(), 'measured_growth_correction': ek[composite].mean(),
                'most_pulled': most_pulled(excess_pull(ck, ek, d['level'][k], d['masks']))}
            for k, ck, ek in zip(contexts, c, e)}
    predictions = arms(raw, anchor, d['control'], held, d['scales'], d['targets'])
    predictions['removal + readers'] = predictions['removal'] + d['readers']
    values = {name: scored(P, held, d['reference']) for name, P in predictions.items()}
    values['no change'] = dict(d['no_change'])
    scale = 100 / scoring.scores(anchor, held)['total_mse']
    for v in values.values():
        v['mse'] *= scale
    return {'storage': storage, 'pull': pull, 'values': values}


def cases():
    """Every screen, fold and model, with the fold's table keys."""
    for screen, folds in FOLDS.items():
        for fold in folds:
            info = {'screen': NAMES[screen], 'fold': fold, 'line': fold_data(screen, fold)['line']}
            for model in MODELS:
                yield screen, fold, model, {**info, 'model': data.MODELS[model]}


def storage_table():
    """One row per screen, fold, model, seed (and the seed mean), side and gene set: the storage slope (the correction
    on the offset), the contexts read, and the readers' out-of-line R² (the same for every model)."""
    rows = []
    for screen, fold, model, info in cases():
        d = fold_data(screen, fold)
        per_seed = {s: fit(screen, fold, model, s)['storage'] for s in SEEDS}
        for side, name in SIDES.items():
            for genes_name in GENE_SETS:
                key = {**info, 'side': name, 'genes': genes_name, 'contexts': len(d['storage'][side]['contexts']),
                       'reader_r2': d['reader_r2'][(side, genes_name)]}
                values = [per_seed[s][(side, genes_name)] for s in SEEDS]
                rows += [{**key, 'seed': s, 'slope': v} for s, v in zip(SEEDS, values)]
                rows.append({**key, 'seed': SEED_MEAN, 'slope': float(np.mean(values))})
    return pd.DataFrame(rows)


def pull_table():
    """One row per screen, fold, model, held-out context and seed (and the seed mean): on the growth composite, the
    context's control level relative to the training lines, the model's correction and the measured correction; per
    seed the most-pulled program."""
    rows = []
    for screen, fold, model, info in cases():
        d = fold_data(screen, fold)
        per_seed = {s: fit(screen, fold, model, s)['pull'] for s in SEEDS}
        for k in d['storage']['held']['contexts']:
            key = {**info, 'context': k, 'growth_level': d['level'][k][d['masks'][COMPOSITE]].mean()}
            for s in SEEDS:
                rows.append({**key, 'seed': s, **per_seed[s][k]})
            rows.append({**key, 'seed': SEED_MEAN, **{c: float(np.mean([per_seed[s][k][c] for s in SEEDS]))
                                                    for c in ('growth_correction', 'measured_growth_correction')}})
    return pd.DataFrame(rows)


def pull_across_lines(pull):
    """One row per model: on Replogle-Nadig's four held-out lines, the Spearman correlation of the seed-mean growth
    correction, and of the measured correction, with the line's control growth level; from ``pull_table`` as
    written."""
    p = data.as_written(pull, dtype={'seed': str})
    p = p[(p.screen == NAMES['replogle']) & (p.seed == SEED_MEAN)]
    return pd.DataFrame([{'model': model, 'lines': len(x),
                          'spearman_correction': spearmanr(x.growth_correction, x.growth_level).statistic,
                          'spearman_measured': spearmanr(x.measured_growth_correction, x.growth_level).statistic}
                         for model, x in p.groupby('model', sort=False)])


def values_table():
    """One row per screen, fold, model, seed and prediction: the nine metrics (squared error in % of the baseline's
    total MSE)."""
    return pd.DataFrame([{**info, 'seed': s, 'prediction': name, **v}
                         for screen, fold, model, info in cases() for s in SEEDS
                         for name, v in fit(screen, fold, model, s)['values'].items()])


def changes_table():
    """One row per screen, fold, model, seed, comparison (``COMPARISONS``) and metric: the first prediction's value
    minus the second's (squared error in points of the baseline's total MSE)."""
    rows = []
    for screen, fold, model, info in cases():
        for s in SEEDS:
            v = fit(screen, fold, model, s)['values']
            rows += [{**info, 'seed': s, 'comparison': name, 'metric': m, 'change': v[first][m] - v[second][m]}
                     for name, (first, second) in COMPARISONS.items() for m in METRICS]
    return pd.DataFrame(rows)


def summary_table(changes, values):
    """One row per comparison and metric, over the 50 model x fold cases: the cases whose seed-mean change improves
    (``improved``), the median seed-mean change, and the medians of the two predictions' seed-mean values; from
    ``changes_table`` and ``values_table`` as written."""
    keys = ['screen', 'fold', 'model']
    c = data.as_written(changes).groupby(keys + ['comparison', 'metric'], sort=False, as_index=False).change.mean()
    v = data.as_written(values).groupby(keys + ['prediction'], sort=False)[list(METRICS)].mean()
    rows = []
    for (name, metric), x in c.groupby(['comparison', 'metric'], sort=False):
        first, second = COMPARISONS[name]
        rows.append({'comparison': name, 'metric': metric, 'cases': len(x),
                     'cases_improved': int(improved(metric, x.change).sum()), 'median_change': x.change.median(),
                     'median_first': v.xs(first, level='prediction')[metric].median(),
                     'median_second': v.xs(second, level='prediction')[metric].median()})
    return pd.DataFrame(rows)


def tables():
    """(file name, table) of every genetic-screen table, in dependency order."""
    yield 'STORAGE.csv', storage_table()
    pull = pull_table()
    yield 'GROWTH_PULL.csv', pull
    yield 'PULL_ACROSS_LINES.csv', pull_across_lines(pull)
    values, changes = values_table(), changes_table()
    yield 'NINE_METRICS_VALUES.csv', values
    yield 'NINE_METRICS_CHANGES.csv', changes
    yield 'NINE_METRICS_SUMMARY.csv', summary_table(changes, values)
