"""Results 7 on Parse PBMC: the interferon programs' dependence on the training donors, read against reference control
wells that no model saw. Terms as in ``parse``.

**Programs.** The Hallmark interferon alpha and gamma programs, each the unweighted mean of its genes on the group's
panel (``program_weights``; programs with at least ``programs.MIN_GENES`` panel genes).

**Readings** (``group_readings``). A context's control level relative to training donors is its pool level minus the
training contexts' mean of the same cell type (``relative_level``). Its measured change is the weighted cytokine mean
of the target minus the anchor plus the input-set level, minus the mean program score of its 256 reference cells (wells
H10-H12; ``cell_scores``, ``measured_change``). A model's change is its context-level correction on the program, the
readers' change their offset on it. Rankings over the 36 held-out contexts are centred within group x cell type.
Ratios divide a change by the control level in the contexts whose level exceeds ``HIGH``. The measured values' 95%
intervals resample, within each context, its scored cytokines and its reference cells (``parse.DRAWS`` draws).

**Split halves** (``split_halves``). Each partition draws half of the interferon cytokines and half of the others. The
random stream restarts in every group, so every group gets the same ``SPLITS`` partitions: Donor3 (group 3) is compared
with Donors 1, 4 and 10 (group 1) partition by partition. Only contexts scored on at least half the cytokines are read.

**Program SDs** (``program_sds_table``). The SD over each group's eight training donors of a program's control-pool
level, within cell type, averaged over cell types and then over groups.

Parse PBMC's data and the tables computed from them are CC BY-NC 4.0.
"""
import json
from functools import cache

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..origin_resolution import ROOT, SEEDS
from ..origin_resolution import parse as binding
from . import data, parse, programs, readers
from .parse import DONORS, DRAWS, GROUPS, MODELS, cell_types, donors

__all__ = ["PANELS", "WELLS", "ALPHA", "GAMMA", "INTERFERON_CYTOKINES", "HIGH", "WITHIN_STREAM", "SPLIT_STREAM",
           "SPLITS", "DONOR3", "DONOR3_COMPARISON", "program_weights", "relative_level", "between_donor_sd",
           "cell_scores", "measured_change", "group_programs", "reference_wells", "well_scores", "group_readings",
           "readings", "split_halves", "ratio", "donor_values_table", "donor_ratios_table", "split_halves_table",
           "ranking_table", "control_wells_table", "program_sds_table", "tables"]

PANELS = ROOT / 'parse-pbmc-gene-panels'   # parse_outer_<group>.csv: each group's 2,000-gene panel
WELLS = ROOT / 'parse-pbmc-reference-wells'   # the reference wells' cells and the gene registry
ALPHA, GAMMA = 'HALLMARK_INTERFERON_ALPHA_RESPONSE', 'HALLMARK_INTERFERON_GAMMA_RESPONSE'
INTERFERON_CYTOKINES = frozenset({'IFN-alpha1', 'IFN-beta', 'IFN-epsilon', 'IFN-gamma', 'IFN-lambda1', 'IFN-lambda2',
                                  'IFN-lambda3', 'IFN-omega'})
HIGH = 0.25   # interferon-high: control level relative to training donors above this
# random streams [parse.DONOR_SEED, label]; the labels are arbitrary fixed numbers
WITHIN_STREAM = 12   # the within-context resamples of the measured changes
SPLIT_STREAM = 7   # the cytokine partitions of the split halves, restarted in every group
SPLITS = 200
DONOR3, DONOR3_COMPARISON = 'Donor3', ('Donor1', 'Donor4', 'Donor10')   # the split halves' two sides


def program_weights(genes, sets):
    """Per program of ``sets`` (name -> its gene symbols) with at least ``programs.MIN_GENES`` of ``genes``: weights
    that average its genes equally."""
    out = {}
    for name, members in sets.items():
        mask = np.isin(genes, sorted(members))
        if mask.sum() >= programs.MIN_GENES:
            out[name] = mask / mask.sum()
    return out


def relative_level(levels, contexts, train_levels, train_contexts):
    """Each context's level minus the mean level of the training contexts of the same cell type."""
    types = cell_types(train_contexts)
    return levels - np.array([train_levels[types == t].mean(axis=0) for t in cell_types(contexts)])


def between_donor_sd(levels, contexts):
    """The SD (ddof 1) of the levels (contexts x programs) over the donors of each cell type, averaged over cell
    types."""
    types = cell_types(contexts)
    return np.mean([levels[types == t].std(axis=0, ddof=1) for t in sorted(set(types))], axis=0)


def cell_scores(ptr, total, gene, count, to_panel, weights, start, end):
    """The program score of the cells ``start``-``end`` of a cell-by-gene count matrix (CSR rows ``ptr``, totals
    ``total``, gene and count per entry): per cell, the sum over panel genes of log1p(count x 10,000 / the cell's total)
    times the gene's program weight (``to_panel``: the panel column of each gene, -1 off the panel)."""
    n = end - start
    lo, hi = int(ptr[start]), int(ptr[end])
    cell = np.repeat(np.arange(n), np.diff(ptr[start:end + 1]))
    cols = to_panel[gene[lo:hi]]
    keep = cols >= 0
    values = np.log1p(count[lo:hi][keep] * (10000.0 / total[start:end][cell[keep]]))
    return np.bincount(cell[keep], weights=values * weights[cols[keep]], minlength=n)


def measured_change(z, weight, scores):
    """A context's measured change on a program: the weighted mean over its cytokines of ``z`` (the target minus the
    anchor plus the input-set level, on the program) minus the mean score of its reference cells."""
    return float((weight * z).sum() / weight.sum() - scores.mean())


@cache
def group_programs(g):
    """The group's panel genes and the Hallmark programs' weights on them (``program_weights``)."""
    genes = pd.read_csv(PANELS / f'parse_outer_{g}.csv').gene_id.astype(str).to_numpy()
    return genes, program_weights(genes, programs.gene_sets(programs.HALLMARK.read_text()))


def reference_wells():
    """The reference wells' cell-by-gene counts (CSR) and per context its first and last cell."""
    units = pd.read_csv(WELLS / 'UNIT_INDEX.csv')
    units = units[units.kind == 'control_reference']
    cells = {f'{d}|{t}': (s, e) for d, t, s, e in zip(units.donor, units.cell_type, units.cell_start, units.cell_end)}
    return {'ptr': np.load(WELLS / 'CELL_INDPTR.npy').astype(np.int64), 'total': np.load(WELLS / 'RAW_TOTAL.npy'),
            'gene': np.load(WELLS / 'GENE_INDEX.npy', mmap_mode='r'),
            'count': np.load(WELLS / 'RAW_COUNT.npy', mmap_mode='r'),
            'cells': cells, 'registry': pd.read_csv(WELLS / 'GENE_REGISTRY.csv')}


def well_scores(wells, genes, weights, contexts):
    """Per context, its reference cells' program scores (``cell_scores``) on a panel ``genes`` with program
    ``weights``."""
    out_index = dict(zip(wells['registry'].gene_symbol, wells['registry'].output_index))
    to_panel = np.full(len(wells['registry']), -1)
    for column, gene in enumerate(genes):
        to_panel[out_index[gene]] = column
    return [cell_scores(wells['ptr'], wells['total'], wells['gene'], wells['count'], to_panel, weights,
                        *wells['cells'][c]) for c in contexts]


def group_readings(g, wells, rng):
    """One group's interferon readings per held-out context (sorted): ``level`` and ``reference_level`` (program ->
    the control level relative to training donors, from the input pool and from the reference wells' mean scores), the
    measured alpha change ``measured`` and its within-context resamples ``draws`` (drawn from ``rng`` context by
    context), the readers' and each fit's alpha change (``readers``; ``models``: (model, seed) ->), and the split
    halves (``halves``, ``split_halves``)."""
    d = parse.group(g)
    held, contexts = d['held'], d['contexts']
    genes, weights = group_programs(g)
    level, reference_level, cells = {}, {}, {}
    for program in (ALPHA, GAMMA):
        wv = weights[program]
        level[program] = relative_level(d['C'] @ wv, contexts, d['C_train'] @ wv, d['train_contexts'])
        cells[program] = well_scores(wells, genes, wv, [*contexts, *d['train_contexts']])
        means = np.array([s.mean() for s in cells[program]])
        reference_level[program] = relative_level(means[:len(contexts)], contexts, means[len(contexts):],
                                                  d['train_contexts'])
    wv = weights[ALPHA]
    change = (held['truth'] - held['anchor']) @ wv
    z = change + (d['b'] @ wv)[held['index']]
    scores = cells[ALPHA][:len(contexts)]
    measured, draws = [], []
    for i in range(len(contexts)):
        sel = held['index'] == i
        zc, wc, rc = z[sel], held['weight'][sel], scores[i]
        pick = rng.integers(0, sel.sum(), size=(DRAWS, sel.sum()))
        rpick = rng.integers(0, len(rc), size=(DRAWS, len(rc)))
        measured.append(measured_change(zc, wc, rc))
        draws.append((wc[pick] * zc[pick]).sum(axis=1) / wc[pick].sum(axis=1) - rc[rpick].mean(axis=1))
    models = {}
    for m in MODELS:
        for s in SEEDS:
            _, correction = data.context_means(parse.prediction(g, m, s) - held['anchor'], held['context'],
                                               held['weight'])
            models[(m, s)] = correction @ wv
    return {'level': level, 'reference_level': reference_level, 'measured': np.array(measured),
            'draws': np.stack(draws), 'readers': d['readers'] @ wv, 'models': models,
            'halves': split_halves(g, change, np.array([s.mean() for s in scores]))}


@cache
def readings():
    """``group_readings`` of every group, the held-out contexts of all groups together (groups in order), with each
    context's ``group``, ``context``, ``donor`` and ``cell_type``. The within-context resamples continue one random
    stream ([parse.DONOR_SEED, WITHIN_STREAM]) across the groups."""
    wells, rng = reference_wells(), np.random.default_rng([parse.DONOR_SEED, WITHIN_STREAM])
    per = {g: group_readings(g, wells, rng) for g in GROUPS}
    contexts = np.concatenate([parse.group(g)['contexts'] for g in GROUPS])

    def joined(key):
        return np.concatenate([per[g][key] for g in GROUPS])

    def by_program(key):
        return {p: np.concatenate([per[g][key][p] for g in GROUPS]) for p in (ALPHA, GAMMA)}

    return {'group': np.concatenate([[g] * len(parse.group(g)['contexts']) for g in GROUPS]), 'context': contexts,
            'donor': donors(contexts), 'cell_type': cell_types(contexts), 'level': by_program('level'),
            'reference_level': by_program('reference_level'), 'measured': joined('measured'), 'draws': joined('draws'),
            'readers': joined('readers'),
            'models': {k: np.concatenate([per[g]['models'][k] for g in GROUPS]) for k in per[GROUPS[0]]['models']},
            'halves': {k: np.concatenate([per[g]['halves'][k] for g in GROUPS]) for k in per[GROUPS[0]]['halves']}}


def split_halves(g, change, reference):
    """(partition, half) -> per held-out context of the group ``g`` scored on at least half the cytokines, the measured
    alpha change from the half's cytokines: half 1 against the input set (the target minus the anchor), half 2 against
    the reference wells (plus the input-set level, minus the reference cells' mean score ``reference``); nan for the
    other contexts. ``change``: each unit's target minus anchor on the program. Each partition draws half of the
    interferon cytokines and half of the others from the stream [parse.DONOR_SEED, SPLIT_STREAM], restarted here so
    that every group gets the same partitions."""
    d = parse.group(g)
    held, wv = d['held'], group_programs(g)[1][ALPHA]
    names = json.loads((binding.files(g)['inputs'].parent / 'INPUT_LIBRARY.json').read_text())['p_ids']
    interferons = [k for k, n in enumerate(names) if n in INTERFERON_CYTOKINES]
    others = [k for k, n in enumerate(names) if n not in INTERFERON_CYTOKINES]
    keep = np.bincount(held['index'], minlength=len(d['contexts'])) >= len(names) / 2
    rng = np.random.default_rng([parse.DONOR_SEED, SPLIT_STREAM])
    out = {}
    for partition in range(SPLITS):
        half = set(rng.permutation(interferons)[:len(interferons) // 2])
        half |= set(rng.permutation(others)[:len(others) // 2])
        in_half = np.isin(held['p_code'], list(half))
        for k, sel in ((1, in_half), (2, ~in_half)):
            sel = sel & keep[held['index']]
            _, means = data.context_means(change[sel][:, None], held['context'][sel], held['weight'][sel])
            y = np.full(len(keep), np.nan)
            y[keep] = means[:, 0] + ((d['b'] @ wv)[keep] - reference[keep] if k == 2 else 0)
            out[(partition, k)] = y
    return out


def ratio(values, level, mask):
    """The mean over the contexts of ``mask`` of value / control level."""
    return float(np.mean(values[mask] / level[mask]))


def donor_values_table():
    """One row per held-out donor and series: the alpha change, mean over the donor's three cell types, measured (with
    its 95% interval), of each model's original (seed mean) and of the readers; with the donor's control level relative
    to training donors."""
    x, rows = readings(), []
    level = x['level'][ALPHA]
    for d in DONORS:
        m = x['donor'] == d
        info = {'donor': d, 'group': x['group'][m][0], 'contexts': int(m.sum()),
                'control_level': float(level[m].mean())}
        low, high = np.percentile(x['draws'][m].mean(axis=0), [2.5, 97.5])
        rows.append({**info, 'series': 'measured', 'value': float(x['measured'][m].mean()), 'ci95_low': low,
                     'ci95_high': high})
        rows += [{**info, 'series': f'{data.MODELS[model]} original',
                  'value': float(np.mean([x['models'][(model, s)][m] for s in SEEDS], axis=0).mean())}
                 for model in MODELS]
        rows.append({**info, 'series': 'readers', 'value': float(x['readers'][m].mean())})
    return pd.DataFrame(rows)


def donor_ratios_table():
    """One row per held-out donor with interferon-high contexts and series: the mean over those contexts of alpha
    change / control level, measured (with its 95% interval) and of the readers."""
    x, rows = readings(), []
    level = x['level'][ALPHA]
    for d in DONORS:
        m = (x['donor'] == d) & (level > HIGH)
        if not m.any():
            continue
        info = {'donor': d, 'group': x['group'][m][0], 'contexts': int(m.sum()),
                'control_level': float(level[m].mean())}
        low, high = np.percentile((x['draws'][m] / level[m, None]).mean(axis=0), [2.5, 97.5])
        rows.append({**info, 'series': 'measured', 'ratio': ratio(x['measured'], level, m), 'ci95_low': low,
                     'ci95_high': high})
        rows.append({**info, 'series': 'readers', 'ratio': ratio(x['readers'], level, m)})
    return pd.DataFrame(rows)


def split_halves_table():
    """One row per partition (``split``): in each half (half 1 against the input set, half 2 against the reference
    wells), the alpha ratio (``ratio``) over Donor3's interferon-high contexts, and over the interferon-high contexts of
    Donors 1, 4 and 10 pooled (the mean over their contexts, not of three donor means); contexts scored on at least
    half the cytokines only."""
    x, rows = readings(), []
    level = x['level'][ALPHA]
    for partition in range(SPLITS):
        row = {'split': partition}
        for k in (1, 2):
            y = x['halves'][(partition, k)]
            high = (level > HIGH) & np.isfinite(y)
            row[f'donor3_ratio_half{k}'] = ratio(y, level, high & (x['donor'] == DONOR3))
            row[f'donors_1_4_10_ratio_half{k}'] = ratio(y, level, high & np.isin(x['donor'], DONOR3_COMPARISON))
        rows.append(row)
    return pd.DataFrame(rows)


def spearman_draws(series, measured, members, picks):
    """The seed mean of the Spearman correlations of each of ``series`` (one array per seed) with ``measured``, and per
    donor draw of ``picks`` the same on the drawn donors' contexts (``members``: each donor's contexts)."""
    point = np.mean([spearmanr(v, measured).statistic for v in series])
    draws = []
    for p in picks:
        ix = np.concatenate([members[j] for j in p])
        draws.append(np.mean([spearmanr(v[ix], measured[ix]).statistic for v in series]))
    return float(point), draws


def ranking_table():
    """One row per series (each model's original, then the readers): the Spearman correlation over the 36 held-out
    contexts of the predicted with the measured alpha change, both centred within group x cell type; the seed mean of
    the seeds' correlations, with its 95% donor interval (``parse.donor_picks``)."""
    x, rows, picks = readings(), [], parse.donor_picks()
    keys = np.array([f'{g}|{t}' for g, t in zip(x['group'], x['cell_type'])])   # centring within group x cell type
    measured = readers.centre(x['measured'], keys)
    members = [np.flatnonzero(x['donor'] == d) for d in DONORS]
    series = {f'{data.MODELS[m]} original': [readers.centre(x['models'][(m, s)], keys) for s in SEEDS]
              for m in MODELS}
    for name, values in (series | {'readers': [readers.centre(x['readers'], keys)]}).items():
        point, draws = spearman_draws(values, measured, members, picks)
        low, high = np.percentile(draws, [2.5, 97.5])
        rows.append({'series': name, 'contexts': len(measured), 'spearman': point, 'ci95_low': low, 'ci95_high': high})
    return pd.DataFrame(rows)


def control_wells_table():
    """One row per interferon program: the Pearson correlation over the 36 held-out contexts of the control level
    relative to training donors from the input pool and from the reference wells."""
    x = readings()
    return pd.DataFrame([{'program': p, 'contexts': len(x['context']),
                          'pearson': float(np.corrcoef(x['level'][p], x['reference_level'][p])[0, 1])}
                         for p in (ALPHA, GAMMA)])


def program_sds_table():
    """One row per Hallmark program with at least ``programs.MIN_GENES`` panel genes in every group: the SD of its
    control-pool level over each group's eight training donors (``between_donor_sd``), averaged over the groups."""
    per = {}
    for g in GROUPS:
        d, weights = parse.group(g), group_programs(g)[1]
        names = sorted(weights)
        sd = between_donor_sd(d['C_train'] @ np.stack([weights[n] for n in names], axis=1), d['train_contexts'])
        per[g] = dict(zip(names, sd))
    common = sorted(set.intersection(*(set(v) for v in per.values())))
    return pd.DataFrame([{'program': n, 'sd_between_donors': float(np.mean([per[g][n] for g in GROUPS]))}
                         for n in common])


def tables():
    """(file name, table) of the interferon tables on Parse PBMC."""
    yield 'INTERFERON_RANKING.csv', ranking_table()
    yield 'INTERFERON_DONORS.csv', donor_values_table()
    yield 'INTERFERON_RATIOS.csv', donor_ratios_table()
    yield 'INTERFERON_SPLIT_HALVES.csv', split_halves_table()
    yield 'CONTROL_WELLS.csv', control_wells_table()
    yield 'PROGRAM_SDS.csv', program_sds_table()
