"""Results 7 on Parse PBMC, computed on a CPU from the saved Mean-level predictions: the readers of a held-out donor's
own control cells in place of a network's context-level correction, scored by total MSE and on the nine metrics. The
interferon readings are in ``parse_interferon``.

**Design.** Twelve donors in three donor groups (G1: Donors 1, 4, 7, 10; G2: 2, 5, 8, 11; G3: 3, 6, 9, 12); each group's
four donors are held out from fits trained on the other eight. A context is one donor's one cell type (``Donor3|CD14
Mono``; CD14 monocytes, CD4 memory and CD4 naive T cells), a unit one context and cytokine. ``b`` is the mean of a
context's 256 input cells (wells H7-H9), ``C`` its control-pool mean, ``r`` the cytokine's average training response as
the models received it, and ``A = b + r`` the anchor. Predictions are clipped at zero before scoring; each training seed
is scored alone and a seed mean is the mean of the seeds' values.

**Readers** (``group``). A training context's offset is the weighted mean over its units of the target minus ``C``
minus ``r`` (``storage.context_levels``); the readers are Tahoe-100M's gene-wise readers (``readers.GeneReaders``) with
the cell type in place of the plate and the donor in place of the line: per gene, ridge regression of the offset on the
gene's own input-set mean, centred within cell type, the penalty chosen per gene by leaving one training donor out.
Parse has no growth or lineage feature.

**Arms** (``arms``). With ``c`` each held-out unit's context mean of ``P - A`` and ``q = P - A - c`` (``split``):
readers ``A + q + H``; for the summary-statistics model and DeepSets also calibrated ``A + lam c + mu q`` and calibrated
+ readers ``A + mu q + H``.

**Calibration amounts** (``amounts``). Chosen on the training units of the group's two calibration donors with their
training weights (not recomputed for the subset, unlike ``scoring.retained``), from the nested parents: +Var fits of
40,000 updates on the group's six nested-fit donors, whose amounts are then applied to the Mean-level fits. The rule
differs from the shared one (``scsailor.resolution.menu.pick_two``: a pair per seed, on ``policies.G`` = {0, 1/8, 1/4,
1/3, 1/2, 3/4, 1}, ties to the larger amounts) in three choices, each named by one constant: ``GRID`` (0-1.5 in steps of
1/8), ``TIES_TO_SMALLER`` and ``ONE_PAIR_FROM_SEED_MEAN`` (one pair per model and group, picked on the mean of the
three seeds' nested-parent predictions and used for every seed). ``arms`` writes the calibrated prediction out rather
than calling ``menu.two_part``, whose context projection rounds differently.

**Error.** A unit's error is its weight times the gene mean of the squared clipped error (``unit_errors``); a group's
total MSE sums its units, the pooled total its twelve donors'. Pooled ratios are each seed's ratio of donor sums, then
the seed mean, with 95% intervals from ``DRAWS`` resamples of the twelve donors (``donor_picks``, the same draws for
every seed).

**Nine metrics** (``scsailor.resolution.metrics``) per unit, summed by weight per donor: references C (the pool mean)
and R = C + the training cytokines' mean input response. Gains are first minus second, positive favouring the first;
squared error in % of the original's total (``scope_gain``). 90% intervals: ``METRIC_DRAWS`` resamples of the scope's
donors (one group's four, or all twelve), the same draws for both predictions and every seed.

Parse PBMC's data and the tables computed from them are CC BY-NC 4.0.
"""
from functools import cache

import numpy as np
import pandas as pd

from scsailor.resolution import grids, metrics
from scsailor.resolution.policies import lowest

from ..origin_resolution import ROOT, SEEDS, bag_means, heldout_units, training_units
from ..origin_resolution import parse as binding
from ..origin_resolution.other_metrics import mean_response, references
from . import data, readers, storage
from .data import SEED_MEAN
from .metrics import METRICS

__all__ = ["GROUPS", "DONORS", "MODELS", "CALIBRATED", "LADDER", "GRID", "TIES_TO_SMALLER", "ONE_PAIR_FROM_SEED_MEAN",
           "DONOR_SEED", "DRAWS", "METRIC_SEED", "METRIC_DRAWS", "COMPARISONS", "donors", "cell_types", "split",
           "arms", "calibration_error", "pick_amounts", "unit_errors", "donor_picks", "pooled_ratio", "group",
           "prediction", "amounts", "fit", "groups_table", "pooled_table", "scope_gain", "metric_gains_table",
           "metric_summary", "tables"]

GROUPS = binding.FOLDS
DONORS = tuple(f'Donor{k}' for k in range(1, 13))
MODELS = ('summary', 'deepsets', 'attention_model', 'state', 'cpa')
CALIBRATED = ('summary', 'deepsets')   # the models with nested parents, calibrated on the calibration donors
LADDER = ROOT / 'parse-pbmc-ladder'   # <group>/<model>_<seed>_L1.npz: the Mean-level fits
# The calibration rule's three choices (module docstring); each differs from menu.pick_two's
GRID = np.arange(0, 1.501, 0.125)   # the amounts tried for both scales; menu.pick_two's: policies.G
TIES_TO_SMALLER = True   # a tie goes to the smaller amounts; menu.pick_two's to the larger
ONE_PAIR_FROM_SEED_MEAN = True   # one pair from the seeds' mean prediction, for every seed; menu.pick_two's: per seed
DONOR_SEED = 20261111   # the donor resamples of the 95% intervals (``donor_picks``)
DRAWS = 2000   # the resamples of every 95% interval: of donors, or within a context of cytokines and reference cells
METRIC_SEED, METRIC_DRAWS = 20261112, 10000   # the donor resamples of the nine metrics' 90% intervals
COMPARISONS = {'readers - original': ('readers', 'original'),
               'calibrated + readers - calibrated': ('calibrated + readers', 'calibrated')}

donors = data.lines   # the donor of each context key ('donor|cell type')
cell_types = data.plates   # the cell type of each context key


def split(P, A, units):
    """A prediction's change from the anchor in two parts: ``c``, each unit's weighted mean over its context's units of
    ``P - A`` (the context-level correction), and ``q = P - A - c``."""
    contexts, means = data.context_means(P - A, units['context'], units['weight'])
    c = means[np.searchsorted(contexts, units['context'])]
    return c, P - A - c


def arms(P, A, units, H, amounts=None):
    """The original and the readers' arm of a prediction ``P`` with anchor ``A`` and the readers' offset of each unit's
    context ``H``; given ``amounts`` (context scale, perturbation scale), also calibrated and calibrated + readers."""
    c, q = split(P, A, units)
    out = {'original': P, 'readers': A + q + H}
    if amounts is not None:
        lam, mu = amounts
        out |= {'calibrated': A + lam * c + mu * q, 'calibrated + readers': A + mu * q + H}
    return out


def calibration_error(P, units):
    """The weighted mean over units of the gene mean of the squared clipped error."""
    return float(np.average(((np.maximum(P, 0) - units['truth']) ** 2).mean(axis=1), weights=units['weight']))


def pick_amounts(P, A, units):
    """The (context scale, perturbation scale) on ``GRID`` x ``GRID`` with the lowest calibration error of ``A + lam c +
    mu q``; a tie goes to the smaller amounts (``TIES_TO_SMALLER``) or the larger."""
    c, q = split(P, A, units)

    def error(pair):
        return calibration_error(A + pair[0] * c + pair[1] * q, units)

    order = GRID if TIES_TO_SMALLER else GRID[::-1]
    (lam, mu), _ = lowest(error, [(lam, mu) for lam in order for mu in order])
    return float(lam), float(mu)


def unit_errors(P, units):
    """Each unit's weight times the gene mean of its squared clipped error."""
    return units['weight'] * ((np.maximum(P, 0) - units['truth']) ** 2).mean(axis=1)


def donor_picks():
    """The donor resamples: ``DRAWS`` draws of the twelve donors (indices into ``DONORS``) with replacement."""
    return np.random.default_rng(DONOR_SEED).integers(0, len(DONORS), size=(DRAWS, len(DONORS)))


def pooled_ratio(numerator, denominator, picks):
    """The seed mean of 100 x each seed's ratio of donor sums (seeds x donors), and its 95% interval: each draw of
    ``picks`` resamples the donors, the same draw for every seed, and takes the seed mean of the resampled ratios."""
    value = np.mean([100 * n.sum() / d.sum() for n, d in zip(numerator, denominator)])
    draws = np.mean([100 * n[picks].sum(axis=1) / d[picks].sum(axis=1) for n, d in zip(numerator, denominator)], axis=0)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(value), float(low), float(high)


@cache
def group(g):
    """One donor group's units and the quantities the readings share:

    * ``held``: the held-out units (``truth``, ``context``, ``p_code``, ``weight``), with ``anchor`` A = b + r, the
      context index ``index`` into ``contexts`` and the donor index ``donor`` into ``DONORS``;
    * ``contexts`` and ``train_contexts`` (sorted); per context ``b`` and ``C`` (held-out) and ``C_train``;
    * ``readers``: the readers' offset of each held-out context, fitted on the training contexts;
    * ``reference``: the nine metrics' references on the held-out units (``other_metrics.references``), summed by
      donor across groups."""
    f = binding.files(g)
    train, held = training_units(f['train'], f['constants']), heldout_units(f['outer'])
    bags = bag_means(f['inputs'])
    with np.load(f['constants']) as z:
        r = z['r_input'].astype(np.float64)
    b_units = data.input_means(bags['train'], train['context'])
    C_units = data.pool_means(bags['train'], train['context'])
    train_contexts, Y, _ = storage.context_levels(train, C_units, b_units, r)
    contexts = np.unique(held['context'])
    b, C = data.input_means(bags['heldout'], contexts), data.pool_means(bags['heldout'], contexts)
    C_train = data.pool_means(bags['train'], train_contexts)
    H = readers.GeneReaders(data.input_means(bags['train'], train_contexts), train_contexts, Y).predict(b, contexts)
    index = np.searchsorted(contexts, held['context'])
    held |= {'anchor': b[index] + r[held['p_code']], 'index': index,
             'donor': np.array([DONORS.index(d) for d in donors(held['context'])])}
    reference = references({'u': held}, {'bags': bags}, 'heldout', mean_response(f['constants'], f['train']))
    return {'held': held, 'contexts': contexts, 'train_contexts': train_contexts, 'b': b, 'C': C, 'C_train': C_train,
            'readers': H, 'reference': reference | {'index': held['donor'], 'n': len(DONORS)}}


def prediction(g, model, seed):
    """A Mean-level fit's held-out prediction (units of ``group(g)['held']``)."""
    return grids.load(LADDER / g / f'{model}_{seed}_L1.npz', 'outer', group(g)['held'])


def amounts(g, model, seed):
    """The calibration amounts (context scale, perturbation scale) of one fit of a model with nested parents: picked
    from the seeds' mean nested-parent prediction (``ONE_PAIR_FROM_SEED_MEAN``) or from the fit's own seed."""
    return picked_amounts(g, model, SEEDS if ONE_PAIR_FROM_SEED_MEAN else (seed,))


@cache
def picked_amounts(g, model, seeds):
    """``pick_amounts`` on the training units of the group's calibration donors, from the mean of the ``seeds``'
    nested-parent predictions and the nested parents' anchor (the same in every seed's file)."""
    f = binding.files(g)
    train = training_units(f['train'], f['constants'])
    keep = np.isin(donors(train['context']), binding.order(g)['calibration'])
    u = {k: v[keep] for k, v in train.items()}   # training weights, not recomputed for the subset (unlike retained)
    paths = binding.case(model, g, outer=False)['grids']
    P = np.mean([grids.load(paths[s]['nested'], 'train', u) for s in seeds], axis=0)
    return pick_amounts(P, grids.load(paths[seeds[-1]]['nested'], 'train', u, 'anchor_raw'), u)


@cache
def fit(g, model, seed):
    """One fit's readings on its group's held-out donors: per arm (``arms``), the total error of each donor
    (``errors``) and the nine metrics' per-donor sums (``metrics``)."""
    d = group(g)
    held = d['held']
    calibrated = amounts(g, model, seed) if model in CALIBRATED else None
    versions = arms(prediction(g, model, seed), held['anchor'], held, d['readers'][held['index']], calibrated)
    return {'errors': {a: np.bincount(held['donor'], unit_errors(V, held), len(DONORS)) for a, V in versions.items()},
            'metrics': {a: metrics.per_line(metrics.unit_values(V, held, d['reference']), held, d['reference'])
                        for a, V in versions.items()}}


def groups_table():
    """One row per group, model, arm and seed (and the seed mean): the arm's total MSE in % of the original's on the
    group's held-out donors."""
    rows = []
    for g in GROUPS:
        for model in MODELS:
            per_seed = {s: fit(g, model, s)['errors'] for s in SEEDS}
            for arm in [a for a in per_seed[SEEDS[0]] if a != 'original']:
                values = [100 * per_seed[s][arm].sum() / per_seed[s]['original'].sum() for s in SEEDS]
                info = {'group': g, 'model': data.MODELS[model], 'arm': arm}
                rows += [{**info, 'seed': s, 'total_pct_of_original': v} for s, v in zip(SEEDS, values)]
                rows.append({**info, 'seed': SEED_MEAN, 'total_pct_of_original': float(np.mean(values))})
    return pd.DataFrame(rows)


def pooled_table():
    """One row per model: the readers' total MSE pooled over the twelve donors, in % of the original's, with its 95%
    donor interval."""
    picks, rows = donor_picks(), []
    for model in MODELS:
        num, den = ([sum(fit(g, model, s)['errors'][arm] for g in GROUPS) for s in SEEDS]
                    for arm in ('readers', 'original'))
        value, low, high = pooled_ratio(num, den, picks)
        rows.append({'model': data.MODELS[model], 'arm': 'readers', 'donors': len(DONORS),
                     'total_pct_of_original': value, 'ci95_low': low, 'ci95_high': high})
    return pd.DataFrame(rows)


def scopes():
    """Each group's held-out donors (indices into ``DONORS``) and all twelve, in group order ('pooled')."""
    out = {g: sorted(set(group(g)['held']['donor'])) for g in GROUPS}
    return out | {'pooled': sum(out.values(), [])}


@cache
def metric_draws(n):
    """``METRIC_DRAWS`` resamples of n donors with replacement, as each donor's count per draw."""
    picks = np.random.default_rng(METRIC_SEED).integers(0, n, size=(METRIC_DRAWS, n))
    return np.stack([np.bincount(p, minlength=n) for p in picks]).astype(np.float64)


def scope_gain(first, second, original, D, K, mse):
    """One seed's gain of the first prediction over the second on the donors ``D`` and per resample of ``K``, from one
    metric's per-donor (weighted sum, weight) of each prediction; for squared error (``mse``) in % of the original's
    total and positive when lower."""
    def point(pair):
        return pair[0][D].sum() / pair[1][D].sum()

    def resampled(pair):
        return (K @ pair[0][D]) / (K @ pair[1][D])

    gain, draws = point(first) - point(second), resampled(first) - resampled(second)
    if mse:
        scale = 100 / point(original)
        return -scale * gain, -scale * draws
    return gain, draws


def metric_gains_table():
    """One row per comparison (``COMPARISONS``), model, scope (a group or the twelve donors pooled), metric and seed
    (and the seed mean): the gain of the first prediction over the second, positive when the first is better (squared
    error in % of the original's total), and the seed mean's 90% donor interval."""
    rows = []
    for comparison, (first, second) in COMPARISONS.items():
        for model in CALIBRATED if 'calibrated' in first else MODELS:
            # per seed and arm: metric -> (weighted sum, weight) per donor, every group's donors together
            sums = {s: {a: {m: tuple(sum(fit(g, model, s)['metrics'][a][m][k] for g in GROUPS) for k in (0, 1))
                            for m in METRICS} for a in (first, second, 'original')} for s in SEEDS}
            for scope, members in scopes().items():
                D, K = np.array(members), metric_draws(len(members))
                info = {'comparison': comparison, 'model': data.MODELS[model], 'scope': scope, 'donors': len(D)}
                for m in METRICS:
                    gains, draws = zip(*(scope_gain(sums[s][first][m], sums[s][second][m], sums[s]['original'][m], D, K,
                                                    m == 'mse') for s in SEEDS))
                    rows += [{**info, 'metric': m, 'seed': s, 'gain': gain} for s, gain in zip(SEEDS, gains)]
                    low, high = np.quantile(np.mean(draws, axis=0), [.05, .95])
                    rows.append({**info, 'metric': m, 'seed': SEED_MEAN, 'gain': float(np.mean(gains)), 'ci90_low': low,
                                 'ci90_high': high})
    return pd.DataFrame(rows)


def metric_summary(gains):
    """Per comparison and metric of ``gains`` (``metric_gains_table``), over the model x group cases' seed means: the
    cases, how many gain, the median gain and how many 90% intervals lie above or below zero."""
    g = gains[(gains.seed == SEED_MEAN) & (gains.scope != 'pooled')]
    return pd.DataFrame([{'comparison': comparison, 'metric': metric, 'cases': len(x),
                          'improved': int((x.gain > 0).sum()), 'median_gain': float(x.gain.median()),
                          'interval_above_zero': int((x.ci90_low > 0).sum()),
                          'interval_below_zero': int((x.ci90_high < 0).sum())}
                         for (comparison, metric), x in g.groupby(['comparison', 'metric'], sort=False)])


def tables():
    """(file name, table) of the readers' tables on Parse PBMC, in dependency order."""
    yield 'READERS_BY_GROUP.csv', groups_table()
    yield 'READERS_POOLED.csv', pooled_table()
    gains = metric_gains_table()
    yield 'NINE_METRICS_GAINS.csv', gains
    yield 'NINE_METRICS_SUMMARY.csv', metric_summary(data.as_written(gains, dtype={'seed': str}))
