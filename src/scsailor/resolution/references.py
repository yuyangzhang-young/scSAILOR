"""Dataset references that origin resolution reads beside a model's error, computed from loaded arrays.

A reference says how much of the error the data and the task set, whatever the model: the cell sampling of the
scored targets (D1), the discrepancy between replicate measurements of the same unit (D2) and its split into a part
shared by the units measured together and a unit-specific part, the sampling of the control pools, the reproducibility
of the true corrections between replicates, and, on a genetic screen, the GEM-group term and the knockdown test.
Energies are in the risk's units: squared error averaged over genes and weighted over units.

- ``cell_sampling`` and ``moments``: a unit's cell-sampling variance v = (1/G) sum_g s2_g / n, from its cells or from
  streamed sums; ``d1_difference`` its drug-difference floor per unit and ``line_floor`` the floor of one whole-line
  context (``attribution.floor_parts`` gives every part per unit);
- ``pool_sampling_term``: the control-pool means' sampling carried by the response version of the twin deviations;
- ``twin_deviation`` and ``well_split``: a unit's deviation from its replicate twin, (y - y') / sqrt(2), and the split of
  its drug-difference energy over the units measured in one well into a shared and a unit-specific part;
- ``pair_sampling_content`` and ``pool_decomposition``: the sampling energy that pair deviations carry, and a nested
  method-of-moments decomposition of control-pool means into campaign, plate and pool levels;
- ``gem_variance`` and ``gem_term``: the variance between GEM groups of a screen's controls, and the term it leaves in a
  unit's target and against a control mean of another GEM composition;
- ``knockdown_readout``: quintile medians of per-unit quantities along a knockdown difference, their slope and a
  stratified permutation null;
- ``split_half`` and ``reliability``: the split-half reliability of the true corrections between two replicates.

Units dictionaries hold ``context`` and ``weight`` (and ``truth``, ``p_code`` where named), in one row order.
"""
from collections import defaultdict

import numpy as np

from .scoring import project

__all__ = ["cell_sampling", "moments", "context_shares", "d1_difference", "line_floor", "pool_sampling_term",
           "twin_deviation", "well_split", "pair_sampling_content", "pool_decomposition", "gem_variance", "gem_term", "KNOCKDOWN_STEP",
           "knockdown_readout", "split_half", "reliability"]


# --------------------------------------------------------------------------- cell sampling (D1)
def cell_sampling(x):
    """A unit's cell-sampling variance from its cells x (cells x genes): v = (1/G) sum_g s2_g / n with s2_g the
    per-gene cell variance (ddof 1), and the per-gene variances."""
    variance = x.var(axis=0, ddof=1)
    return float(variance.mean() / len(x)), variance


def moments(s1, s2, n):
    """Per-row mean and cell variance (ddof 1) from sums of x and x^2 over n cells."""
    mean = s1 / n[:, None]
    return mean, (s2 - s1 * mean) / (n[:, None] - 1)


def _context_mass(context, weight):
    """Weights normalized over the units given, and each context's mass, summed unit by unit in row order."""
    w = weight / weight.sum()
    mass = defaultdict(float)
    for c, x in zip(context, w):
        mass[c] += x
    return w, mass


def context_shares(context, weight):
    """Weights normalized over the units given, w, and each unit's share of its context's mass, q = w / m_context."""
    w, mass = _context_mass(context, weight)
    return w, w / np.asarray([mass[c] for c in context])


def d1_difference(units, noise):
    """Each unit's D1 drug-difference contribution, w (1 - q) v. Exact for
    the totals over whole contexts; per drug it is an allocation, not each well's expected floor."""
    w, q = context_shares(units['context'], units['weight'])
    return w * (1 - q) * noise


def line_floor(noise, weight):
    """The D1 floor of units that form one context (a whole cell line): total sum w v, common sum w q v with
    q = w / m, weights normalized over the units."""
    w = weight / weight.sum()
    total, common = float(w @ noise), float((w * w) @ noise)
    return {'total': total, 'common': common, 'difference': total - common}


def pool_sampling_term(context, weight, sampling, partner):
    """Sum over contexts of m_c (u_c + u_c') / 2: the sampling noise of the control-pool means C and C' that the response
    version of the twin deviations carries in its common and total parts. ``sampling`` maps a context to u_c, the
    gene-averaged sampling variance of its pool mean; ``partner`` a context to its twin context."""
    _, mass = _context_mass(context, weight)
    return sum(m * (sampling[c] + sampling[partner[c]]) / 2 for c, m in mass.items())


# --------------------------------------------------------------------------- replicate twins (D2)
def twin_deviation(y, twin, pool=None):
    """Each unit's deviation from its twin, (y - y') / sqrt(2): treated, or response when per-unit control-pool means
    are given, ((y - C) - (y' - C')) / sqrt(2)."""
    if pool is None:
        return (y - y[twin]) / np.sqrt(2)
    return ((y - pool) - (y[twin] - pool[twin])) / np.sqrt(2)


def well_split(delta, weight, drug, plate, d1):
    """The well-shared split of the drug-difference deviations delta (units x genes). A well is one drug on one plate
    (``plate`` holds integer codes, iterated in increasing order) and holds every pooled unit of that drug there.
    Per well, over its L units with weights pi = w / W_well:
      mean          d_bar = sum pi delta
      between       B = |d_bar|^2 / G
      within        R = sum pi |delta - d_bar|^2 / G
      unit-specific s2 = R / (1 - sum pi^2)
      shared        B - s2 sum pi^2
    They add exactly, shared + unit-specific = the well's drug-difference energy. Under independent per-unit noise of
    equal variance both are unbiased; with unequal variances and weights the shared part is approximate. Each drug's
    row sums its wells, weighted by their share of all weight, with the D1 of its units and the equal-weight values
    beside. A well with one unit has no spread to separate: its energy and D1 are summed apart, unsplit."""
    G, W = delta.shape[1], weight.sum()
    drugs = defaultdict(lambda: defaultdict(float))
    single = {'wells': 0, 'weight_share': 0., 'energy': 0., 'd1': 0.}
    for p in np.unique(drug):
        for pl in np.unique(plate):
            m = np.flatnonzero((drug == p) & (plate == pl))
            share = weight[m].sum() / W
            if len(m) == 1:
                single['wells'] += 1
                single['weight_share'] += share
                single['energy'] += share * float(delta[m[0]] @ delta[m[0]] / G)
                single['d1'] += float(d1[m].sum())
                continue
            if not len(m):
                continue
            pi = weight[m] / weight[m].sum()
            mean = pi @ delta[m]
            between = mean @ mean / G
            within = pi @ (np.sum((delta[m] - mean) ** 2, axis=1) / G)
            s2 = within / (1 - pi @ pi)
            equal = delta[m].mean(axis=0)
            spread = np.sum((delta[m] - equal) ** 2) / (G * (len(m) - 1))
            row = drugs[int(p)]
            row['lines'] = max(row['lines'], len(m))
            row['weight_share'] += share
            row['shared'] += share * (between - s2 * (pi @ pi))
            row['line_specific'] += share * s2
            row['d1'] += float(d1[m].sum())
            row['equal_weight_shared'] += share * (equal @ equal / G - spread / len(m))
            row['equal_weight_line_specific'] += share * spread
    return drugs, single


def pair_sampling_content(own, partner, omega, ctx, var):
    """Expected sampling energy carried by pair rows (z_own - z_partner) / sqrt(2) for independent zero-mean errors z of
    gene-averaged variance var[item]: per context of the rows, (mass, total energy, common energy), the common energy
    being that of the context's weighted mean row (the risk's projection). The drug-difference energy is total -
    common."""
    per = {}
    for c in np.unique(ctx):
        r = np.flatnonzero(ctx == c)
        mass = float(omega[r].sum())
        coef = defaultdict(float)
        for i in r:
            coef[own[i]] += omega[i]
            coef[partner[i]] -= omega[i]
        per[c] = (mass, float(np.sum(omega[r] * (var[own[r]] + var[partner[r]])) / (2 * mass)),
                  sum(x * x * var[j] for j, x in coef.items()) / (2 * mass * mass))
    return per


def pool_decomposition(pools, campaigns):
    """Method-of-moments nested decomposition of one identity's control-pool means into campaign, plate within campaign
    and pool within plate (gene-averaged energies). ``pools`` holds per pool its mean ``C``, the sampling variance of
    that mean ``pool_sampling``, its ``plate``, ``campaign`` and ``pool_cells``; ``campaigns`` names the two campaigns
    in the order of the campaign contrast. Each level is net of everything below it, the pools' known sampling variance
    included. Means are unweighted: pools equal within a plate, plates equal within a campaign. Estimates are not
    truncated at zero."""
    C = np.stack([p['C'] for p in pools])
    G = C.shape[1]
    u = np.asarray([p['pool_sampling'] for p in pools])
    plate = np.asarray([p['plate'] for p in pools])
    camp = np.asarray([p['campaign'] for p in pools])
    plates = sorted(set(plate))
    mean_j, n_j, usum_j, camp_j = {}, {}, {}, {}
    s_e = e_e = 0.
    df_e = 0
    for j in plates:
        m = plate == j
        mean_j[j], n_j[j], usum_j[j], camp_j[j] = C[m].mean(axis=0), int(m.sum()), float(u[m].sum()), camp[m][0]
        s_e += float(np.sum((C[m] - mean_j[j]) ** 2) / G)
        e_e += (1 - 1 / n_j[j]) * usum_j[j]
        df_e += n_j[j] - 1
    pool_level = (s_e - e_e) / df_e if df_e else float('nan')
    tau = {j: (pool_level if df_e else 0.) / n_j[j] + usum_j[j] / n_j[j] ** 2 for j in plates}   # Var of a plate mean
    camps = sorted(set(camp))
    by_camp = {k: [j for j in plates if camp_j[j] == k] for k in camps}
    mean_k = {k: np.mean([mean_j[j] for j in js], axis=0) for k, js in by_camp.items()}
    s_b = sum(float(np.sum((mean_j[j] - mean_k[k]) ** 2) / G) for k, js in by_camp.items() for j in js)
    e_b = sum((1 - 1 / len(js)) * sum(tau[j] for j in js) for js in by_camp.values())
    df_b = sum(len(js) - 1 for js in by_camp.values())
    plate_level = (s_b - e_b) / df_b if df_b else float('nan')
    out = {'pools': len(pools), 'plates': len(plates), 'mean_pool_cells': float(np.mean([p['pool_cells'] for p in pools])),
           'mean_pool_sampling': float(u.mean()),
           **{f'{k}_{x}': v for k in campaigns for x, v in
              (('pools', int(np.isin(plate, by_camp.get(k, [])).sum())), ('plates', len(by_camp.get(k, []))))},
           'pool_df': df_e, 'pool_mean_square': s_e / df_e if df_e else None,
           'pool_sampling_expected': e_e / df_e if df_e else None, 'pool_within_plate': pool_level if df_e else None,
           'plate_df': df_b, 'plate_mean_square': s_b / df_b if df_b else None,
           'plate_lower_expected': e_b / df_b if df_b else None, 'plate_within_campaign': plate_level if df_b else None}
    if len(camps) == 2:
        raw = float(np.sum((mean_k[campaigns[0]] - mean_k[campaigns[1]]) ** 2) / G)
        lower = sum(sum((plate_level if df_b else 0.) + tau[j] for j in js) / len(js) ** 2 for js in by_camp.values())
        sampling = sum(sum(usum_j[j] / n_j[j] ** 2 for j in js) / len(js) ** 2 for js in by_camp.values())
        out |= {'campaign_raw': raw, 'campaign_lower_expected': lower, 'campaign_sampling_expected': sampling,
                'campaign_shift': raw - lower, 'campaign_shift_net_of_sampling_only': raw - sampling}
    return out


# --------------------------------------------------------------------------- GEM groups (a genetic screen)
def gem_variance(group_mean, group_var, group_n):
    """Per gene, the variance (ddof 1) of the GEM groups' control means minus the mean over groups of s2_k / n_k: the
    variance between GEM groups net of their cell sampling; its gene mean is tau2."""
    return group_mean.var(axis=0, ddof=1) - (group_var / group_n[:, None]).mean(axis=0)


def gem_term(tau2, share, reference=None):
    """A unit's GEM term from the share s_uk of its cells in each GEM group: tau2 sum_k s_uk^2 in its target, or
    tau2 sum_k (s_uk - c_k)^2 against a control mean whose cells have the shares c_k."""
    return tau2 * (share * share).sum(1) if reference is None else tau2 * ((share - reference) ** 2).sum(1)


# --------------------------------------------------------------------------- the knockdown test (T7)
KNOCKDOWN_STEP = np.arange(5) - 2.0   # quintile index, centred: the least-squares slope is medians @ STEP / 10


def qbin(values, k):
    """Quantile bin 0..k-1 of each value (equal values share a bin)."""
    return np.searchsorted(np.quantile(values, np.arange(1, k) / k), values, side='right')


def medians(q, bins):
    """Per row of q, the median within each of five bins."""
    return np.stack([np.median(q[:, bins == b], axis=1) for b in range(5)], axis=1)


def knockdown_readout(q, falls, dkd, rnorm, cells, base, rng, permutations=10000):
    """Per row of the per-unit quantities q: the medians in each quintile of the knockdown difference dkd, the
    least-squares slope of the five medians on the quintile index, and its null from ``permutations`` permutations of
    the quintile labels within strata of (||r|| quintile x cell-count tercile x tercile of the baseline-expression
    difference ``base``). One-sided: rows marked ``falls`` count null slopes at or below the observed one, the others at
    or above; p = (1 + count) / (1 + permutations)."""
    bins = qbin(dkd, 5)
    stratum = qbin(rnorm, 5) * 9 + qbin(cells, 3) * 3 + qbin(base, 3)
    med = medians(q, bins)
    slope = med @ KNOCKDOWN_STEP / 10
    by_stratum = np.argsort(stratum, kind='stable')
    null = np.empty((permutations, len(q)))
    permuted = np.empty_like(bins)
    for i in range(permutations):
        permuted[np.lexsort((rng.random(len(bins)), stratum))] = bins[by_stratum]
        null[i] = medians(q, permuted) @ KNOCKDOWN_STEP / 10
    count = np.where(falls, (null <= slope).sum(0), (null >= slope).sum(0))
    sizes = np.bincount(stratum, minlength=45)
    return {'bins': bins, 'medians': med, 'slope': slope, 'null_mean': null.mean(0), 'null_sd': null.std(0, ddof=1),
            'p': (1 + count) / (1 + permutations), 'dkd_medians': [float(np.median(dkd[bins == b])) for b in range(5)],
            'dkd_edges': np.quantile(dkd, np.arange(1, 5) / 5).tolist(), 'quintile_sizes': np.bincount(bins, minlength=5).tolist(),
            'strata': {'occupied': int((sizes > 0).sum()), 'min': int(sizes[sizes > 0].min()), 'median': float(np.median(sizes[sizes > 0])),
                       'max': int(sizes.max()), 'singletons': int((sizes == 1).sum())}}


# --------------------------------------------------------------------------- replicate reliability
def split_half(A, B, m):
    """The split-half reliability of two replicate measurements A and B (rows x genes) with row masses m: their
    uncentred cosine, sum m A B / sqrt(sum m A^2 sum m B^2)."""
    num = np.sum(m[:, None] * A * B)
    return num / np.sqrt(np.sum(m[:, None] * A * A) * np.sum(m[:, None] * B * B))


def _line_halves(common, weight, line, half):
    """Per line, the context-level rows of replicate 0 and replicate 1 (the first unit of each), with the line's mass
    over both, normalized."""
    w = weight / weight.sum()
    A, B, m = [], [], []
    for l in np.unique(line):
        i0 = np.flatnonzero((line == l) & (half == 0))
        i1 = np.flatnonzero((line == l) & (half == 1))
        A.append(common[i0[0]]); B.append(common[i1[0]]); m.append(w[i0].sum() + w[i1].sum())
    return np.stack(A), np.stack(B), np.asarray(m) / np.sum(m)


def _twin_split_half(D, weight, half, twin):
    """The unit-level split half of the drug-difference part: each replicate-0 unit with its twin, weighted by the pair's
    mean weight; with the number of pairs."""
    own = np.flatnonzero((half == 0) & (twin >= 0))
    j = twin[own]
    w = (weight[own] + weight[j]) / 2
    w = w / w.sum()
    num = np.sum(w[:, None] * D[own] * D[j])
    return num / np.sqrt(np.sum(w[:, None] * D[own] ** 2) * np.sum(w[:, None] * D[j] ** 2)), len(own)


def reliability(correction, units, line, half, twin):
    """The split-half reliability of the true corrections (truth - anchor, units x genes) between two replicates of each
    line (``half`` 0 or 1 per unit), and its square-root ceilings: the highest cosine a predictor can reach with one
    replicate's target. Cell-line-level part (the per-context weighted mean, the risk's projection): R between the
    replicates' context rows over lines, sqrt(R), and Spearman-Brown for both replicates together, 2R / (1 + R), with
    its square root. Drug-difference part (the rest): R between unit twins (``twin``: each unit's twin in the other
    replicate, -1 where missing), sqrt(R), the pairs and Spearman-Brown."""
    common = project(correction, units['context'], units['weight'])
    A, B, m = _line_halves(common, units['weight'], line, half)
    R = split_half(A, B, m)
    Rd, pairs = _twin_split_half(correction - common, units['weight'], half, twin)
    return {'common': {'split_half_R_single_plate': float(R), 'sqrt_R': float(np.sqrt(max(R, 0))),
                       'spearman_brown_R_two_plates': float(2 * R / (1 + R)), 'sqrt_R_SB': float(np.sqrt(max(2 * R / (1 + R), 0)))},
            'differential': {'split_half_R_unit_twins': float(Rd), 'sqrt_R': float(np.sqrt(max(Rd, 0))), 'pairs': int(pairs),
                             'spearman_brown': float(2 * Rd / (1 + Rd))}}
