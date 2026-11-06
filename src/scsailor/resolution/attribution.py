"""Where a prediction's error sits, and the references that add like the error.

Read-outs on per-unit contributions (``scoring.unit_parts``), all descriptive unless a test is named:

- ``where``: per group of units (identity, context, perturbation group), the contribution, the mass, the relative
  error and the excess over the group's share, net of a floor where one is given; ``gene_where`` the same per gene
  group, with random gene sets of the same size as a reference;
- ``perturbation_group_test``: the excess of a group of perturbations against a null that permutes drug labels once
  across identities;
- ``four_parts``: the error split into the context-common part, the shared drug part, the identity x drug part and the
  within-identity residual (nested weighted projections).

References: ``floor_parts`` (the expected contribution of independent target noise per unit, "D1"), ``bag_gap`` (the
input bag's sampling against its pool, "T2"), ``oracle_context_mean``, the replicate-plate twins and the between-plate
discrepancy (``replicate_twins``, ``d2_parts``, ``d2_by_context``, "D2"), the plate test (``plate_test``, "D7") and the
matched comparison of annotated units (``t7_matched``, "T7"). The plate read-outs are for a screen that measures every
line on the same two replicate plates (Tahoe-100M's plates 6 and 14): ``plates`` names them, as the second field of the
context keys (``"<line>|<plate>"``).
"""
import numpy as np

from .comparison import ALPHA, SIGNFLIP_DRAWS, design_floor, sign_patterns
from .scoring import PARTS, identities, project, risk

__all__ = ["where", "gene_where", "group_excess", "perturbation_group_test", "four_parts", "floor_parts", "bag_gap",
           "oracle_context_mean", "matched_excess", "plate_contrast", "replicate_twins", "d2_by_context", "plate_test",
           "t7_matched", "d2_parts", "gene_floor", "NestedProjection"]


def where(parts, units, keys, scale, floor=None):
    """Per group, descriptively: contribution in % of `scale`, mass, relative error, excess T(A) - m_A T, and,
    given per-unit floor contributions, the group's floor and its excess net of the floor."""
    w = units['weight'] / units['weight'].sum()
    whole = {part: float(v.sum()) for part, v in parts.items()}
    rows = []
    for key in np.unique(keys):
        m = keys == key
        mass = float(w[m].sum())
        for part, v in parts.items():
            c = float(v[m].sum())
            row = {'key': str(key), 'part': part, 'contribution_pts': 100 * c / scale, 'mass': mass,
                   'relative_error': c / (mass * whole[part]) if whole[part] > 0 else float('nan'),
                   'excess_pts': 100 * (c - mass * whole[part]) / scale}
            if floor is not None:
                f, F = float(floor[part][m].sum()), float(floor[part].sum())
                row |= {'floor_pts': 100 * f / scale, 'excess_net_of_floor_pts': 100 * ((c - f) - mass * (whole[part] - F)) / scale}
            rows.append(row)
    return rows


def gene_where(vectors, groups, scale, floor=None, draws=1000, seed=20261019):
    """Per gene group and part, as where() names them: genes, mass |S|/G, contribution, relative error and raw excess;
    with per-gene floors, the group's floor and its excess net of it. Random gene sets of the same size give a
    descriptive reference for the excess read (net where a floor exists, raw otherwise), because genes are correlated."""
    n = len(groups)
    rng = np.random.default_rng(seed)
    rows = []
    for key in np.unique(groups):
        genes = np.flatnonzero(groups == key)
        share = len(genes) / n
        sets = np.stack([rng.choice(n, len(genes), replace=False) for _ in range(draws)])
        for part in PARTS:
            v = vectors[part]
            c, whole = float(v[genes].sum()), float(v.sum())
            row = {'key': str(key), 'part': part, 'genes': len(genes), 'mass': share, 'contribution_pts': 100 * c / scale,
                   'relative_error': c / (share * whole) if whole > 0 else float('nan'), 'excess_pts': 100 * (c - share * whole) / scale}
            if floor is not None:
                v = v - floor[part]
                row |= {'floor_pts': 100 * float(floor[part][genes].sum()) / scale}
            x = v[genes].sum() - share * v.sum()
            if floor is not None:
                row['excess_net_of_floor_pts'] = 100 * float(x) / scale
            ref = v[sets].sum(axis=1) - share * v.sum()
            rows.append(row | {'random_sets_basis': 'net of the floor' if floor is not None else 'raw',
                               'random_sets_at_or_above': float(np.mean(ref >= x))})
    return rows


def group_excess(parts, units, group_of_code, part='differential'):
    """Each perturbation group's excess X_A = T(A) - m_A T in one part, in units of the risk."""
    w = units['weight'] / units['weight'].sum()
    v = parts[part]
    labels = np.asarray([group_of_code[int(c)] for c in units['p_code']])
    return {str(g): float(v[labels == g].sum() - w[labels == g].sum() * v.sum()) for g in np.unique(labels)}


def perturbation_group_test(parts, units, group_of_code, draws=SIGNFLIP_DRAWS, seed=20261019, part='differential', chosen=None):
    """Family-wise p-value of each perturbation group's excess in one part. The null permutes drug labels once and
    applies the same permutation in every identity, keeping a drug's error dependence across identities; the
    statistic is the maximum excess over groups. With a group chosen beforehand, only that group's own excess is
    tested against its own null (not a family-wise p), with the resolution and the drug count."""
    codes = np.unique(units['p_code'])
    labels = np.asarray([group_of_code[c] for c in codes])
    w = units['weight'] / units['weight'].sum()
    v = parts[part]
    total = float(v.sum())
    index = np.searchsorted(codes, units['p_code'])
    per_code_v = np.bincount(index, weights=v, minlength=len(codes))
    per_code_m = np.bincount(index, weights=w, minlength=len(codes))
    names = np.unique(labels)

    def excess(lab):
        return np.asarray([per_code_v[lab == g].sum() - per_code_m[lab == g].sum() * total for g in names])

    observed = excess(labels)
    rng = np.random.default_rng(seed)
    if chosen is not None:   # choose then test: the chosen group's own excess against its own null
        j = list(names).index(chosen)
        null = np.asarray([excess(rng.permutation(labels))[j] for _ in range(draws)])
        return [{'group': str(chosen), 'part': part, 'excess': float(observed[j]), 'p': float((1 + np.sum(null >= observed[j])) / (draws + 1)),
                 'p_label': ("the chosen group's own excess against its drug-label permutation null (not family-wise); the group was "
                             "chosen for its calibration excess, so the p reads whether its drugs carry excess outer error, not "
                             "whether the label explains it (drugs hard on every line also qualify)"),
                 'draws': draws, 'resolution': 1 / (draws + 1), 'exchangeable_drugs': len(codes),
                 'group_drugs': int(np.sum(labels == chosen))}]
    null_max = np.asarray([excess(rng.permutation(labels)).max() for _ in range(draws)])
    return [{'group': str(g), 'part': part, 'excess': float(x), 'familywise_p': float((1 + np.sum(null_max >= x)) / (draws + 1))}
            for g, x in zip(names, observed)]


def _column_space(matrix):
    """The fitted-value basis, with NumPy's standard numerical-rank rule."""
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    return u[:, :rank]


class NestedProjection:
    """The four-part split of an error: context-common, shared drug, line x drug and within-line residual, as nested
    weighted projections (context within context + drug within context + line x drug) under one support's weights."""

    def __init__(self, contexts, codes, weights):
        self.contexts = np.asarray(contexts).astype(str)
        self.lines = np.asarray([c.split('|')[0] for c in self.contexts])
        self.weights = np.asarray(weights, dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.root_weight = np.sqrt(self.weights)[:, None]
        self.context_groups = [np.flatnonzero(self.contexts == c) for c in np.unique(self.contexts)]
        drugs, inverse = np.unique(codes, return_inverse=True)
        design = np.eye(len(drugs), dtype=np.float64)[inverse] * self.root_weight
        adjusted = design - self.context_project(design)
        self.shared_basis = _column_space(adjusted)
        self.line_bases = []
        for line in np.unique(self.lines):
            indices = np.flatnonzero(self.lines == line)
            self.line_bases.append((indices, _column_space(adjusted[indices])))

    def context_project(self, weighted_values):
        output = np.empty_like(weighted_values)
        for indices in self.context_groups:
            root = self.root_weight[indices]
            coefficient = (root.T @ weighted_values[indices]) / self.weights[indices].sum()
            output[indices] = root * coefficient
        return output

    def split(self, values):
        weighted = np.asarray(values, dtype=np.float64) * self.root_weight
        common = self.context_project(weighted)
        differential = weighted - common
        shared = self.shared_basis @ (self.shared_basis.T @ differential)
        per_line = np.empty_like(differential)
        for indices, basis in self.line_bases:
            per_line[indices] = basis @ (basis.T @ differential[indices])
        return {'context_common': common, 'shared_drug_adjusted': shared,
                'line_drug_associated': per_line - shared,
                'within_line_residual': differential - per_line}


def four_parts(pred, units, clip=True):
    """Energies of the four parts of the error: context-common, shared drug, identity x drug, within-identity
    residual, in units of the risk (normalized weights, gene mean); clipped at zero unless clip is False."""
    e = (np.maximum(pred, 0) if clip else pred) - units['truth']
    split = NestedProjection(units['context'], units['p_code'], units['weight']).split(e)
    n = e.shape[1]
    return {k: float(np.sum(v ** 2) / n) for k, v in split.items()}


def floor_parts(units, noise):
    """D1 per unit: each unit's expected noise contribution to each part of unit_parts, for independent zero-mean
    target noise of gene-averaged variance v. With q = w / m_context and S_c the context's sum of q^2 v: total w v,
    common w S_c, differential w ((1 - 2q) v + S_c). Over whole contexts they sum to total w v, common w q v and
    differential w (1 - q) v; for groups that cut contexts (perturbation groups) they are the expected contributions."""
    w = units['weight'] / units['weight'].sum()
    ctx = units['context']
    mass = {c: w[ctx == c].sum() for c in np.unique(ctx)}
    q = w / np.asarray([mass[c] for c in ctx])
    S = {c: float(np.sum(q[ctx == c] ** 2 * noise[ctx == c])) for c in mass}
    Sc = np.asarray([S[c] for c in ctx])
    return {'total': w * noise, 'common': w * Sc, 'differential': w * ((1 - 2 * q) * noise + Sc)}


def bag_gap(units, base, pool):
    """T2 per unit: its weighted share of |s_c - C_c|^2 / G (the input bag's mean against its pool's mean), common part
    only (base and pool are per-unit arrays carrying the context's means)."""
    w = units['weight'] / units['weight'].sum()
    gap = np.mean((base - pool) ** 2, axis=1)
    return {'total': w * gap, 'common': w * gap, 'differential': np.zeros_like(w)}


def oracle_context_mean(units):
    """The transductive context mean: each unit predicted by its context's weighted mean truth (an oracle reference)."""
    return project(units['truth'], units['context'], units['weight'])


def matched_excess(parts, units, annotated, part='differential'):
    """T7's matched comparison: the mean, over annotated units, of a unit's contribution minus the mean contribution of
    the same perturbation's units on the other identities (equal-weight mean), in the units of the risk."""
    ident = identities(units)
    v = parts[part]
    gaps = []
    for i in np.flatnonzero(annotated):
        others = (units['p_code'] == units['p_code'][i]) & (ident != ident[i]) & ~annotated
        if others.any():
            gaps.append(v[i] - v[others].mean())
    return float(np.mean(gaps)) if gaps else float('nan')


def plate_contrast(parts, units, part='common', *, plates):
    """D7's read-out: per line, the context-level contribution of the first plate minus the second in one part."""
    ident = identities(units)
    plate = np.asarray([c.split('|')[1] for c in units['context']])
    v = parts[part]
    return {l: float(v[(ident == l) & (plate == plates[0])].sum() - v[(ident == l) & (plate == plates[1])].sum())
            for l in np.unique(ident)}


def replicate_twins(units, *, plates):
    """Each unit's twin on the other replicate plate (the same line and drug), or -1 where it is missing."""
    line = identities(units)
    plate = np.asarray([c.split('|')[1] for c in units['context']])
    index = {(l, pl, p): i for i, (l, pl, p) in enumerate(zip(line, plate, units['p_code']))}
    other = {plates[0]: plates[1], plates[1]: plates[0]}
    return np.asarray([index.get((l, other.get(pl), p), -1) for l, pl, p in zip(line, plate, units['p_code'])])


def d2_by_context(units, twin, pool=None):
    """D2 per context: the mean-gene common energy of the twin deviations (y - y')/sqrt(2), treated, or response when
    per-unit pool means are given; None for a context whose units have no twin (undefined, never excluded)."""
    out = {}
    for c in np.unique(units['context']):
        m = np.flatnonzero(units['context'] == c)
        if np.any(twin[m] < 0):
            out[c] = None
            continue
        y, yt = units['truth'][m], units['truth'][twin[m]]
        if pool is not None:
            y, yt = y - pool[m], yt - pool[twin[m]]
        dev = (y - yt) / np.sqrt(2)
        w = units['weight'][m] / units['weight'][m].sum()
        out[c] = float(np.mean((w[:, None] * dev).sum(axis=0) ** 2))
    return out


def plate_test(parts, units, part='common', *, plates):
    """D7's read-out: each line's plate contrast (the first plate minus the second), with a two-sided sign flip over
    lines. Every line shares the one plate pair, so the p is conditional on that pair and assumes the lines' signs
    exchangeable within it; a shock shared by the pair's wells breaks that, so the reading is descriptive. Undefined
    when a line lacks a plate or fewer than two lines remain."""
    ident = identities(units)
    plate = np.asarray([c.split('|')[1] for c in units['context']])
    lines = np.unique(ident)
    if any(not set(plates) <= set(plate[ident == l]) for l in lines) or len(lines) < 2:
        return {'status': 'undefined'}
    contrast = plate_contrast(parts, units, part, plates=plates)
    values = np.asarray([contrast[l] for l in lines])
    floor = 2 * design_floor(len(values))
    p = float(np.mean(np.abs((sign_patterns(len(values)) * values).mean(axis=1)) >= abs(values.mean()) - 1e-18))
    status = 'descriptive by design (below the floor)' if floor > 2 * ALPHA else 'descriptive (one plate pair)'
    z = (values - values.mean()) / values.std(ddof=1) if len(values) > 2 else np.zeros_like(values)
    return {'status': status, 'p': p, 'p_label': 'conditional on the one plate pair; line signs assumed exchangeable',
            'floor': floor, f'lines_{plates[0]}_higher': int(np.sum(values > 0)), 'lines': len(values), 'per_line': contrast,
            'most_extreme': str(lines[int(np.argmax(np.abs(z)))]), 'most_extreme_z': float(np.max(np.abs(z)))}


def t7_matched(parts, units, annotated=None, part='differential'):
    """T7's matched comparison where a documented unequal treatment annotates units; "not tested" without one."""
    if annotated is None or not np.any(annotated):
        return {'status': 'not tested', 'evidence': 'no documented unequal treatment'}
    return {'status': 'computed', 'matched_excess': matched_excess(parts, units, annotated, part)}


def d2_parts(units, twin, pool=None):
    """D2's reference per part: the twin deviations (y - y')/sqrt(2), treated, or response when per-unit pool means
    are given, split by the risk on the twin-bearing units with their scoring weights. None when no unit has a
    twin."""
    keep = twin >= 0
    if not keep.any():
        return None
    y, yt = units['truth'][keep], units['truth'][twin[keep]]
    if pool is not None:
        y, yt = y - pool[keep], yt - pool[twin[keep]]
    return risk((y - yt) / np.sqrt(2), units['context'][keep], units['weight'][keep]) | {'units': int(keep.sum())}


def gene_floor(floor, share):
    """Per-gene floor vectors per part: each part's unit floor spread over genes by a per-gene variance share (an
    approximation, stated in the readings)."""
    return {part: float(v.sum()) * share for part, v in floor.items()}
