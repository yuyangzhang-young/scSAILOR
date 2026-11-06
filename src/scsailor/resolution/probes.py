"""Fixed-weight probes: what a trained model does with a control statistic, read from predictions under each condition.

A probe changes a trained model's control input and keeps its weights. Each condition replaces the statistic, ``true``
being the model's own input, and the anchor stays the context's own, so only the model's correction reads the replaced
input. These functions take the predictions on one side (the held-out or the training units) and read, in % of the
baseline's total MSE:

    L            the error under a condition, relative to the baseline
    dependence   the risk of the prediction under the own input minus under the replaced one
    H            benefit: L(true) - L(condition), negative when the own statistic lowers error
    E            added cost: the changed model's L minus the original model's, under one condition
    Delta_E      what the condition removes of the added cost, E(true) - E(condition)

The matched shrink scales the context-level correction, with weights fixed, to the size the line-agnostic control mean gives
it. Increment energy and best scale come from the moments of a change: the change's energy, its inner product with what
the model still needs, and that need's energy. The value of own controls is error under a line-agnostic input minus error
under the own input, across training length. A probe says how the model's output depends on its input, not what would
repair it.

A *side* extends a units dictionary (``scoring``) with ``anchor`` (per unit), ``baseline`` (the three parts of the
baseline's risk) and ``means`` (context -> the control mean).
"""
import numpy as np

from .card import alignment
from .scoring import PARTKEY, identities, project, risk, scores

__all__ = ["KEYS", "PARTS", "MOMENTS", "STATS", "l2_atoms", "condition_scores", "excess_columns", "matched_shrink", "line_risk", "score_model",
           "paired_tables", "line_draws", "ratio", "derived", "moments", "pair_geometry", "pair_contrast", "response_metrics", "own_value"]

KEYS = tuple(PARTKEY.values())                # the risk's three parts
PARTS = ('total', 'common', 'contrast')       # their names in a probe's tables
NAMES = dict(zip(KEYS, PARTS))
BOOTSTRAP_SEED = 2026102505
MOMENTS = ['norm2', 'dot', 'reference_norm2', 'norm2_pts', 'dot_pts', 'reference_norm2_pts']
STATS = MOMENTS + ['alpha_star', 'cosine', 'quadratic_cost_pts']


def l2_atoms(s, h, m, upper, rng):
    """A carrier of the second ladder level rebuilt from a given mean ``s`` and variance ``h`` (ddof 1), ``m`` rows, values
    in [0, upper] (``upper`` is ``scsailor.carrier.U``), by the carrier's construction: the amplitude ratio is capped at 1 where the variance exceeds
    what the mean allows, and each gene's rows are shuffled. Returns the atoms and the capped fraction of genes."""
    g = len(s)
    t = m * s
    full = np.floor(t / upper).astype(int)
    row = np.arange(m)[:, None]
    z = np.where(row < full[None, :], upper, 0.0)
    z += np.where(row == full[None, :], (t - full * upper)[None, :], 0.0)
    vmax = np.mean((z - s) ** 2, axis=0)
    v = (m - 1) * h / m
    amplitude = np.divide(v, vmax, out=np.zeros_like(v), where=vmax > 0)
    capped = float(np.mean(amplitude > 1 + 1e-8))
    c = s + np.sqrt(np.minimum(amplitude, 1.0)) * (z - s)
    return np.take_along_axis(c, np.argsort(rng.random((m, g)), axis=0), axis=0), capped


def condition_scores(preds, u, baseline):
    """condition -> L, dependence and H, in % of the baseline's total MSE. ``preds``: condition -> predictions, with
    'true' first; ``baseline``: the clipped baseline per unit."""
    contexts, w = u['context'], u['weight']
    base = risk(baseline - u['truth'], contexts, w)
    errors = {c: scores(p, u) for c, p in preds.items()}
    rows = {}
    for c in preds:
        row = {f'L_{n}': 100 * (errors[c][k] - base[k]) / base['total_mse'] for k, n in NAMES.items()}
        if c != 'true':
            d = risk(np.maximum(preds['true'], 0) - np.maximum(preds[c], 0), contexts, w)
            row.update({f'dependence_{n}': 100 * d[k] / base['total_mse'] for k, n in NAMES.items()})
            row.update({f'H_{n}': 100 * (errors['true'][k] - errors[c][k]) / base['total_mse'] for k, n in NAMES.items()})
        rows[c] = row
    return rows


def excess_columns(changed, original, changed_true, original_true, levels=False):
    """E, Delta_E and the original model's own change, per part, from four rows of L: the changed and the original model
    under the condition and under the own input. ``levels`` adds the two L themselves."""
    row = {}
    for part in PARTS:
        key = f'L_{part}'
        row[f'E_{part}'] = changed[key] - original[key]
        row[f'Delta_E_{part}'] = changed_true[key] - original_true[key] - row[f'E_{part}']
        row[f'parent_change_{part}'] = original_true[key] - original[key]
        if levels:
            row[f'repaired_{part}'], row[f'parent_{part}'] = changed[key], original[key]
    return row


def matched_shrink(true, swapped, anchor, contexts, weights):
    """The context-level correction scaled, per context and never up, to the size it has under ``swapped``; the
    perturbation-level correction stays. Returns the shrunk prediction and the scale per unit."""
    common = project(true - anchor, contexts, weights)
    target = project(swapped - anchor, contexts, weights)
    before = np.linalg.norm(common, axis=1)
    after = np.linalg.norm(target, axis=1)
    alpha = np.minimum(1, np.divide(after, before, out=np.ones_like(after), where=before > 0))
    return true + (alpha - 1)[:, None] * common, alpha


def line_risk(error, u, names):
    """Each line of ``names``: its additive share of the risk's three parts."""
    common = project(error, u['context'], u['weight'])
    values = np.stack([np.mean(a * a, axis=1) for a in (error, common, error - common)], axis=1)
    values *= (u['weight'] / u['weight'].sum())[:, None]
    lines = identities(u)
    return np.stack([values[lines == name].sum(0) for name in names])


def score_model(meta, state, grids, sides, diagnostics):
    """One model state under every condition, with the matched shrink added as the condition 'output_shrink'. ``grids``:
    side -> condition -> predictions; ``sides``: side -> its units and anchors; ``diagnostics``: (side, context, condition)
    -> the input's readings. Returns the rows of the score table, the per-context geometry and, per (side, condition), the
    lines' additive risk."""
    rows, geometries, line_values = [], [], {}
    for side, d in sides.items():
        contexts, w, anchor = d['context'], d['weight'], d['anchor']
        preds = grids[side]
        preds['output_shrink'], alpha = matched_shrink(preds['true'], preds['train_avg'], anchor, contexts, w)
        c_true = project(preds['true'] - anchor, contexts, w)
        shrunk_common = project(preds['output_shrink'] - anchor, contexts, w)
        residual_gap = float(np.max(np.abs((preds['output_shrink'] - anchor - shrunk_common) - (preds['true'] - anchor - c_true))))
        names = np.unique(identities(d))
        base, before = d['baseline'], line_risk(np.maximum(preds['true'], 0) - d['truth'], d, names)
        for condition, pred in preds.items():
            losses = line_risk(np.maximum(pred, 0) - d['truth'], d, names)
            dependence = line_risk(np.maximum(preds['true'], 0) - np.maximum(pred, 0), d, names)
            line_values[(side, condition)] = (names, losses)
            common = project(pred - anchor, contexts, w)
            geometry = []
            for context in np.unique(contexts):
                mask = contexts == context
                index = np.flatnonzero(mask)[0]
                a, b = c_true[index], common[index]
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                cosine = float(np.dot(a, b) / denominator) if denominator > 0 else float('nan')
                diagnostic = diagnostics[(side, context, 'true')] if condition == 'output_shrink' else diagnostics.get((side, context, condition), {})
                geom = dict(meta, state=state, side=side, condition=condition, context=context, support_weight=float(w[mask].sum() / w.sum()),
                            common_energy=float(np.mean(b * b)), cosine=cosine, **diagnostic)
                if condition == 'output_shrink':
                    geom['alpha'] = float(alpha[index])
                geometries.append(geom)
                geometry.append(geom)
            valid = [g for g in geometry if np.isfinite(g['cosine'])]
            row = dict(meta, state=state, side=side, condition=condition, baseline_total=base[0],
                       common_correction_energy=sum(g['support_weight'] * g['common_energy'] for g in geometry),
                       cosine_weighted=sum(g['support_weight'] * g['cosine'] for g in valid) / sum(g['support_weight'] for g in valid) if valid else float('nan'),
                       cosine_undefined_weight=sum(g['support_weight'] for g in geometry if not np.isfinite(g['cosine'])),
                       shrink_contrast_max_abs=residual_gap if condition == 'output_shrink' else 0.)
            for j, part in enumerate(PARTS):
                row.update({f'absolute_L_{part}': losses[:, j].sum(), f'L_{part}': 100 * (losses[:, j].sum() - base[j]) / base[0],
                            f'H_{part}': 100 * (before[:, j] - losses[:, j]).sum() / base[0],
                            f'dependence_{part}': 100 * dependence[:, j].sum() / base[0]})
            for key in ('distance_to_train', 'distance_to_train_ratio', 'mean_change_norm', 'mean_change_ratio', 'mean_clipped_fraction',
                        'variance_capped_fraction', 'carrier_change_rms'):
                available = [g for g in geometry if key in g]
                if available:
                    row[key] = sum(g['support_weight'] * g[key] for g in available) / sum(g['support_weight'] for g in available)
            rows.append(row)
    return rows, geometries, line_values


def paired_tables(pair_records):
    """The added cost per fold and seed with intervals over lines, and each line's contribution. Each record holds
    (meta, state -> ``score_model``'s line risks, sides) for one change, fold and seed. The mean over the random mean
    permutations (conditions 'S1_<seed>') is 'S1_mean'."""
    rows, line_rows, groups = [], [], {}
    for meta, values, sides in pair_records:
        for side, d in sides.items():
            names, rt = values['repaired'][(side, 'true')]
            pt, ra, pa = values['parent'][(side, 'true')][1], values['repaired'][(side, 'train_avg')][1], values['parent'][(side, 'train_avg')][1]
            for c in sorted(c for s, c in values['repaired'] if s == side):
                rs, ps = values['repaired'][(side, c)][1], values['parent'][(side, c)][1]
                vectors = {'E': rs - ps, 'Delta_E': rt - pt - rs + ps, 'parent_change': pt - ps, 'E_minus_train_avg': rs - ps - ra + pa}
                for metric, vec in vectors.items():
                    for j, part in enumerate(PARTS):
                        scaled = 100 * vec[:, j] / d['baseline'][0]
                        key = (meta['model'], meta['fold'], meta['cell'], side, c, metric, part)
                        groups.setdefault(key, []).append((meta['seed'], names, scaled))
                        for n, v in zip(names, scaled):
                            line_rows.append(dict(meta, side=side, condition=c, metric=metric, part=part, line=n, contribution=float(v)))
    # S1 mean is a mean of metrics for the three fixed interventions, within seed and line.
    directions = {}
    for k, entries in groups.items():
        if k[4].startswith('S1_'):
            directions.setdefault(k[:4] + k[5:], []).append(entries)
    for key, by_direction in directions.items():
        if len(by_direction) != 3:
            raise ValueError(f'{key}: S1_mean needs all three random directions')
        seeds = sorted({s for entries in by_direction for s, _, _ in entries})
        means = []
        for seed in seeds:
            found = [next(v for v in entries if v[0] == seed) for entries in by_direction]
            means.append((seed, found[0][1], np.mean([e[2] for e in found], axis=0)))
        groups[key[:4] + ('S1_mean',) + key[4:]] = means
    for key, entries in sorted(groups.items()):
        model, fold, cell, side, c, metric, part = key
        n = len(entries[0][1])
        draws = line_draws(n)
        entries = entries + [('mean', entries[0][1], np.mean([x[2] for x in entries], axis=0))]
        for seed, names, vector in entries:
            row = dict(model=model, fold=fold, cell=cell, side=side, condition=c, metric=metric, part=part, seed=seed,
                       value=float(vector.sum()), n_lines=n)
            if side == 'heldout':
                samples = vector[draws].sum(1)
                row.update(ci_low=float(np.percentile(samples, 2.5)), ci_high=float(np.percentile(samples, 97.5)))
            rows.append(row)
    return rows, line_rows


def line_draws(n):
    """The fixed resamples of ``n`` lines behind every interval over lines."""
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(2000, n))


def ratio(x, y):
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.divide(x, y, out=np.full(x.shape, np.nan), where=y != 0)


def derived(v):
    """Ratios of aggregated moments, not an average prediction or average angle. ``v``: the six ``MOMENTS`` on the last
    axis. ``alpha_star`` is the best scale, ``quadratic_cost_pts`` what the object costs at full strength."""
    a, b, c, ap, bp, cp = np.moveaxis(np.asarray(v), -1, 0)
    return dict(zip(MOMENTS, (a, b, c, ap, bp, cp))) | dict(
        alpha_star=ratio(b, a), cosine=ratio(b, np.sqrt(a * c)), quadratic_cost_pts=ap - 2 * bp)


def moments(v, ref, w):
    """Per part, the energy of v, its inner product with ref and the energy of ref, in one context."""
    av, ar = np.average(v, axis=0, weights=w), np.average(ref, axis=0, weights=w)
    return {'total': np.array([np.average(np.mean(v * v, axis=1), weights=w),
                               np.average(np.mean(v * ref, axis=1), weights=w),
                               np.average(np.mean(ref * ref, axis=1), weights=w)]),
            'common': np.array([np.mean(av * av), np.mean(av * ar), np.mean(ar * ar)])}


def pair_geometry(meta, grids, anchors, sides, baseline, mean_conditions):
    """Per context, condition, part and object, the moments and readings of a change on its original model (clipped
    predictions): 'repair_increment' the change, 'whole_parent' and 'whole_repaired' each model's correction and, under a
    condition of ``mean_conditions`` (those that change the control mean), the 'response_*' to that change. ``grids``:
    'parent' and 'repaired' -> side -> condition -> predictions; ``baseline``: side -> the baseline's total. A whole
    correction's common part keeps ``card.alignment``'s reading."""
    rows = []
    for side, u in sides.items():
        scale = 100 / baseline[side]
        contexts, w, y = u['context'], u['weight'], u['truth']
        for context in np.unique(contexts):
            ix = np.flatnonzero(contexts == context)
            weight = w[ix]
            mass = float(weight.sum() / w.sum())
            truth, anchor = y[ix], anchors[side][ix]
            p0, q0 = grids['parent'][side]['true'][ix], grids['repaired'][side]['true'][ix]
            for condition in grids['parent'][side]:
                p, q = grids['parent'][side][condition][ix], grids['repaired'][side][condition][ix]
                objects = {'repair_increment': (q - p, truth - p), 'whole_parent': (p - anchor, truth - anchor),
                           'whole_repaired': (q - anchor, truth - anchor)}
                if condition in mean_conditions:
                    dp, dq = p - p0, q - q0
                    objects.update(response_parent=(dp, truth - p0), response_repaired=(dq, truth - q0), response_excess=(dq - dp, truth - q0))
                for name, (v, ref) in objects.items():
                    values = moments(v, ref, weight)
                    existing = None
                    if name.startswith('whole_'):
                        existing = alignment(p if name == 'whole_parent' else q, anchor,
                                             dict(truth=truth, context=contexts[ix], weight=weight))['common']
                    for part, m in values.items():
                        stats = {k: float(x) for k, x in derived(np.r_[m, m * scale]).items()}
                        if existing is not None and part == 'common' and m[0] > 0 and m[2] > 0:
                            stats.update(cosine=existing['cosine'], alpha_star=existing['best_scale'])
                        rows.append(meta | dict(side=side, condition=condition, part=part, object=name, context=context, line=context.split('|')[0],
                                                support_weight=mass, **stats))
    return rows


def pair_contrast(x, y):
    """Two conditions' ``derived`` readings of a change: their differences and the terms of the cost's difference."""
    aa, ab = x['alpha_star'], y['alpha_star']
    ea, eb = x['norm2_pts'], y['norm2_pts']
    return {f'difference_{k}': x[k] - y[k] for k in STATS} | dict(energy_term=ea - eb, alignment_term=-2 * (x['dot_pts'] - y['dot_pts']),
                                                                  symmetric_energy_term=(ea - eb) * (1 - aa - ab),
                                                            symmetric_alpha_term=-(aa - ab) * (ea + eb))


def response_metrics(p, r, e):
    """How the original model, the changed one and their excess respond to a changed mean, from ``derived`` readings. A
    'changed' residual is the one after the input change, r_after = r_true - response, so it needs no grid."""
    def changed_cosine(v):
        after_norm = v['reference_norm2'] + v['norm2'] - 2 * v['dot']
        return ratio(v['dot'] - v['norm2'], np.sqrt(v['norm2'] * after_norm))
    extra_after_dot = e['dot'] - (r['norm2'] - p['norm2'] + e['norm2']) / 2
    extra_after_dot_pts = e['dot_pts'] - (r['norm2_pts'] - p['norm2_pts'] + e['norm2_pts']) / 2
    repaired_after_norm = r['reference_norm2'] + r['norm2'] - 2 * r['dot']
    return dict(repaired_to_parent_energy_ratio=ratio(r['norm2'], p['norm2']), repaired_minus_parent_energy_pts=r['norm2_pts'] - p['norm2_pts'],
                parent_residual_cosine=p['cosine'], repaired_residual_cosine=r['cosine'], parent_changed_residual_cosine=changed_cosine(p),
                repaired_changed_residual_cosine=changed_cosine(r),
                excess_response_changed_residual_cosine=ratio(extra_after_dot, np.sqrt(e['norm2'] * repaired_after_norm)),
                excess_response_changed_residual_dot_pts=extra_after_dot_pts, excess_response_residual_cosine=e['cosine'],
                excess_response_residual_dot_pts=e['dot_pts'])


def line_stats(error, u):
    """Per cell line: mass and the risk's three parts (mass-weighted line averages add up to the overall risk)."""
    lines, contexts, weights = identities(u), u['context'], u['weight']
    names = np.unique(lines)
    mass = np.array([weights[lines == l].sum() for l in names])
    stats = np.array([[risk(error[lines == l], contexts[lines == l], weights[lines == l])[k] for k in KEYS] for l in names])
    return mass, stats


def own_value(preds, u, baseline, draws, agnostic):
    """condition -> error relative to the baseline, per part, and for each line-agnostic condition the value of the own
    controls with its 95% interval over paired resamples of lines. ``preds``: 'own' and the ``agnostic`` conditions;
    ``draws``: resampled line indices."""
    truth = u['truth']
    b = risk(baseline - truth, u['context'], u['weight'])
    mass, b_lines = line_stats(baseline - truth, u)
    rows, lines = {}, {}
    for condition, pred in preds.items():
        q = scores(pred, u)
        rows[condition] = {NAMES[k]: 100 * (q[k] - b[k]) / b['total_mse'] for k in KEYS}
        lines[condition] = line_stats(np.maximum(pred, 0) - truth, u)[1]
    weights = np.stack([np.bincount(d, minlength=len(mass)) for d in draws]) * mass[None, :]
    denominator = weights @ b_lines[:, 0]
    for condition in agnostic:
        diff = lines[condition] - lines['own']
        for i, k in enumerate(KEYS):
            name = NAMES[k]
            samples = 100 * (weights @ diff[:, i]) / denominator
            rows[condition].update({f'value_{name}': rows[condition][name] - rows['own'][name],
                                    f'value_{name}_low': float(np.percentile(samples, 2.5)),
                                    f'value_{name}_high': float(np.percentile(samples, 97.5))})
    return rows
