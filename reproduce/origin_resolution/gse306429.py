"""GSE306429's binding, fold 1: the case from the nested parents, the dataset's documented
origin rows and its evidence for the computed ones.

A context is one cell identity|plate|library|replicate, so identities take the place of cell lines; four identities
are fitted (the first four of the fixed order), two calibrate (CD34+, HEK293T) and four are held out (A549, CD8+, PAD,
PC3). Each identity comes with its own plates, libraries, medium and campaign mix, so these coincide with identity by
design. The conditions without a fit-side training target (``UNSUPPORTED.json`` of the nested runs) leave the
calibration units and the nested parent's held-out units, with weights recomputed by the datasets' rule.

Files: the nested parents' data (``nested-parents-gse306429-replogle-nadig``: ``data/gse306429/``,
``scoring/gse306429_HELDOUT_TRUTH.npz``); the nested runs' grids and ``UNSUPPORTED.json`` under ``RESULTS``
(``<dataset>/<model>_<seed>/``); the saved selections in ``SELECTIONS``; the dataset references the stages read in
``REFERENCES``. There is no plate-aware target, no restoring change and no published model here, so the menu adds only
the two-amount shrink toward the anchor.
"""
import csv
import gzip

import numpy as np

from scsailor.resolution.scoring import PARTKEY, PARTS, identities, project, scores
from . import (ROOT, REFERENCES, SEEDS, listed_unsupported, nested_calibration_lines, nested_case, noise_table,
          read_csv)

__all__ = ["SEEDS", "FOLDS", "PACKAGE", "RESULTS", "SELECTIONS", "REFERENCES", "files", "line_of", "stratum_of", "calibration_lines",
           "pool_cells", "design_shift", "unsupported", "records", "grid_paths", "case", "menu_inputs", "selection_path", "twins",
           "side_contrasts", "evidence", "STATIC"]

FOLDS = ('F1',)   # the folds with nested parents; the fold arguments below are always fold 1
PACKAGE = ROOT / 'nested-parents-gse306429-replogle-nadig'
RESULTS = ROOT / 'nested-parents-gse306429-replogle-nadig'   # the nested runs: <dataset>/<model>_<seed>/
SELECTIONS = ROOT / 'nested-parents-gse306429-replogle-nadig/selections'   # SELECTIONS_<dataset>_<model>.json (``selections``, nested)
MEMBERSHIP = ROOT / 'gse306429-control-membership/control_membership.csv.gz'   # every control cell's side, context and role
N_FIT = 4
GROUPS = None   # no perturbation grouping is declared


def files(fold='F1'):
    return {'train': PACKAGE / 'data/gse306429/train/train_targets.npz',
            'constants': PACKAGE / 'data/gse306429/training_constants.npz',
            'outer': PACKAGE / 'scoring/gse306429_HELDOUT_TRUTH.npz',
            'order': PACKAGE / 'data/gse306429/LINE_ORDER.json',
            'inputs': PACKAGE / 'data/gse306429/evaluation_inputs.h5', 'n_fit': N_FIT}


def line_of(context):
    return context.split('|')[0]


def stratum_of(context):
    """No design stratum: partners may come from any context of another line."""
    return ''


def calibration_lines(fold='F1'):
    return nested_calibration_lines(files(fold), line_of)


def pool_cells(fold='F1'):
    """'<side>/<context>' -> the cells of the context's control input pool, counted from the membership table."""
    out = {}
    with gzip.open(MEMBERSHIP, 'rt') as handle:
        for r in csv.DictReader(handle):
            if r['role'] == 'input':
                key = f"{r['side']}/{r['q_key']}"
                out[key] = out.get(key, 0) + 1
    return out


def design_shift(case, u, nested, bank):
    """No design stratum: the design-aware target is the anchor."""
    return 0.


def unsupported(dataset, model, seeds):
    """The conditions without a fit-side training target, as every seed's nested run lists them."""
    return listed_unsupported(RESULTS / dataset / f'{model}_{s}' for s in seeds)


def records(model, seed, fold='F1'):
    """A nested run's record folder (``COMPLETE.json``, ``UNSUPPORTED.json``, ``nested-parent/RUN_IDENTITY.json``)."""
    return RESULTS / 'gse306429' / f'{model}_{seed}'


def grid_paths(model, seeds, fold='F1'):
    return {s: {'nested': RESULTS / 'gse306429' / f'{model}_{s}' / 'predictions-nested.npz',
                'full': RESULTS / 'gse306429' / f'{model}_{s}' / 'predictions-full.npz'} for s in seeds}


def case(model, fold='F1', outer=True, seeds=SEEDS):
    if fold != 'F1':
        raise ValueError('gse306429 has nested parents for fold 1 only')
    out = nested_case('gse306429', model, files(), grid_paths(model, seeds), unsupported('gse306429', model, seeds), outer)
    ref = REFERENCES / 'gse306429'
    out['noise'] = {'fit': noise_table(ref / 'UNIT_NOISE_TRAIN.csv', out['units']['fit']),
                    'calibration': noise_table(ref / 'UNIT_NOISE_TRAIN.csv', out['units']['calibration']),
                    'outer': noise_table(ref / 'UNIT_NOISE_OUTER.csv', out['units']['outer']) if outer else None,
                    'outer_nested': noise_table(ref / 'UNIT_NOISE_OUTER.csv', out['units']['outer_nested']) if outer else None}
    out['meta'] = {'coincident': 'identity = its plates, libraries, medium and campaign mix (GSE306429 design)',
                   'type_variable': {'CD8+': 'T cell', 'PAD': 'adipocyte progenitor', 'PC3': 'prostate cancer line',
                                     'A549': 'lung cancer line'}}
    out['static_rows'] = {'static': STATIC, 'fold1_only': (), 'seed_permutation': 'not tested (precondition unavailable)'}
    return out


def menu_inputs(case, which, seeds, a):
    """No plate-aware target, restoring change or published model on GSE306429: the menu's side needs nothing more."""
    return {}


def selection_path(model, fold, cell, out):
    return SELECTIONS / f'SELECTIONS_gse306429_{model}.json'


def twins(units):
    """GSE306429's within-campaign twins as the references form them: every replicate-1 unit with every replicate-2
    unit of the same identity, condition and campaign. Returns the pairs and each unit's campaign."""
    meta = {(r['context'], int(r['p_code'])): r for r in read_csv(REFERENCES / 'gse306429/UNITS_META.csv') if r['side'] == 'outer'}
    m = [meta[(c, int(q))] for c, q in zip(units['context'], units['p_code'])]
    sides = {}
    for i, r in enumerate(m):
        sides.setdefault((r['identity'], r['p_code'], r['campaign']), {}).setdefault(r['replicate'], []).append(i)
    pairs = np.asarray([(i, j) for s in sides.values() for i in s.get('1', ()) for j in s.get('2', ())], dtype=np.int64)
    return pairs, np.asarray([r['campaign'] for r in m])


def side_contrasts(pred, units, ident, campaign, pairs, omega):
    """D7 (a) for one prediction, as the references compute it for the anchor and no change: per identity and
    campaign, the sum over matched twin pairs of omega (E_rep1 - E_rep2), with each unit's unweighted energy per part,
    in points of the prediction's own outer total (descriptive)."""
    e = np.maximum(pred, 0) - units['truth']
    pe = project(e, units['context'], units['weight'])
    energy = {'total': np.mean(e ** 2, axis=1), 'common': np.mean(pe ** 2, axis=1), 'differential': np.mean((e - pe) ** 2, axis=1)}
    T = scores(pred, units)['total_mse']
    i_, j_ = pairs[:, 0], pairs[:, 1]
    out = {}
    for key in sorted({(ident[x], campaign[x]) for x in i_}):
        g = np.flatnonzero((ident[i_] == key[0]) & (campaign[i_] == key[1]))
        out[key] = {part: 100 * float(np.sum(omega[g] * (energy[part][i_[g]] - energy[part][j_[g]]))) / T for part in PARTS} | {'pairs': len(g)}
    return out


def evidence(case, held, mean_parts, pool, parent_part, preds):
    """D2 on the twin-bearing units, D7 by replicate side and campaign, T1 and T7 for attribution's rows, with their
    reference and "where" rows."""
    ev, refs, where_rows = {}, [], []
    REF = REFERENCES
    ident = identities(held)
    w = held['weight'] / held['weight'].sum()
    pairs, campaign = twins(held)
    bearing = np.zeros(len(ident), bool)
    bearing[pairs.ravel()] = True
    d2 = {(r['identity'], r['version']): r for r in read_csv(REF / 'gse306429/D2_GSE306429.csv') if r['level'] == 'identity'}
    own = lambda s: {part: float(np.mean([scores(x, held, s)[key] for x in preds['parent']])) for part, key in PARTKEY.items()}
    text = []
    for name, members in (('PAD', ('PAD',)), ('A549', ('A549',)), ('PC3', ('PC3',)), ('A549, PAD, PC3', ('A549', 'PAD', 'PC3'))):
        inside = np.isin(ident, members)
        s = bearing & inside
        if int(d2[(name, 'response')]['twin_units']) != int(s.sum()):   # the matched support must be the reference's
            raise ValueError(f"GSE306429 {name}: {int(s.sum())} twin-bearing units against the reference's {d2[(name, 'response')]['twin_units']}")
        matched, whole = own(s), own(inside)
        support = f"{int(s.sum())} of {int(inside.sum())} units, {float(d2[(name, 'response')]['twin_mass_share']):.0%} of their mass"
        refs += [{'reference': f'parent on the twin-bearing units, {name}', 'support': support, **matched, 'source': 'card'},
                 {'reference': f'parent on all units, {name}', 'support': f'{int(inside.sum())} units', **whole, 'source': 'card'}]
        for version in ('response', 'treated'):
            r = d2[(name, version)]
            for label, prefix in (('twin deviations', ''), ('cell sampling', 'cells_'), ('control-pool sampling', 'pools_'), ('excess over sampling', 'excess_')):
                if prefix == 'pools_' and version == 'treated':
                    continue
                vals = {part: float(r[prefix + part]) for part in PARTS}
                refs.append({'reference': f'D2 {version}, {name}: {label}', 'support': support, **vals,
                             **{f'{part}_over_parent_same_units': vals[part] / matched[part] for part in PARTS},
                             'source': 'references/gse306429/D2_GSE306429.csv'})
        r = d2[(name, 'response')]
        text.append(f"{name} ({support}): " + ', '.join(f"{float(r[part]) / matched[part]:.2f}" for part in PARTS)
                    + '; excess ' + ', '.join(f"{float(r['excess_' + part]) / matched[part]:.2f}" for part in PARTS))
    ev['d2'] = {'status': 'reference',
                'evidence': "response version on the twin-bearing units, as shares of the parent's error on the same units (total, common, "
                            "drug difference; then the excess over cell and control-pool sampling): " + '; '.join(text)
                            + "; CD8+ missing (3 twin pairs); the parent on all units and the treated version in REFERENCES "
                              "(references/gse306429/D2_GSE306429.csv)"}
    k = np.bincount(pairs.ravel(), minlength=len(ident))
    omega = (w[pairs[:, 0]] / k[pairs[:, 0]] + w[pairs[:, 1]] / k[pairs[:, 1]]) / 2   # one matched weight per pair
    side = {}
    for name in ('parent', 'anchor', 'nochange'):
        per_seed = [side_contrasts(x, held, ident, campaign, pairs, omega) for x in preds[name]]
        side[name] = {key: {part: float(np.mean([s_[key][part] for s_ in per_seed])) for part in PARTS} | {'pairs': per_seed[0][key]['pairs']}
                      for key in per_seed[0]}
    for key, v in side['parent'].items():
        where_rows += [{'grouping': 'D7 replicate side: replicate-1 minus replicate-2 plates over matched twin pairs (descriptive)',
                        'key': f'{key[0]} {key[1]}', 'part': part, 'twin_pairs': v['pairs'],
                        **{f'{name}_contrast_pts': side[name][key][part] for name in side}} for part in PARTS]
    meta = {r['context']: r['campaign'] for r in read_csv(REF / 'gse306429/UNITS_META.csv')}
    ratios = {}
    for l in np.unique(ident):
        per_ctx = {}
        for cx in np.unique(held['context'][ident == l]):
            m = held['context'] == cx
            per_ctx.setdefault(meta.get(cx), []).append(float(mean_parts['common'][m].sum() / w[m].sum()))
        if 'ILD1' in per_ctx and 'ILD2' in per_ctx:
            ratios[l] = np.mean(per_ctx['ILD1']) / np.mean(per_ctx['ILD2'])
    dmso = {r['identity']: r for r in read_csv(REF / 'gse306429/DMSO_DECOMPOSITION.csv') if r['level'] == 'identity'}
    shift = {l: float(dmso[l]['campaign_shift']) if dmso.get(l, {}).get('campaign_shift') else None for l in ('PAD', 'A549', 'PC3', 'CD8+')}
    refs += [{'reference': f'DMSO campaign shift (controls), {l}', 'value': v if v is not None else 'one campaign',
              'source': 'references/gse306429/DMSO_DECOMPOSITION.csv'} for l, v in shift.items()]
    ev['d7'] = {'status': 'descriptive by design (2 campaign units per identity, 1 for CD8+); identity level confounded by design',
                'evidence': "replicate-1 minus replicate-2 plates over matched twin pairs, total part, points of each predictor's own total "
                            "(parent; anchor; no change): "
                            + ', '.join(f"{l} {c} {v['total']:+.2f}; {side['anchor'][(l, c)]['total']:+.2f}; {side['nochange'][(l, c)]['total']:+.2f} ({v['pairs']} pairs)"
                                        for (l, c), v in side['parent'].items())
                            + "; the parent's per-context common error, campaign ILD1 over ILD2: " + ', '.join(f'{l} {v:.2f}' for l, v in ratios.items())
                            + '; DMSO campaign shift (controls): ' + ', '.join(f'{l} {1e3 * v:+.1f}e-3' if v is not None else f'{l} one campaign' for l, v in shift.items())
                            + '; every part in WHERE; model-free contrasts in references/gse306429/D7_GSE306429_MODELFREE.csv'}
    rel = {l: float(mean_parts['total'][ident == l].sum() / (w[ident == l].sum() * mean_parts['total'].sum())) for l in np.unique(ident)}
    ev['t1'] = {'status': 'descriptive (development: the lineage variable was declared after per-identity errors were read)',
                'evidence': 'relative error by identity: ' + ', '.join(f'{l} ({case["meta"]["type_variable"].get(l, "")}) {v:.2f}' for l, v in rel.items())}
    ev['t7'] = {'status': 'not testable as relative dose',
                'evidence': 'nominal doses equal for every outer condition (units.csv); relative dose needs the inaccessible supplement'}
    ev['reference_rows'], ev['where_rows'] = refs, where_rows
    return ev


# The rows attribution does not compute.
STATIC = {
    'M1': ('needs information localization', 'information localization cannot read GSE306429 yet', "information localization's light version on GSE306429"),
    'M2': ('needs information localization', "as M1", "information localization's light version on GSE306429"),
    'M3': ('not applicable', "squared-error training of the scored targets", 'none'),
    'M4': ('not tested', "needs crossed tests", 'fits'),
    'M6': ('not tested', 'checkpoints on the cluster; no symptom scan', 'fits'),
    'D3': ('not tested', "needs noise refits", 'fits'),
    'D4': ('not tested', 'E-test not run on outer units', 'local E-test, about 20 minutes'),
    'D5': ('not applicable', "designed biological states (CD8+ activated, CD34+ in TPO medium, PAD differentiating) are read under D6 and T1", 'none'),
    'D6': ('reference', "pseudobulk distances; the coverage contrast nested against full parent is in the three readings", 'none (split undefined: two calibration identities)'),
    'D8': ('see counts', "eligible outer units: A549 160, CD8+ 20, PAD 152, PC3 163; CD8+ carries 7.9% of the weight; one background without treated units", 'none'),
    'D9': ('not tested', "no reader fit on GSE306429", 'CPU reader fits'),
    'T4': ('holds', "selections written before outer truth (SELECTIONS_*.json; CALIBRATION_*.json)", 'none'),
    'T5': ('not used', "no synthetic reading enters this card", 'none'),
    'T6': ('not tested', "only 40k is saved", 'fits'),
    'T8': ('not applicable', "each identity in its own wells", 'none'),
    'O2': ('holds', "anchors within 3.9e-5 of the saved baseline; per-identity errors equal GSE306429's own scores", 'none'),
}
