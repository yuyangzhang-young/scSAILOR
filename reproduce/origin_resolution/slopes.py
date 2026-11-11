"""The layers' slopes on every screen, model and fold, from saved grids.

Three stages, run in this order, writing into ``--out``:

- ``calibration``: from calibration lines alone: ``CALIBRATION_READINGS.csv`` (one row per model, fold and layer; the
  transferred response once per fold), ``RELIABILITY.csv`` (the leave-out reading on calibration lines) and ``INERT_LAYERS.csv``
  (seed-difference layers). Builds no held-out unit;
- ``heldout``: the same readings on held-out lines, beside the calibrated ones: ``READINGS.csv``;
- ``tables``: every table from ``READINGS.csv`` alone, so another cut needs no rerun: ``PRIMARY.csv``,
  ``TOLERANCE_CURVE.csv``, ``RANK_CORRELATION.csv``, ``USE.csv``, ``SPECIFICITY.csv`` and ``SENSITIVITY.csv``.

    python -m reproduce.origin_resolution.slopes --stage calibration --out DIR
"""
import argparse
from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from scsailor.resolution import slopes
from scsailor.resolution.stages import bag_of
from . import SEEDS, gse306429, jiang, parse, read_csv, replogle, tahoe, write_csv

__all__ = ["BUILT", "CASES", "BINDINGS", "BAG", "load_case", "read_side", "calibration_stage", "heldout_stage", "tables_stage", "main"]

BUILT = ('summary', 'deepsets')
CASES = ([('tahoe', f, m) for f in ('F1', 'F2', 'F3') for m in ('summary', 'deepsets', 'attention', 'official_state')]
         + [('gse306429', 'F1', m) for m in BUILT] + [('jiang', f, m) for f in ('F4', 'F5') for m in BUILT]
         + [('replogle', f, m) for f in ('F1', 'F2') for m in BUILT] + [('parse', g, m) for g in ('G1', 'G2', 'G3') for m in BUILT])
BINDINGS = {'tahoe': tahoe, 'gse306429': gse306429, 'replogle': replogle, 'jiang': jiang, 'parse': parse}
BAG = 256   # cells of every input set
OUTSIDE_PARTNER = ('replogle', 'jiang')   # one held-out line: partners are the calibration contexts, on the nested chain
FLOAT = {'slope', 'slope_bag', 'fraction', 'assigned', 'ordinary', 'fixed', 'slope_matched', 'm', 'S', 'S_matched'}
TEXT = ('screen', 'fold', 'model', 'layer', 'judgment', 'context_class')
CHAINS = ('nested', 'full')
METHODS = (('this rule', 'fraction'), ('ordinary calibration', 'ordinary'), ('fixed per layer', 'fixed'), ('fixed 1/2', .5),
           ('fixed 1/8', .125), ('as it stands', 1.))
TOLERANCES = np.round(np.arange(0, .5001, .01), 2)
KEEP_CUTS = (.40, .45, .50, .55, .60)
EMPTY_CUTS = (1 / 32, 3 / 64, 1 / 16, 3 / 32, 1 / 8)
S_CUTS = (.06, .08, .10, .125, .15)


def load_case(screen, fold, model, outer, cell=None):
    return BINDINGS[screen].case(model, fold, outer=outer, seeds=SEEDS, **({'cell': cell} if cell else {}))


@cache
def pool_cells(screen, fold):
    return BINDINGS[screen].pool_cells(fold)


def read_side(screen, case, which):
    """One side (``slopes``): 'calibration' (nested chain, calibration units), 'nested' (nested chain, held-out units the
    nested model can be scored on) or 'full' (full chain, held-out units). Builds held-out units only for the last two,
    which only the held-out stage asks for."""
    binding = BINDINGS[screen]
    key, chain, grid_side, bank = {'calibration': ('calibration', 'nested', 'train', 'train'),
                                   'nested': ('outer_nested', 'nested', 'outer', 'heldout'),
                                   'full': ('outer', 'full', 'outer', 'heldout')}[which]
    u = case['units'][key]
    seeds = {s: case['load'](s, chain, grid_side, u) for s in SEEDS}
    T = np.mean([case['load'](s, chain, grid_side, u, 'anchor_raw') for s in SEEDS], axis=0)
    if screen == 'tahoe':   # the plate-aware baseline; the perturbation mean elsewhere
        T = T + tahoe.design_shift(case, u, which != 'full', bank)
    b, C = bag_of(case['bags'][bank], u)
    pools = pool_cells(screen, case['fold'])
    n = np.asarray([pools[f'{bank}/{c}'] for c in u['context']], dtype=np.float64)[:, None]
    return {'u': u, 'seeds': seeds, 'T': T, 'b': b, 'outside': slopes.pool_without_set(C, b, n, BAG),
            'lines': np.asarray([binding.line_of(c) for c in u['context']]), 'strata': np.asarray([binding.stratum_of(c) for c in u['context']])}


def calibration_stage(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    readings, inert, mock, seen = [], [], {}, set()
    for screen, fold, model in CASES:
        key = (screen, fold)
        sd = read_side(screen, load_case(screen, fold, model, False), 'calibration')
        n_lines = len(BINDINGS[screen].calibration_lines(fold))
        outside, bag = slopes.seed_slopes(sd, 'outside'), slopes.seed_slopes(sd, 'b')
        amounts = dict(zip(slopes.LAYERS, slopes.ordinary(sd)))
        terms = slopes.line_terms(sd, 'outside')
        specific = slopes.specificity(sd, outside['perturbation'])
        for name in slopes.LAYERS:
            if name == 'response' and key in seen:   # the transferred response does not depend on the model
                continue
            s = outside[name]
            row = {'screen': screen, 'fold': fold, 'model': 'any' if name == 'response' else model, 'layer': name,
                   'calibration_lines': n_lines, 'calibration_units': len(sd['u']['truth']), 'slope': s,
                   'slope_bag': bag[name], 'judgment': slopes.judge(s), 'fraction': slopes.fraction(s),
                   'assigned': slopes.assigned(s), 'ordinary': amounts[name]}
            if name == 'perturbation':
                row |= specific | {'context_class': slopes.context_class(s, specific['S'])}
            readings.append(row)
            for n, hit in slopes.mock_readings(terms[name]):
                c = mock.setdefault((screen, name, n), [0, 0])
                c[0] += hit
                c[1] += 1
        seen.add(key)
        inert += [{'screen': screen, 'fold': fold, 'model': model, **r} for r in slopes.seed_difference_slopes(sd)]
        print(screen, fold, model, flush=True)
    fixed = {name: float(np.median([r['fraction'] for r in readings if r['layer'] == name])) for name in slopes.LAYERS}
    for r in readings:
        r['fixed'] = fixed[r['layer']]
    write_csv(out / 'CALIBRATION_READINGS.csv', readings)
    write_csv(out / 'RELIABILITY.csv', [{'screen': k[0], 'layer': k[1], 'calibrating_lines': k[2], 'hits': v[0], 'splits': v[1], 'share': v[0] / v[1]}
                                        for k, v in sorted(mock.items())])
    write_csv(out / 'INERT_LAYERS.csv', inert)


def calibration_rows(out):
    rows = {}
    for r in read_csv(Path(out) / 'CALIBRATION_READINGS.csv'):
        rows[(r['screen'], r['fold'], r['model'], r['layer'])] = {k: float(v) if k in FLOAT and v != '' else v for k, v in r.items()}
    return rows


def heldout_stage(out):
    calibration = calibration_rows(out)
    table, done_response = [], set()
    for screen, fold, model in CASES:
        key = (screen, fold)
        case = load_case(screen, fold, model, True)
        cal = read_side(screen, case, 'calibration') if screen in OUTSIDE_PARTNER else None
        sides = {chain: read_side(screen, case, chain) for chain in CHAINS}
        specific = {}
        for chain, sd in sides.items():
            s = slopes.seed_slopes(sd, 'outside')['perturbation']
            if screen in OUTSIDE_PARTNER:   # their calibration contexts trained the full model, so that chain has no partner
                specific[chain] = slopes.specificity(sd, s, cal) if chain == 'nested' else dict.fromkeys(('slope_matched', 'm', 'S', 'S_matched'))
            else:
                specific[chain] = slopes.specificity(sd, s)
        for name in slopes.LAYERS:
            if name == 'response' and key in done_response:
                continue
            row = dict(calibration[(screen, fold, 'any' if name == 'response' else model, name)])
            for chain, sd in sides.items():
                row |= slopes.held_out_columns(sd, name, row['fraction'], chain, specific[chain])
            table.append(row)
        done_response.add(key)
        print(screen, fold, model, flush=True)
    write_csv(Path(out) / 'READINGS.csv', table)


def read_readings(path):
    number = lambda v: float(v) if v != '' else None
    return [{k: v if k in TEXT else number(v) for k, v in r.items()} for r in read_csv(path)]


def amount(r, method):
    return r[method] if isinstance(method, str) else method


def groups(rows):
    """The whole set, then the splits by screen, layer and number of calibration lines."""
    yield 'all', '', rows
    for key, label in (('screen', 'screen'), ('layer', 'layer'), ('calibration_lines', 'calibration lines')):
        for value in sorted({r[key] for r in rows}, key=str):
            yield label, f'{value:g}' if isinstance(value, float) else value, [r for r in rows if r[key] == value]


def primary(rows):
    def row(chain, split, value, method, gap):
        hits = sum(g <= slopes.TOLERANCE + 1e-12 for g in gap)
        return {'chain': chain, 'split': split, 'value': value, 'method': method, 'hits': hits, 'readings': len(gap),
                'share': hits / len(gap), 'mean_gap': float(np.mean(gap))}

    out = []
    for chain in CHAINS:
        for split, value, part in groups(rows):
            for name, method in METHODS:
                out.append(row(chain, split, value, name, [abs(amount(r, method) - r[f'best_{chain}']) for r in part]))
            out.append(row(chain, split, value, 'this rule, need from the input bag',
                           [abs(slopes.fraction(r['slope_bag']) - slopes.fraction(r[f'heldout_slope_bag_{chain}'])) for r in part]))
    return out


def curve(rows):
    out = []
    for chain in CHAINS:
        for name, method in METHODS:
            gap = np.asarray([abs(amount(r, method) - r[f'best_{chain}']) for r in rows])
            out += [{'chain': chain, 'method': name, 'tolerance': float(t), 'hits': int((gap <= t + 1e-12).sum()), 'readings': len(rows),
                     'share': float((gap <= t + 1e-12).mean())} for t in TOLERANCES]
    return out


def rank(rows):
    out = []
    corrections = [r for r in rows if r['layer'] != 'response']
    for chain in CHAINS:
        for scope, part in (('all layers', rows), ('line and perturbation', corrections),
                            *((name, [r for r in rows if r['layer'] == name]) for name in slopes.LAYERS)):
            out.append({'chain': chain, 'measure': 'calibration slope, held-out slope', 'scope': scope, 'readings': len(part),
                        'spearman': float(spearmanr([r['slope'] for r in part], [r[f'heldout_slope_{chain}'] for r in part]).statistic)})
    return out


def used(r, chain, keep=slopes.KEEP, empty=slopes.EMPTY):
    """(gain, hit) of the layer used as judged: kept or shrunk lowers error; no usable signal means the grid's smallest
    amount does not lower it; refused means no amount on the grid does."""
    judgment = slopes.judge(r['slope'], keep, empty)
    if judgment == 'keep':
        return r[f'gain_raw_{chain}'], r[f'gain_raw_{chain}'] > 0
    if judgment == 'shrink':
        return r[f'gain_fraction_{chain}'], r[f'gain_fraction_{chain}'] > 0
    return 0., (r[f'gain_eighth_{chain}'] if judgment == 'no usable signal' else r[f'gain_best_grid_{chain}']) <= 0


def use(rows):
    out = []
    for chain in CHAINS:
        for split, value, part in groups(rows):
            judged = [used(r, chain) for r in part]
            for name, gains, hits in (('used as judged', [g for g, _ in judged], sum(h for _, h in judged)),
                                      ('as it stands', [r[f'gain_raw_{chain}'] for r in part], sum(r[f'gain_raw_{chain}'] > 0 for r in part)),
                                      ('fixed 1/8', [r[f'gain_eighth_{chain}'] for r in part], sum(r[f'gain_eighth_{chain}'] > 0 for r in part))):
                out.append({'chain': chain, 'split': split, 'value': value, 'method': name, 'hits': int(hits), 'readings': len(part),
                            'share': hits / len(part), 'mean_gain_pts': float(np.mean(gains))})
        for judgment in ('keep', 'shrink', 'no usable signal', 'refuse'):
            part = [r for r in rows if r['judgment'] == judgment]
            if part:
                judged = [used(r, chain) for r in part]
                out.append({'chain': chain, 'split': 'judgment', 'value': judgment, 'method': 'used as judged', 'hits': sum(h for _, h in judged),
                            'readings': len(part), 'share': sum(h for _, h in judged) / len(part), 'mean_gain_pts': float(np.mean([g for g, _ in judged]))})
    return out


def s_match(rows, chain, cut=slopes.S_CUT, column='S'):
    """The perturbation-level readings with a held-out S on this chain: accepted against held-out S at the cut, and the
    three classes (a single calibration line's 'not accepted' reads as generic)."""
    held = f'heldout_{column}_{chain}'
    part = [r for r in rows if r['layer'] == 'perturbation' and r[held] is not None]
    cal = lambda r: slopes.context_class(r['slope'], r[column], cut).replace('not accepted', 'generic')
    two = sum((cal(r) == 'specific') == (r[held] >= cut) for r in part)
    three = sum(cal(r) == slopes.context_class(r[f'heldout_slope_{chain}'], r[held], cut) for r in part)
    return {'cards': len(part), 'accepted': sum(cal(r) == 'specific' for r in part), 'two_class_matches': two, 'three_class_matches': three,
            'accept_all_matches': sum(r[held] >= cut for r in part), 'refuse_all_matches': sum(r[held] < cut for r in part)}, part


def specificity(rows):
    out = []
    for chain in CHAINS:
        for column in ('S', 'S_matched'):
            summary, part = s_match(rows, chain, column=column)
            with_S = [r for r in part if r[column] is not None]
            out.append({'chain': chain, 'S': column, 'screen': 'all', **summary, 'spearman_calibration_heldout':
                        float(spearmanr([r[column] for r in with_S], [r[f'heldout_{column}_{chain}'] for r in with_S]).statistic)})
            for screen in sorted({r['screen'] for r in part}):
                out.append({'chain': chain, 'S': column, 'screen': screen, **s_match([r for r in part if r['screen'] == screen], chain, column=column)[0]})
    return out


def sensitivity(rows):
    out = []
    corrections = [r for r in rows if r['layer'] != 'response']
    for chain in CHAINS:
        for scope, part in (('all layers', rows), ('line and perturbation', corrections)):
            for label, cut, keep, empty in (*(('keep or shrink', c, c, slopes.EMPTY) for c in KEEP_CUTS),
                                            *(('no usable signal', c, slopes.KEEP, c) for c in EMPTY_CUTS)):
                judged = [used(r, chain, keep, empty) for r in part]
                classes = [slopes.judge(r['slope'], keep, empty) for r in part]
                out.append({'chain': chain, 'cut': label, 'value': cut, 'scope': scope, 'readings': len(part),
                            **{k.replace(' ', '_'): classes.count(k) for k in ('keep', 'shrink', 'no usable signal', 'refuse')},
                            'use_hits': sum(h for _, h in judged), 'mean_gain_pts': float(np.mean([g for g, _ in judged])),
                            'class_matches': sum(c == slopes.judge(r[f'heldout_slope_{chain}'], keep, empty) for c, r in zip(classes, part))})
        for cut in S_CUTS:
            out.append({'chain': chain, 'cut': 'S', 'value': cut, 'scope': 'perturbation cards', **s_match(rows, chain, cut)[0]})
    return out


def tables_stage(out):
    out = Path(out)
    rows = read_readings(out / 'READINGS.csv')
    for name, make in (('PRIMARY', primary), ('TOLERANCE_CURVE', curve), ('RANK_CORRELATION', rank), ('USE', use), ('SPECIFICITY', specificity),
                       ('SENSITIVITY', sensitivity)):
        write_csv(out / f'{name}.csv', make(rows), mode='w')   # derived: another cut rebuilds them


def main(argv=None):
    parser = argparse.ArgumentParser(description='The layers\' slopes on every screen, model and fold.')
    parser.add_argument('--stage', choices=('calibration', 'heldout', 'tables'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    {'calibration': calibration_stage, 'heldout': heldout_stage, 'tables': tables_stage}[args.stage](args.out)


if __name__ == '__main__':
    main()
