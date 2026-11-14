"""The figure tables of Figs. 5b-c and Supplementary Fig. 4b that no other stage writes, on Tahoe-100M's 17 restoring
changes and nine negative controls and on each original model's context-level correction. ``README.md`` maps every
panel of Figs. 5 and 6a-b to its table.

Stages, each reading the tables of earlier steps from ``--out`` and writing into it:

- ``heldout-parts``: from the saved predictions on held-out lines, with the amounts fixed on calibration lines
  (``AMOUNTS.json`` of ``net_cost --stage calibration``) and the changes of ``CARDS.csv``:
  ``CHANGE_PARTS_HELDOUT.csv``, per change and seed, the kept change (the original model plus the kept fraction of
  the change, calibrated toward the baseline) and the half change (the original model plus half of the change, not
  calibrated); ``CLOSEST_REFERENCE_HELDOUT.csv``, per model, fold and seed, the standard reference (no change, the
  baseline, the original as trained and its three shrinks) with the lowest held-out error. Total, common and contrast
  error minus the original model's, in points of the original's total MSE, on clipped predictions;
- ``figure5b``: ``FIG5_B_CHANGES_AND_COMPARATORS.csv``, Fig. 5b's held-out ΔMSE per change and comparator, means over
  seeds: per change, the change at full strength (``Full``) and at its kept fraction (``Kept``); per model and fold,
  continued training (``Continued``), weight decay (summary-statistics, DeepSets and State) and the best standard
  reference (``Reference``). Reads ``CARDS.csv``, the two tables above, ``PER_SEED_<fold>.csv`` (``il_cells --stage
  outer``) and ``WEIGHT_DECAY_READING.csv`` (``weight_decay``);
- ``figure5c``: ``FIG5_C_CORRECTION_BY_SCREEN.csv``, Fig. 5c: per screen, model and fold, the alignment (cosine) and
  best scale of the original model's context-level correction on held-out and training lines, and its held-out
  squared size relative to the baseline's common error, means over seeds, read from the cards' and the cost readings'
  saved alignment; the median cost of its restoring changes (``RESTORED_LIST.csv``); and the Replogle-Nadig fold-1
  average training response against the needed context-level correction on the held-out line. All values are written;
  Fig. 5c omits alignment and best scale where ``size_over_anchor_error_heldout`` is below 0.06;
- ``kept-fraction``: ``KEPT_FRACTION_INCREMENT.csv``, Supplementary Fig. 4b: per restoring change, its held-out error
  at full strength and, estimated, at its kept fraction before calibration, minus the original model. The estimate
  (``estimated_at_fraction``) is, per seed, the quadratic in the fraction through the clipped scores at strength 0, 1/2
  and 1, read at the seed's kept fraction; it is exact where no prediction is clipped. Reads ``CARDS.csv``,
  ``AMOUNTS.json``, ``PER_SEED_<fold>.csv`` (the per-seed costs whose mean is ``CARDS.csv``'s) and
  ``CHANGE_PARTS_HELDOUT.csv``.

    python -m reproduce.origin_resolution.figure_tables --stage figure5b --out DIR
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from scsailor.resolution.card import alignment
from scsailor.resolution.net_cost import actions, references
from scsailor.resolution.scoring import PARTKEY, scores
from scsailor.resolution.stages import bag_of
from . import ATTENTION_F23_READINGS, SEEDS, STATE_F23_READINGS, bag_means, heldout_units, read_csv, replogle, write_csv
from .baselines import GENETIC_CARDS, NESTED_F1, PROGRAM
from .net_cost import COST_ONLY, MODELS, read_side
from .paper_tables import MODEL, REFERENCE, SCREEN, change_name
from .slopes import load_case
from .tahoe import IL_NAME

__all__ = ["PART_KEYS", "TYPE", "points", "closest_reference", "estimated_at_fraction", "correction_table",
           "average_response_alignment", "heldout_parts_stage", "figure5b_stage", "figure5c_stage", "kept_fraction_stage",
           "main"]

# the parts of the clipped error, under the paper's names, and their keys in risk
PART_KEYS = tuple(('contrast' if part == 'differential' else part, key) for part, key in PARTKEY.items())
TYPE = {'restoring': 'restoring change', 'control': 'negative control'}   # saved tables: 'intervention recovering nothing'
FOLDS = ('F1', 'F2', 'F3')
SIDE = {'full parent, outer': 'held-out', 'fit': 'training'}   # the cards' readings of the original model
# The saved alignment of each original model's context-level correction: the cost readings' (screens read on the full
# chain only, ALIGNMENT.csv beside COST.csv), then the cards' readings (Tahoe-100M; the built models on Replogle-Nadig
# and Jiang et al.)
CARD_READINGS = (PROGRAM, ATTENTION_F23_READINGS / 'nested', STATE_F23_READINGS / 'nested', *GENETIC_CARDS.values(), NESTED_F1)
AVERAGE_RESPONSE = 'Average training response'


def points(score, base):
    """Each part of ``score`` minus the original model's, in points of the original's total MSE."""
    return {part: 100 * (score[key] - base[key]) / base['total_mse'] for part, key in PART_KEYS}


def prefixed(prefix, pts):
    return {f'{prefix}_{part}_pts': v for part, v in pts.items()}


def closest_reference(refs, base):
    """The reference of ``refs`` (name -> scores) with the lowest total, and its parts against ``base``."""
    name = min(refs, key=lambda n: refs[n]['total_mse'])
    return name, points(refs[name], base)


def estimated_at_fraction(cost, half, fraction):
    """The change's error at ``fraction`` of it, estimated by the quadratic in the fraction through 0 at strength 0,
    ``half`` at 1/2 and ``cost`` at 1. The error of p + l (R - p) is that quadratic where no prediction is clipped."""
    a, b = 2 * cost - 4 * half, 4 * half - cost
    return a * fraction * fraction + b * fraction


def correction_table(alignment_rows, changes):
    """Per screen, fold and model: the means over seeds of the cosine and best scale on held-out and training lines and
    of the held-out size relative to the baseline's common error (``alignment_rows``: screen, fold, model, seed, side,
    cosine, best_scale, size_over_anchor_error); the median cost of the restoring changes of ``changes``."""
    means = alignment_rows.pivot_table(index=['screen', 'fold', 'model'], columns='side', aggfunc='mean',
                                       values=['cosine', 'best_scale', 'size_over_anchor_error'])
    means.columns = [f"{value}_{side.replace('held-out', 'heldout')}" for value, side in means.columns]
    restoring = changes[changes.kind == 'restoring']
    cost = restoring.groupby(['screen', 'fold', 'model'], as_index=False).cost_pts.median().rename(columns={'cost_pts': 'median_cost'})
    table = means.reset_index().merge(cost, on=['screen', 'fold', 'model'], how='outer')
    table['Screen'], table['Model'] = table.screen.map(SCREEN), table.model.map(MODEL)
    return table[['Screen', 'fold', 'Model', 'cosine_heldout', 'cosine_training', 'best_scale_heldout', 'best_scale_training',
                  'size_over_anchor_error_heldout', 'median_cost']]


def average_response_alignment(units, bag, response):
    """The cosine and best scale, common part, of the average training response carried onto each held-out unit
    (``response``) against the needed correction, the truth minus the unit's input-bag mean ``bag``."""
    common = alignment(bag + response, bag, units)['common']
    return common['cosine'], common['best_scale']


# ---------------------------------------------------------------- stages
def tahoe_cards(out):
    return [c for c in read_csv(Path(out) / 'CARDS.csv') if c['screen'] == 'tahoe']


def heldout_parts_stage(out):
    out = Path(out)
    amounts = json.loads((out / 'AMOUNTS.json').read_text())
    parts = []
    for c in tahoe_cards(out):
        held = read_side('tahoe', load_case('tahoe', c['fold'], c['model'], True, c['cell']), 'held-out', c['cell'])
        u, saved = held['u'], amounts[c['card']]
        for s in SEEDS:
            p = held['p'][s]
            base = scores(p, u)
            kept = scores(actions(held, s, saved['kept'][str(s)], saved['model']['kappa'])[1], u)
            half = scores(p + .5 * (held['R'][s] - p), u)   # the half change, for the kept fraction's estimate
            parts.append({'card': c['card'], 'seed': s, **prefixed('kept', points(kept, base)), **prefixed('half', points(half, base))})
        print(c['card'], flush=True)
    write_csv(out / 'CHANGE_PARTS_HELDOUT.csv', parts)
    closest = []
    for fold in FOLDS:
        for model in MODELS:
            sd = read_side('tahoe', load_case('tahoe', fold, model, True), 'held-out', None)
            for s in SEEDS:
                base = scores(sd['p'][s], sd['u'])
                refs = {n: scores(x, sd['u']) for n, x in references(sd, s, amounts[f'tahoe|{fold}|{model}'])[1].items()}
                name, pts = closest_reference(refs, base)
                closest.append({'fold': fold, 'model': model, 'seed': s, 'closest': name, **prefixed('closest', pts)})
            print('tahoe', fold, model, flush=True)
    write_csv(out / 'CLOSEST_REFERENCE_HELDOUT.csv', closest)


def per_seed(out):
    """(fold, model, cell, seed) -> the il_cells reading of a change on held-out lines, models by origin resolution's names."""
    name = {il: model for model, il in IL_NAME.items()}
    return {(r['fold'], name.get(r['model'], r['model']), r['cell'], int(r['seed'])): r
            for fold in FOLDS for r in read_csv(Path(out) / f'PER_SEED_{fold}.csv')}


def fig5b_row(model, fold, prediction, values, change='', kind='', kept_fraction='', reference=''):
    """A Fig. 5b row: the means over seeds of (total, common, contrast) in ``values``."""
    return {'Screen': SCREEN['tahoe'], 'fold': fold, 'Model': MODEL[model], 'Change': change, 'Type': kind, 'prediction': prediction,
            'total_vs_original': mean(v[0] for v in values), 'common_vs_original': mean(v[1] for v in values),
            'contrast_vs_original': mean(v[2] for v in values), 'seeds': len(values), 'kept_fraction': kept_fraction,
            'reference': reference}


def figure5b_stage(out):
    out = Path(out)
    readings = per_seed(out)
    parts = {(r['card'], int(r['seed'])): r for r in read_csv(out / 'CHANGE_PARTS_HELDOUT.csv')}
    closest = {(r['fold'], r['model'], int(r['seed'])): r for r in read_csv(out / 'CLOSEST_REFERENCE_HELDOUT.csv')}
    decay = {(r['fold'], r['model'], int(r['seed'])): r for r in read_csv(out / 'WEIGHT_DECAY_READING.csv') if r['seed'] != 'mean'}
    cards = tahoe_cards(out)
    # (total, common, contrast) of one reading; il_cells' tables name the contrast part 'differential'
    triple = lambda r, prefix, contrast='differential': (float(r[f'{prefix}_total_pts']), float(r[f'{prefix}_common_pts']),
                                                        float(r[f'{prefix}_{contrast}_pts']))
    rows = []
    for model in MODELS:
        for fold in FOLDS:
            mine = [c for c in cards if (c['model'], c['fold']) == (model, fold)]
            first = mine[0]['cell']   # continued training is the same beside every change of a model and fold
            rows.append(fig5b_row(model, fold, 'Continued', [triple(readings[(fold, model, first, s)], 'continuation') for s in SEEDS]))
            if (fold, model, SEEDS[0]) in decay:   # weight decay has no attention-model run
                # the reading is the original minus weight decay (positive: weight decay has the lower error); negated
                rows.append(fig5b_row(model, fold, 'Weight decay', [tuple(-float(decay[(fold, model, s)][k]) for k in (
                    'wd_vs_original', 'wd_vs_original_common', 'wd_vs_original_contrast')) for s in SEEDS]))
            refs = [closest[(fold, model, s)] for s in SEEDS]
            rows.append(fig5b_row(model, fold, 'Reference', [triple(r, 'closest', 'contrast') for r in refs],
                                  reference=' / '.join(REFERENCE[r['closest']] for r in refs)))
            for c in mine:
                change, kind = change_name(c['cell']), TYPE[c['kind']]
                rows.append(fig5b_row(model, fold, 'Full', [triple(readings[(fold, model, c['cell'], s)], 'cell') for s in SEEDS],
                                      change, kind))
                rows.append(fig5b_row(model, fold, 'Kept', [triple(parts[(c['card'], s)], 'kept', 'contrast') for s in SEEDS],
                                      change, kind, kept_fraction=float(c['alpha_mean'])))
    write_csv(out / 'FIG5_B_CHANGES_AND_COMPARATORS.csv', rows)


def saved_alignment():
    """The saved alignment rows, common part: screen, fold, model, seed, side, cosine, best_scale and
    size_over_anchor_error, the cost readings' first; a model, fold, seed and side read twice keeps the first."""
    rows = []
    for path, model, _ in COST_ONLY:
        t = pd.read_csv(path.with_name('ALIGNMENT.csv'))
        rows.append(t[t.part == 'common'].assign(model=model).rename(columns={'dataset': 'screen'}))
    for folder in CARD_READINGS:
        for path in sorted(folder.glob('READINGS_*.csv')):
            _, screen, fold, model = path.stem.split('_', 3)
            if screen == 'tahoe' or (screen in ('jiang', 'replogle') and model in ('summary', 'deepsets')):
                t = pd.read_csv(path)
                t = t[t.reading.isin(list(SIDE))]
                rows.append(pd.DataFrame({'screen': screen, 'fold': fold, 'model': model, 'seed': t.seed, 'side': t.reading.map(SIDE),
                                          'cosine': t.common_cosine, 'best_scale': t.common_best_scale,
                                          'size_over_anchor_error': t.common_size_over_anchor_error}))
    columns = ['screen', 'fold', 'model', 'seed', 'side', 'cosine', 'best_scale', 'size_over_anchor_error']
    return pd.concat([r[columns] for r in rows], ignore_index=True).drop_duplicates(['screen', 'fold', 'model', 'seed', 'side'])


def replogle_average_response():
    """Replogle-Nadig fold 1: each held-out unit's average training response for its perturbation against the needed
    context-level correction."""
    files = replogle.files('F1')
    u = heldout_units(files['outer'])
    bag = bag_of(bag_means(files['inputs'])['heldout'], u)[0]
    with np.load(files['constants']) as constants:
        response = constants['r_input'].astype(np.float64)[u['p_code']]
    return average_response_alignment(u, bag, response)


def figure5c_stage(out):
    table = correction_table(saved_alignment(), pd.read_csv(Path(out) / 'RESTORED_LIST.csv'))
    cosine, best = replogle_average_response()
    average = pd.DataFrame([{'Screen': SCREEN['replogle'], 'fold': 'F1', 'Model': AVERAGE_RESPONSE, 'cosine_heldout': cosine,
                             'best_scale_heldout': best}])
    table = pd.concat([table, average], ignore_index=True).sort_values(['Screen', 'fold', 'Model'])
    table.to_csv(Path(out) / 'FIG5_C_CORRECTION_BY_SCREEN.csv', index=False)


def kept_fraction_stage(out):
    out = Path(out)
    amounts = json.loads((out / 'AMOUNTS.json').read_text())
    readings = per_seed(out)
    half = {(r['card'], int(r['seed'])): float(r['half_total_pts']) for r in read_csv(out / 'CHANGE_PARTS_HELDOUT.csv')}
    rows = []
    for c in tahoe_cards(out):
        if c['kind'] != 'restoring':
            continue
        kept = [amounts[c['card']]['kept'][str(s)]['alpha'] for s in SEEDS]
        at_kept = [estimated_at_fraction(float(readings[(c['fold'], c['model'], c['cell'], s)]['cell_total_pts']), half[(c['card'], s)], k)
                   for s, k in zip(SEEDS, kept)]
        rows.append({'fold': c['fold'], 'Model': MODEL[c['model']], 'Change': change_name(c['cell']), 'kept_fraction': float(c['alpha_mean']),
                     'full_strength_pts': float(c['cost_total_pts']), 'at_kept_fraction_estimated_pts': float(np.mean(at_kept))})
    write_csv(out / 'KEPT_FRACTION_INCREMENT.csv', rows)


STAGES = {'heldout-parts': heldout_parts_stage, 'figure5b': figure5b_stage, 'figure5c': figure5c_stage,
          'kept-fraction': kept_fraction_stage}


def main(argv=None):
    parser = argparse.ArgumentParser(description='The figure tables of Figs. 5b-c and Supplementary Fig. 4b.')
    parser.add_argument('--stage', choices=tuple(STAGES), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    STAGES[args.stage](args.out)


if __name__ == '__main__':
    main()
