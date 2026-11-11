"""Tahoe-100M's binding: the case for one model and fold, the menu's plate-aware target, the dataset's
documented origin rows and its evidence for the computed ones.

A context is a cell line on one of two replicate plates (6 and 14); a drug x plate well pools every line, so each
line x drug unit has a twin on the other plate. Folds 1-3 hold 14 lines out each; the fold's fixed line order puts 21
training lines in the nested chain and keeps the last 7 as calibration lines (``NESTED``). Files of a fold come from
the data root (``files``).

Parents (40k of the 160k schedule, three seeds), each with a nested parent fitted on the first 21 lines and a full
parent on all 28:

- summary-statistics and DeepSets: from information localization's real-task runs of the model's default cell
  (``CELL``), whose parent grids they carry, or of the cell given;
- the attention model: origin resolution's head chains (their step-0 grids), or, with a restoring change, IL's runs of
  that change, so that parent and change end are one prediction pass;
- State (v0.11.3, anchored output, L2): origin resolution's grids at its fixed length, 5k of the 160k
  schedule, or on fold 1 at another length of its own trajectory (``state_grids``); folds 2-3 from their package
  (``STATE_F23``). It is a published model: its menu adds the correction by each built model of the fold
  (``IN_HOUSE``).

The menu's target is the plate-aware anchor: the training lines' drug averages per plate, each context's plate read
from its input bag (``menu``).
"""
import json
from functools import cache
from pathlib import Path

import h5py
import numpy as np

from scsailor.resolution import attribution, grids, menu
from scsailor.resolution.stages import LABEL
from scsailor.resolution.scoring import PARTKEY, PARTS, identities, retained
from . import (ATTENTION_F23_READINGS, ROOT, REFERENCES, SEEDS, STATE_F23, STATE_F23_READINGS, at, bag_means,
          cell_menu_inputs, heldout_units, input_cells, loader, per_unit, read_csv, runs, training_units)

__all__ = ["SEEDS", "NESTED", "CELL", "IL_NAME", "IN_HOUSE", "PUBLISHED", "files", "line_order", "units", "line_of", "stratum_of",
           "calibration_lines", "pool_cells", "state_root", "state_grids", "grid_paths", "case", "drug_effects", "design_shift", "menu_inputs",
           "selection_path", "evidence", "STATIC", "STATIC_OFFICIAL_STATE", "FOLD1_ONLY"]

ORIGINALS = ROOT / 'origin-resolution'
PROGRAM = ORIGINALS / 'cards'   # the attribution program, whose references the stages read
CHECKS = ORIGINALS / 'checks'   # fold 1's FLOORS.csv (D1, D2 and their sampling terms)
NF = ROOT / 'tahoe-100m-noise-floor'   # fold 1's per-unit D1 and per-gene variance
ORGANS = ROOT / 'tahoe-100m/cell_line_metadata.parquet'   # Tahoe-100M's cell-line metadata, unchanged
NESTED = 21   # lines of the nested chain; the rest of the fold's line order are its calibration lines
CELL = {'summary': 'projections/e/x30', 'deepsets': 'phi/a/x1'}   # the restoring changes whose runs carry the parents
IL_NAME = {'attention': 'state'}   # IL's name for the built attention model
PUBLISHED = ('official_state',)
PLATES = ('plate6', 'plate14')   # the two replicate plates, in the order of the plate contrast
IN_HOUSE = {'F1': ('summary', 'deepsets', 'attention'), 'F2': ('summary', 'deepsets'), 'F3': ('summary', 'deepsets')}
# The attention model's parents per fold: origin resolution's head chains (step 0 of each: 40k of the 160k schedule, on 28
# lines and on the first 21 lines of the line order), the ones the repair selections read.
ATTENTION = {'F1': ROOT / 'origin-resolution/attention',
             **{fold: ROOT / f'origin-resolution/folds/{fold}/attention' for fold in ('F2', 'F3')}}
# State's grids: the nested parents on the first 21 lines and the 28-line parents at 5k (state-nested-chains); past 5k,
# the continued nested parents and the 28-line fit's own saved predictions (10k to 80k).
OFFICIAL_STATE = ROOT / 'state-nested-chains'
STATE_LONGER = ROOT / 'state-nested-chains'
REPAIR_SELECTIONS = ORIGINALS / 'selections/results'   # the repair selections (the attention model without a cell)
GROUPS = REFERENCES / 'tahoe/PERTURBATION_GROUPS.csv'   # the declared perturbation groups (drug mechanism classes), every fold
IL_SELECTIONS = ORIGINALS / 'restoring-changes/results'   # the restoring changes' selections (``il_cells``)
F23_READINGS = {'official_state': STATE_F23_READINGS, 'attention': ATTENTION_F23_READINGS}   # folds 2-3, where not IL_SELECTIONS


def files(fold):
    """The fold's files: training targets and controls, model constants, held-out units, line order, input library."""
    if fold == 'F1':
        train, inputs = ROOT / 'training-arrays-parse-pbmc-tahoe-100m/training-arrays/tahoe_F1', ROOT / 'tahoe-100m-inputs-fold-1/inputs'
        outer, lines = ROOT / 'tahoe-100m-evaluation-units-fold-1/scoring/inputs/EVAL_UNITS.npz', ROOT / 'tahoe-100m-line-subsets/LINE_SUBSETS.json'
    else:
        train = inputs = ROOT / 'tahoe-100m-inputs-folds-2-3' / fold
        outer = ROOT / f'tahoe-100m-evaluation-units-folds-2-3/EVAL_UNITS_{fold}.npz'
        lines = ROOT / f'tahoe-100m-line-subsets/folds-2-3/{fold}/LINE_SUBSETS.json'
    return {'train_targets': train / 'train_targets.npz', 'constants': inputs / 'training_constants.npz', 'outer': outer,
            'line_subsets': lines, 'inputs': inputs / 'evaluation_inputs.h5', 'controls': train / 'controls.h5'}


def line_order(fold):
    return json.loads(files(fold)['line_subsets'].read_text())['line_order']


def units(fold, side):
    """The fold's scoring units: 'train' (no held-out truth is read) or 'outer'."""
    f = files(fold)
    return training_units(f['train_targets'], f['constants']) if side == 'train' else heldout_units(f['outer'])


def line_of(context):
    return context.split('|')[0]


def stratum_of(context):
    """The design stratum of a context: its plate."""
    return context.split('|')[1]


def calibration_lines(fold):
    return line_order(fold)[NESTED:]


def pool_cells(fold):
    return input_cells(Path(files(fold)['inputs']).parent)


def state_root(fold):
    """The folder of State's grids on a fold (its nested, 28-line and 5k-to-10k fits)."""
    return OFFICIAL_STATE if fold == 'F1' else STATE_F23 / fold


def state_grids(seed, length, fold='F1'):
    """State's nested and full grids at a length of its trajectory: at 5k the nested and 28-line pair; past 5k the
    nested parent continued from its 5k state and the 28-line fit's own saved prediction at that length. Folds 2-3
    have the pair at 5k only."""
    if length == 5000:
        return {'nested': state_root(fold) / f'st_anchored_or21_L2_{seed}/predictions-or-grid.npz',
                'full': state_root(fold) / f'full_train28_L2_{seed}/predictions-or-grid.npz'}
    if fold != 'F1':
        raise ValueError(f"official State on {fold} has origin resolution's grids at its fixed length (5000) only")
    full = OFFICIAL_STATE if length == 10000 else STATE_LONGER
    return {'nested': STATE_LONGER / f'st_anchored_or21_L2_{seed}_longer/predictions-or-grid-step{length}.npz',
            'full': full / f'continued_train28_L2_{seed}/predictions-or-grid-step{length}.npz'}


def grid_paths(model, fold, seeds, cell=None, length=5000):
    """seed -> prediction files: 'nested' and 'full' parents, and with a cell its ends on both ('cell_nested',
    'cell_full'). A model's default cell (``CELL``) applies when none is given."""
    cell = cell or CELL.get(model)
    if model == 'official_state':
        paths = {s: state_grids(s, length, fold) for s in seeds}
    elif model == 'attention' and cell is None:
        paths = {s: {'nested': ATTENTION[fold] / f'head-ordinary-{s}-n21-nestedparent/predictions-step00000.npz',
                     'full': ATTENTION[fold] / f'head-ordinary-{s}/predictions-step00000.npz'} for s in seeds}
    if cell is not None:
        il = IL_NAME.get(model, model)
        found, states = runs.cell_runs(runs.find_runs(runs.MODELS[fold][il]['runs']), il, fold, cell, files(fold), line_order(fold))
        if any(s != 'complete' for s in states.values()):
            raise ValueError(f'Tahoe {model}, {cell}: runs not complete: {states}')
        if model != 'official_state':   # the parents come from the cell's runs
            paths = {s: {'nested': found[(s, 'n21')]['parent'], 'full': found[(s, 'n28')]['parent']} for s in seeds}
        for s in seeds:
            paths[s] |= {'cell_nested': found[(s, 'n21')]['end'], 'cell_full': found[(s, 'n28')]['end']}
    return paths


def case(model, fold='F1', outer=True, seeds=SEEDS, cell=None, length=5000):
    """The case (``stages``). outer=False builds no held-out units, so a calibration stage cannot read held-out
    truth."""
    f = files(fold)
    train = units(fold, 'train')
    held = units(fold, 'outer') if outer else None
    order = line_order(fold)
    fit, calibration = order[:NESTED], order[NESTED:]
    paths = grid_paths(model, fold, seeds, cell, length)
    lines = identities(train)
    is_fit, is_cal = np.isin(lines, fit), np.isin(lines, calibration)
    unit_sets = {'fit': retained(train, is_fit), 'calibration': retained(train, is_cal), 'outer': held, 'outer_nested': held}
    # D1 per unit: fold 1 from the noise-floor tables; folds 2 and 3 from the dataset references
    noise_train, noise_outer, column = ((NF / 'UNIT_NOISE.csv', NF / 'heldout/UNIT_NOISE.csv', 'noise_energy') if fold == 'F1' else
                                        (REFERENCES / f'tahoe-{fold}/UNIT_NOISE_TRAIN.csv', REFERENCES / f'tahoe-{fold}/UNIT_NOISE_OUTER.csv', 'noise'))
    rows = {int(r['train_row']): float(r[column]) for r in read_csv(noise_train)}   # training side, keyed by training row
    train_noise = np.asarray([rows[i] for i in range(len(train['truth']))])
    noise = {'fit': train_noise[is_fit], 'calibration': train_noise[is_cal],
             'outer': per_unit(noise_outer, held, column) if outer else None}
    noise['outer_nested'] = noise['outer']
    gene_variance = NF / 'GENE_VARIANCE.npy' if fold == 'F1' else REFERENCES / f'tahoe-{fold}/GENE_VARIANCE_MEAN.npy'   # per unit / its mean
    static = {**STATIC, **(STATIC_OFFICIAL_STATE if model == 'official_state' else {})}
    fold1_only = FOLD1_ONLY + (('M2',) if model == 'official_state' else ()) if fold != 'F1' else ()   # State's M2: fold 1's map
    # each held-out line x drug unit is measured on both replicate plates (half 0: plate 6, half 1: plate 14)
    replicates = ({'line': identities(held), 'half': np.asarray([int(c.endswith('|plate14')) for c in held['context']]),
                   'twin': attribution.replicate_twins(held, plates=PLATES), 'label': 'between replicate plates (a plate-blind reference, not a ceiling)'}
                  if outer else None)
    return {'dataset': 'tahoe', 'fold': fold, 'model': model, 'units': unit_sets, 'grids': paths, 'load': loader(paths),
            'bags': bag_means(f['inputs']), 'noise': noise, 'fit': fit, 'calibration_lines': calibration,
            'meta': {'coincident': None, 'type_variable': None},
            'train_files': {'targets': f['train_targets'], 'constants': f['constants'], 'controls': f['controls']},
            'gene_variance': np.load(gene_variance).mean(axis=0) if outer else None,
            'static_rows': {'static': static, 'fold1_only': fold1_only, 'seed_permutation': 'see the existing cards'},
            'replicates': replicates}


def drug_effects(fold, lines):
    """r[drug] and r_plate[plate][drug] from the training units of ``lines``: the weighted mean of the target minus its
    context's control-pool mean, over both plates or per plate (training data only; menu.plate_effects)."""
    f = files(fold)
    with np.load(f['train_targets']) as z, np.load(f['constants']) as c:
        keys, n = c['context_keys'].astype(str), c['r_input'].shape[0]
        target, parent = z['target'].astype(np.float64), z['parent']
        code, w = z['p_code'].astype(np.int64), z['weight'].astype(np.float64)
    with h5py.File(f['controls'], 'r') as h:
        means = {int(k): h[k]['input/x'][:].mean(axis=0, dtype=np.float64) for k in h}
    change = target - np.stack([means[int(q)] for q in parent])
    line = np.array([keys[q].split('|')[0] for q in parent])
    plate = np.array([keys[q].split('|')[1] for q in parent])
    return menu.plate_effects(change, line, plate, code, w, lines, n, plates=PLATES)


def chain_lines(fold, nested):
    order = line_order(fold)
    return order[:NESTED] if nested else order


@cache
def chain_effects(fold, nested):
    return drug_effects(fold, chain_lines(fold, nested))


def plate_shift(case, u, nested, bank):
    """``menu.plate_aware_shift`` on the chain's lines (``bank``: the side of the input sets): per unit the plate-aware target
    minus the anchor, and the plates' check."""
    r, r_plate = chain_effects(case['fold'], nested)
    return menu.plate_aware_shift(u, r, r_plate, case['bags']['train'], chain_lines(case['fold'], nested), case['bags'][bank],
                                  plates=PLATES)


def design_shift(case, u, nested, bank):
    return plate_shift(case, u, nested, bank)[0]


def menu_inputs(case, which, seeds, a):
    """``menu.side``'s Tahoe inputs on one side ('calibration': the nested chain on the calibration units, the 21 fit
    lines' drug effects; 'outer': the full chain on the held-out units, all 28 training lines): the plate-aware
    target's shift and its plate check, the restoring change's end where the case has one, and, for a published model,
    each built model's parents of the fold (their anchors must equal the model's anchors ``a``, seed -> array)."""
    u = case['units']['calibration' if which == 'calibration' else 'outer']
    chain, side, bank = ('nested', 'train', 'train') if which == 'calibration' else ('full', 'outer', 'heldout')
    shift, check = plate_shift(case, u, which == 'calibration', bank)
    out = {'plate_shift': shift, 'plate_check': check} | cell_menu_inputs(case, which, seeds, a)
    if case['model'] in PUBLISHED:
        anchor = a[seeds[0]]
        out['options'] = {}
        for m in IN_HOUSE.get(case['fold'], ()):
            other = grid_paths(m, case['fold'], seeds)
            gap = float(np.abs(grids.load(other[seeds[0]][chain], side, u, 'anchor_raw') - anchor).max())
            if gap > 1e-5:
                raise ValueError(f"tahoe {case['fold']}: {m} has another anchor than {case['model']} (gap {gap:.2e}); no shared target")
            out['options'][m] = {s: grids.load(other[s][chain], side, u) for s in seeds}
    return out


def selection_path(model, fold, cell, out):
    """The saved selection the calibration stage reads: the repair selections' for the attention model without a
    restoring change, the selection stage's own (in ``out``) for State without one, else the change's (State's and the
    attention model's on folds 2-3 in the stages' readings, ``F23_READINGS``)."""
    if cell is None and model == 'attention':
        return REPAIR_SELECTIONS / f'SELECTIONS_{fold}.json'
    if cell is None and model == 'official_state':
        return Path(out) / f'SELECTION_tahoe_{fold}_official_state.json'
    folder = F23_READINGS[model] / 'il-cells' if model in F23_READINGS and fold != 'F1' else IL_SELECTIONS
    return folder / f"SELECTIONS_{fold}_{IL_NAME.get(model, model)}_{(cell or CELL[model]).replace('/', '-')}.json"


def evidence(case, held, mean_parts, pool, parent_part, preds):
    """D2, D7, D8 and T1 for attribution's rows, with their reference rows: D2 from the units' own twins and the
    references' floors and well-shared split; D7 by the plate test; T1 by Tahoe-100M's organ annotation (metadata
    only) where it is local."""
    fold = case['fold']
    ev, refs, where_rows = {}, [], []
    twins = case['replicates']['twin']
    versions = {'treated': attribution.d2_parts(held, twins), 'response': attribution.d2_parts(held, twins, pool)}
    ev['d2_by_context'] = attribution.d2_by_context(held, twins)
    undefined = sum(v is None for v in ev['d2_by_context'].values())
    # fold 1: the attribution checks' FLOORS.csv and references/tahoe/; folds 2-3: references/tahoe-<fold>/
    floors_file = CHECKS / 'FLOORS.csv' if fold == 'F1' else REFERENCES / f'tahoe-{fold}/FLOORS.csv'
    split_file = REFERENCES / 'tahoe/D2_TAHOE_SPLIT.csv' if fold == 'F1' else REFERENCES / f'tahoe-{fold}/D2_TAHOE_SPLIT.csv'
    floors = {(r['kind'], r['version']): r for r in read_csv(floors_file) if r['side'] == 'heldout'}
    values = lambda r: {'total': float(r['total']), 'common': float(r['common']), 'differential': float(r['difference'])}
    terms = {'treated': (('cell sampling (D1 of the twins)', ('cell-sampling floor (D1)', '')),
                         ('excess over cell sampling', ('D2 excess over cell sampling', 'treated'))),
             'response': (('cell sampling (D1 of the twins)', ('cell-sampling floor (D1)', '')),
                          ('control-pool sampling of C (estimated from the evaluation bag)', ('control-pool sampling of C (response version)', 'estimated from the evaluation bag')),
                          ('excess over cell and control-pool sampling', ('D2 excess over cell sampling and C sampling', 'response')))}
    text = []
    for name, v in versions.items():
        own = {part: v[key] for part, key in PARTKEY.items()}
        saved = values(floors[('between-plate discrepancy (D2)', name)])
        if abs(saved['total'] - own['total']) > 1e-9 * own['total']:   # the excess below is FLOORS.csv's, so its D2 must be this one
            raise ValueError(f"Tahoe D2 ({name}) {own['total']} differs from FLOORS.csv's {saved['total']}")
        refs.append({'reference': f'D2 {name}: between-plate twin deviations', 'support': f"{v['units']} units", **own,
                     'source': 'card (card.d2_parts)'})
        refs += [{'reference': f'D2 {name}: {label}', 'support': f"{v['units']} units", **values(floors[k]),
                  'source': 'checks/FLOORS.csv'} for label, k in terms[name]]
        excess = values(floors[terms[name][-1][1]])
        text.append(f"{name}: " + ', '.join(f"{LABEL[part]} {own[part] / parent_part[part]:.3f}" for part in PARTS)
                    + " of the parent's part; " + terms[name][-1][0] + ' ' + ', '.join(f"{excess[part] / parent_part[part]:.3f}" for part in PARTS))
    split = {r['side']: r for r in read_csv(split_file) if r['row'] == 'total'}['heldout']
    for label, col in (('shared by the lines of a well', 'shared_contribution'), ('line-specific beyond cell sampling', 'line_specific_excess_contribution')):
        refs.append({'reference': f'D2 drug-difference excess: {label}', 'differential': float(split[col]),
                     'source': f'{split_file.relative_to(PROGRAM)} (the shared estimate assumes equal per-line sampling variances)'})
    ev['d2'] = {'status': 'reference' if versions['treated'] else 'undefined',
                'evidence': '; '.join(text) + f"; the drug-difference excess splits {float(split['shared_share_of_excess']):.0%} shared by the "
                            f"lines of a well, {float(split['line_specific_share_of_excess']):.0%} line-specific"
                            + (f", {float(split['single_line_share_of_excess']):.0%} in wells with one held-out line (not split)"
                               if split.get('single_line_share_of_excess') else '') + f" (approximate: equal per-line "
                            f"sampling variances assumed; {split_file.relative_to(PROGRAM)}); contexts without a twin: {undefined}"}
    ev['d8'] = {'status': 'balanced' if not undefined and (twins >= 0).all() else 'unbalanced',
                'evidence': f"{len(held['truth'])} outer units; " + ('every line x drug on both plates' if (twins >= 0).all()
                                                                    else f"{int((twins < 0).sum())} units without their other plate")}
    # T1's declared type on Tahoe: Tahoe-100M's organ annotation (metadata only), an input of attribution
    import pandas as pd
    meta = pd.read_parquet(ORGANS)
    organ_of = dict(zip(meta['Cell_ID_Cellosaur'].astype(str), meta['Organ'].astype(str)))
    ident = identities(held)
    w = held['weight'] / held['weight'].sum()
    rel = {l: float(mean_parts['total'][ident == l].sum() / (w[ident == l].sum() * mean_parts['total'].sum())) for l in np.unique(ident)}
    trained = {organ_of.get(l) for l in case['fit'] + case['calibration_lines']}
    absent = [l for l in rel if organ_of.get(l) not in trained]
    others = [v for l, v in rel.items() if l not in absent]
    worse = [l for l in absent if rel[l] > np.mean(others)]
    ev['t1'] = {'status': ('no absent type' if not absent else
                           'compatible (descriptive: one unit per absent type, above the others)' if worse else
                           'weakened (descriptive: one unit per absent type, no worse than the others)'),
                'evidence': (', '.join(f'{l} ({organ_of.get(l)}) relative error {rel[l]:.2f}' for l in absent)
                             + f'; lines of types present in training: {np.mean(others):.2f} on average ({min(others):.2f}-{max(others):.2f})')}
    tests = {part: attribution.plate_test(mean_parts, held, part, plates=PLATES) for part in ('differential', 'common')}
    pattern = ', '.join(f"plate 6 higher in {t['lines_plate6_higher']} of {t['lines']} lines ({LABEL[part]})" for part, t in tests.items())
    ev['d7'] = {'status': f"{tests['differential']['status']}: {pattern}",
                'evidence': '; '.join(f"{LABEL[part]}: line-sign p {t['p']:.2g} ({t['p_label']}; a shock shared by the plate pair's wells breaks "
                                      f"that, so the p is not a test across plate pairs); most extreme line {t['most_extreme']} "
                                      f"(z {t['most_extreme_z']:.2f}, descriptive)" for part, t in tests.items())}
    ev['reference_rows'], ev['where_rows'] = refs, where_rows
    return ev


# The rows attribution does not compute: (status, evidence and source, next check and cost), from the saved evidence.
# A documented property is never a lead.
STATIC = {
    'M1': ('supported for the synthetic cue only', "information localization's presence profiles read the planted cue, not the natural failure", 'none on natural units'),
    'M2': ('see the information localization card', "information localization's map contrasts, folds 1-3: summary-statistics has qualified cells; DeepSets: neither arm recovers behind phi", 'none'),
    'M3': ('not applicable', "the built models' loss is the squared error of the scored bag-mean targets, so the natural target is rewarded by construction", 'none'),
    'M4': ('supported (crossed tests)', 'the crossed tests; DeepSets blocked in all seeds', 'none'),
    'M6': ('not tested', "no symptom scan has run", 'symptom scan of local checkpoints plus an intervention on a flagged part: fits'),
    'D3': ('not tested', 'synthetic evidence for summary-statistics only', 'noise-redrawn refits: fits'),
    'D4': ('not supported by the E-test', 'all 4,722 training units detected; the outer E-test is not run (it has no discriminating power here)', 'control-well against control-well reference: not available'),
    'D5': ('undefined', "no reference state is stated for Tahoe lines", 'a stated reference state'),
    'D6': ('reference', 'distances against error: Spearman 0.29-0.49 across six models, folds 1-3; descriptive', 'none (the split is not run)'),
    'D8': ('balanced', "2,482 outer units; every line x drug on both plates", 'none'),
    'D9': ('weakened only', "Part 2 linear reader, folds 1-3 (own over borrowed controls)", 'none'),
    'T1': ('not computed', "the organ annotation of Tahoe-100M's cell-line metadata is the declared type (computed by the card when the metadata is local)", 'none'),
    'T4': ('holds', 'calibration files, the perturbation-group choice included, written before outer truth by a stage that loads no outer units', 'none'),
    'T5': ('holds for the readings used', 'positive controls under the same construction', 'none'),
    'T6': ('direction only', "continuing from 40k to 80k raises new-line error (DeepSets +6.10 pooled); below 40k untested", 'fits saving checkpoints below 40k: 3 per model'),
    'T7': ('not applicable', "pooled lines share well, drug, dose and time", 'none'),
    'T8': ('undefined', "no per-drug list of indirect action", 'a declared list'),
    'O2': ('holds', 'anchor provenance checked', 'none'),
}

# State is a published model with its own loss, checks and information localization map, so the built rows above
# do not describe it.
STATIC_OFFICIAL_STATE = {
    'M1': ('see the information localization card', "information localization's State map (information-localization-state); its presence profiles are not read by this card", 'none'),
    'M2': ('see the information localization card', "information localization's State map (information-localization-state/MAP_CARRIED*.csv): on the "
           "natural half a junction repair recovers the common mean (seeds 31002-31003); the head intervention recovers neither "
           "the statistic-related perturbation identity nor the common variance", "information localization's setting 3 on the nested parents"),
    'M3': ('not tested', "official State trains on the energy distance between predicted and observed cell sets (blur 0.05), not on "
           "the scored bag-mean error; the check under each model's own training loss covers CPA and PrePR-CT only",
           "the same check on State's loss: CPU"),
    'M4': ('not tested', 'the crossed tests ran on the built models only', "crossed tests from State's real-trained start: fits"),
    'T5': ('not used', "no synthetic reading enters this card", 'none'),
    'T6': ('weakened', "the fixed 5k is at the smallest unseen-line correction of State's saved trajectory and within 0.003 of its "
           "lowest held-out total", 'none'),
    'O2': ('holds', "both grids' anchors equal the base package's saved anchors (at most 3.1e-14) and origin resolution's "
           "evaluation-bank mean plus each model's own r_input (the package's local check); the full parent's grid is its saved "
           "5k prediction", 'none'),
}

# Rows whose saved evidence was read on fold 1 only; on folds 2-3 the row says so.
FOLD1_ONLY = ('M4', 'D4', 'T6', 'O2')

# What the headline adds when a recommendation that holds on held-out lines shrinks toward the plate-aware baseline.
