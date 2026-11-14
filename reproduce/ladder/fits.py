"""The per-fit table: every natural-task fit of the information ladder on the five screens, with its held-out scores
and its status under the fit rules, read from the saved per-fit score files under ``natural-scores/``.

A fit is one model trained on one screen, fold, training seed and ladder level, read at the model's fixed training
length (``UPDATES``). Its scores are the held-out total, common and contrast MSE of the model and of the baseline it is
compared with, both clipped at zero: for the built models, the baseline the model adds its correction to, read along
the model's own scoring path (the files call it the anchor); for the published models, the fold's baseline. The files
hold them in four forms:

* per-fit tables (built models everywhere except Tahoe-100M and Parse PBMC, and State and CPA on the genetic screens):
  one row per fit with the model's and the anchor's held-out MSE and their training MSE (``per_fit``);
* per-seed tables of the published models on Tahoe-100M and GSE306429: one row per seed, level and training length,
  each MSE divided by the fold baseline's total, with the scorer's record of that baseline beside it (``per_seed``);
* the built models on Tahoe-100M: one record per fit on fold 1, one long table on folds 2-3 (``tahoe_built``);
* Parse PBMC: each fit's weighted error sums per held-out donor (the files' ``unit`` column), with the anchor's sums for
  the built models and the fold baseline's (``unit_fits``); a fit's MSE is the sum over its donors divided by their
  mass.

The fit rules (``fit_rules``): a fit is set aside when its training total MSE is at least 0.99 of the anchor's
training total MSE (the anchor is the baseline prediction stored before clipping, scored like the model after clipping
at zero; only the per-fit tables and the Tahoe-100M built models' records hold training MSE), and then when its
held-out relative MSE exceeds 1.4 times the median over the fits of the same screen, fold, model and level that passed
the training rule, itself included. Seeds 31005-31007 are the replacement seeds trained where a seed failed training.

The files name the attention model ``state``, ``STATE-like`` or ``attention``, and State ``state_official``,
``state_pub`` or ``State`` (``CODE``); ``fit_table`` prints the paper's names.
"""
import csv
import json

import numpy as np
import pandas as pd

from ..origin_resolution import ROOT

__all__ = ["SCORES", "SCREENS", "FOLDS", "MODELS", "BUILT", "LEVELS", "UPDATES", "CODE", "MSE", "score_columns",
           "per_seed", "per_fit", "unit_sums", "unit_fits", "fit_rules", "table", "named", "no_change", "fit_table"]

SCORES = ROOT / 'natural-scores'
SCREENS = {'Tahoe-100M': 'tahoe-100m', 'GSE306429': 'gse306429', 'Parse PBMC': 'parse-pbmc',
           'Replogle–Nadig': 'replogle-nadig', 'Jiang et al.': 'jiang'}   # paper name -> folder
FOLDS = {'Tahoe-100M': ('F1', 'F2', 'F3'), 'GSE306429': ('F1', 'F2', 'F3'), 'Parse PBMC': ('G1', 'G2', 'G3'),
         'Replogle–Nadig': ('F1', 'F2', 'F3', 'F4'), 'Jiang et al.': ('F1', 'F2', 'F3', 'F4', 'F5', 'F6')}
MODELS = {'attention_model': 'Attention model', 'deepsets': 'DeepSets model', 'summary': 'Summary-statistics model',
          'state': 'State', 'cpa': 'CPA', 'prepr': 'PrePR-CT'}   # code -> the paper's name
BUILT = ('attention_model', 'deepsets', 'summary')
SMALL_MOLECULES = ('Tahoe-100M', 'GSE306429')   # PrePR-CT encodes small molecules only
PER_SEED = ('Tahoe-100M', 'GSE306429')   # screens whose published models' scores are saved as per-seed tables
LEVELS = {'L1': 'Mean', 'L2': '+Var', 'L3': '+Dist', 'L3d': '+DepCp', 'L4': 'Cells'}   # file label -> level
UPDATES = {'attention_model': 40000, 'deepsets': 40000, 'summary': 40000, 'state': 5000, 'cpa': 140000,
           'prepr': 10000}   # each model's fixed training length
SEEDS = ('31002', '31003', '31004', '31005', '31006', '31007')
FAILED_TRAINING, OUTLIER = .99, 1.4
CODE = {'state': 'attention_model', 'STATE-like': 'attention_model', 'attention': 'attention_model',
        'deepsets': 'deepsets', 'DeepSets': 'deepsets', 'summary': 'summary', 'Summary': 'summary',
        'state_official': 'state', 'state_pub': 'state', 'State': 'state', 'cpa': 'cpa', 'CPA': 'cpa',
        'prepr': 'prepr', 'PrePR-CT': 'prepr'}   # a model's label in the score files -> its code
# The prefix of the held-out columns of the per-fit tables (else 'heldout'). GSE306429's ladder support: the held-out
# prediction units whose perturbation is among the fold's eligible training units and whose compound has a structure
# PrePR-CT can encode (Supplementary Note S2.3; 479, 288 and 245 units in folds 1-3), on which every model is scored.
SIDE = {'GSE306429': 'heldout_matched99'}
MSE = ('total_mse', 'common_error_mse', 'difference_error_mse')
COLUMNS = ['fit_id', 'fit_status', 'screen', 'fold', 'model', 'seed', 'level', 'training_updates', 'model_mse_total',
           'model_mse_common', 'model_mse_contrast', 'baseline_mse_total', 'baseline_mse_common',
           'baseline_mse_contrast', 'relative_mse', 'delta_common_points', 'delta_contrast_points']


def score_columns(model, baseline):
    """The table's score columns of one fit from the model's and the baseline's (total, common, contrast) MSE."""
    out = {f'{who}_mse_{c}': v for who, mse in (('model', model), ('baseline', baseline))
           for c, v in zip(('total', 'common', 'contrast'), mse)}
    return out | {'relative_mse': model[0] / baseline[0],
                  'delta_common_points': 100 * (model[1] - baseline[1]) / baseline[0],
                  'delta_contrast_points': 100 * (model[2] - baseline[2]) / baseline[0]}


def baseline_of(path):
    """(total, common, contrast) MSE of the baseline record a published-model scorer wrote: its total and the two
    components' shares of it."""
    b = json.loads(path.read_text())['baseline']
    total, common, contrast = (b['total'], b['common'], b['d']) if 'total' in b else \
        (b['base_total'], b['base_common'], b['base_d'])
    return total, total * common, total * contrast


def per_seed(path, baseline, step):
    """(seed, level, model MSE, baseline MSE, None) per row of a published model's per-seed table at training length
    ``step``; the table divides each MSE by the baseline's total."""
    b = baseline_of(baseline)
    with open(path) as handle:
        for r in csv.DictReader(handle):
            if int(r['step']) == step and r['seed'] in SEEDS and r['level'] in LEVELS:
                yield r['seed'], r['level'], tuple(float(r[c]) * b[0] for c in MSE), b, None


def per_fit(path, side, model):
    """(seed, level, model MSE, anchor MSE, training ratio) per row of a per-fit table for ``model``; ``side`` is the
    prefix of the held-out columns and the training ratio is the model's training total over the anchor's."""
    with open(path) as handle:
        for r in csv.DictReader(handle):
            if CODE[r['model']] == model and r['seed'] in SEEDS and r['level'] in LEVELS:
                yield (r['seed'], r['level'], tuple(float(r[f'{side}_model_{c}']) for c in MSE),
                       tuple(float(r[f'{side}_anchor_{c}']) for c in MSE),
                       float(r['train_model_total_mse']) / float(r['train_anchor_total_mse']))


def tahoe_built(fold, model):
    """The built models on Tahoe-100M, as ``per_fit``: one record per fit on fold 1, a long table on folds 2-3; the
    model is the record's ``original`` column, its baseline the ``anchor``."""
    found = {}
    if fold == 'F1':
        for seed in SEEDS[:3]:
            for level in LEVELS:
                sides = json.loads((SCORES / f'tahoe-100m/F1/{model}_{seed}_{level}.json').read_text())['sides']
                found |= {(seed, level, side, column): tuple(values[c] for c in MSE)
                          for side in ('heldout', 'train') for column, values in sides[side]['risk_columns'].items()}
    else:
        with open(SCORES / 'tahoe-100m/built_models_F2_F3.csv') as handle:
            for r in csv.DictReader(handle):
                if (r['fold'], CODE[r['model']], int(r['step'])) == (fold, model, UPDATES[model]):
                    found[(r['seed'], r['level'], r['side'], r['column'])] = (
                        float(r['total_mse']), float(r['common_mse']), float(r['difference_mse']))
    for seed in SEEDS[:3]:
        for level in LEVELS:
            yield (seed, level, found[(seed, level, 'heldout', 'original')], found[(seed, level, 'heldout', 'anchor')],
                   found[(seed, level, 'train', 'original')][0] / found[(seed, level, 'train', 'anchor')][0])


def unit_sums(dataset):
    """The rows of the error sums per held-out line or donor (``held-out-unit-sums/``) of one screen (``dataset`` as the
    files name it: 'Tahoe', 'Parse'), in file order, with the model's code in ``model``. Its ``unit`` is the held-out
    line or donor; a fold baseline's row that appears in more than one part file is read once, from the first."""
    rows, seen = [], set()
    for part in sorted((SCORES / 'held-out-unit-sums').glob('part-*.csv')):
        with open(part) as handle:
            for r in csv.DictReader(handle):
                if r['dataset'] != dataset:
                    continue
                if r['kind'] == 'baseline':
                    if (r['fold'], r['unit']) in seen:
                        continue
                    seen.add((r['fold'], r['unit']))
                rows.append(dict(r, model=CODE.get(r['model'], r['model'])))
    return rows


def unit_fits(rows, leave_out=()):
    """(fold, model, seed, level) -> (model MSE, baseline MSE) from ``unit_sums`` rows: each sum over the held-out
    lines or donors not in ``leave_out``, in row order, divided by their mass. A built model's baseline is its own
    anchor (``fit_anchor`` rows), a published model's the fold baseline (``baseline`` rows)."""
    model, anchor, base = {}, {}, {}
    for r in rows:
        if r['unit'] in leave_out:
            continue
        v = np.array([float(r[c]) for c in ('mass', 'total', 'common', 'contrast')])
        if r['kind'] == 'baseline':
            base[r['fold']] = base.get(r['fold'], 0) + v
        else:
            target = anchor if r['kind'] == 'fit_anchor' else model
            key = (r['fold'], r['model'], r['seed'], r['level'])
            target[key] = target.get(key, 0) + v
    out = {}
    for key, v in model.items():
        b = anchor[key] if key[1] in BUILT else base[key[0]]
        out[key] = (tuple(float(x) for x in v[1:] / v[0]), tuple(float(x) for x in b[1:] / b[0]))
    return out


def sources(screen, fold, model, parse_fits):
    """The score rows of one screen, fold and model, as ``per_fit``; ``parse_fits`` are Parse PBMC's ``unit_fits``."""
    folder = SCORES / SCREENS[screen] / fold
    if screen == 'Parse PBMC':
        return ((s, lv, m, b, None) for (f, mo, s, lv), (m, b) in parse_fits.items() if (f, mo) == (fold, model))
    if screen == 'Tahoe-100M' and model in BUILT:
        return tahoe_built(fold, model)
    if screen in PER_SEED and model not in BUILT:
        baseline = folder / ('baseline.json' if screen == 'Tahoe-100M' else f'{model}_baseline.json')
        return per_seed(folder / f'{model}.csv', baseline, UPDATES[model])
    return per_fit(folder / f"{'built_models' if model in BUILT else model}.csv", SIDE.get(screen, 'heldout'), model)


def fit_rules(fits):
    """The status of each fit (dicts with screen, fold, model, level, seed, relative_mse, training_ratio) under the fit
    rules of the module docstring: 'kept' or 'excluded: <reason>'; a fit whose training ratio is None (no training MSE
    saved) meets the outlier rule only."""
    status = ['kept'] * len(fits)
    groups = {}
    for i, f in enumerate(fits):
        groups.setdefault((f['screen'], f['fold'], f['model'], f['level']), []).append(i)
    for members in groups.values():
        trained = []
        for i in members:
            ratio = fits[i]['training_ratio']
            if ratio is not None and ratio >= FAILED_TRAINING:
                status[i] = 'excluded: training failed (training total / training anchor >= 0.99)'
            else:
                trained.append(i)
        median = float(np.median([fits[i]['relative_mse'] for i in trained]))
        for i in trained:
            if fits[i]['relative_mse'] > OUTLIER * median:
                status[i] = (f"excluded: outlier ({fits[i]['relative_mse'] / median:.2f} x sibling median; rule: "
                             f"above 1.4 x the median of the kept sibling seeds)")
    return status


def table(screens, leave_out=()):
    """One row per fit of the screens' folds, applicable models, seeds and levels, with its scores and its status,
    sorted by ``fit_id``; Parse PBMC without the donors ``leave_out``. Model and level are codes (``named``)."""
    parse_fits = unit_fits(unit_sums('Parse'), leave_out) if 'Parse PBMC' in screens else {}
    rows = []
    for screen in screens:
        for fold in FOLDS[screen]:
            for model in MODELS:
                if model == 'prepr' and screen not in SMALL_MOLECULES:
                    continue
                for seed, level, m, b, ratio in sources(screen, fold, model, parse_fits):
                    rows.append(dict(fit_id=f'{screen}|{fold}|{model}|{seed}|{level}|{UPDATES[model]}', screen=screen,
                                     fold=fold, model=model, seed=seed, level=level, training_ratio=ratio,
                                     training_updates=UPDATES[model], **score_columns(m, b)))
    for row, status in zip(rows, fit_rules(rows)):
        row['fit_status'] = status
    return pd.DataFrame(rows).sort_values('fit_id', kind='stable').reset_index(drop=True)[COLUMNS]


def named(table):
    """``table`` with the paper's names of models and levels."""
    return table.assign(model=table['model'].map(MODELS), level=table['level'].map(LEVELS))


def no_change(screen, fold):
    """No change's held-out total MSE over the baseline's, from a fold's saved reference scores (GSE306429 and the
    genetic screens)."""
    side = SIDE.get(screen, 'heldout')
    r = pd.read_csv(SCORES / SCREENS[screen] / fold / 'no_change.csv').iloc[0]
    return r[f'{side}_input_only_raw_total_mse'] / r[f'{side}_anchor_raw_total_mse']


def fit_table():
    """Every fit of the five screens (``table``), with its adapted weighted R2, total and contrast versions
    (``weighted_r2.fit_scores``), and the paper's names."""
    from .weighted_r2 import fit_scores
    out = table(list(FOLDS))
    keys = list(zip(out['screen'], out['fold'], out['model'], out['seed'], out['level']))
    for column in ('weighted_r2', 'weighted_r2_contrast'):
        out[column] = [fit_scores()[k][column] for k in keys]
    return named(out)
