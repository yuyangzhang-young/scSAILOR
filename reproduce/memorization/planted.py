"""The planted tasks of Results 7: copies of the training targets (and of the held-out truth) with one known term
added, on which the built models are retrained (``retraining``) and read back.

Tahoe-100M, fold 1 (``task``). Each task adds ``beta`` times a cue of the context to the treated means of the growth
genes (the three growth programs, ``programs``) and clips them at zero, as treated means are. The models' inputs, the
control cells, are unchanged.

* ``f`` (the readable shift): the cue is the context's proliferation score, the mean over the E2F-target and G2M
  genes of its control-pool mean, standardized over the training contexts; it is planted on the units of three drugs
  only (``gated_drugs``), so the planted term is drug-specific but readable from the untreated cells;
* ``c`` (the planted null): every drug, with the score permuted across lines (``line_permutation``; both plates of a
  line move together), separately among the training and among the held-out lines, so that no control statistic
  predicts it;
* ``fnull`` (the drug x line null): the drugs of ``f`` with the cue of ``c``.

``beta`` = beta_strong x sqrt(1,950 / genes), beta_strong being the first coefficient of the fold's known-answer
generator: one SD of the cue moves a unit by the norm of a strong bit of the synthetic-answer assay. ``r_change`` is the
change of the training drug average (what the anchor absorbs): per drug, the weighted mean over its training units of
the planted minus the real target.

``Known(name, side)`` reads a fit's change on a task's planted block (the task's units and genes) against the planted
change of the truth beyond ``r_change``.

Parse PBMC (``carrier``): in donor groups 1 and 3, the training targets of the carrier donors' contexts change on the
interferon genes (the union of the Hallmark interferon alpha and gamma programs) by ``sign`` x ``beta`` x the
context's interferon score (the control-pool mean on the average of the two programs, centred within cell type over
the training contexts and divided by the SD of those centred values). Group 1's carrier, Donor3, persists in the
interferon-high state and is planted to fade (sign -1); group 3's carriers, Donors 1, 4 and 10, fade and are planted to
persist (+1). beta is group 1's beta_strong x sqrt(1,950 / genes) in both groups. Of the held-out side only the context
names are read.
"""
import ast
from functools import cache

import h5py
import numpy as np

from ..origin_resolution import ROOT, tahoe
from ..origin_resolution import parse as binding
from . import parse_interferon, programs

__all__ = ["FOLD", "TASKS", "SCORE", "RECEPTOR", "TARGETS", "STEROIDS", "N_GATED", "PERMUTATION_SEED",
           "PERMUTATION_MAX_CORR", "SCORED_GENES", "CARRIERS", "plant", "drug_changes", "line_permutation",
           "gated_drugs", "task", "Known", "carrier_cue", "carrier"]

FOLD = 'F1'
TASKS = ('c', 'f', 'fnull')
SCORE = ('HALLMARK_E2F_TARGETS', 'HALLMARK_G2M_CHECKPOINT')   # the proliferation score's genes
RECEPTOR = 'NR3C1'   # the glucocorticoid receptor; ``gated_drugs`` picks the drugs whose response ignores it
TARGETS = ('FKBP5', 'TSC22D3', 'PER1', 'KLF9', 'DUSP1')   # the receptor's target genes
STEROIDS = ('Triamcinolone', 'Medroxyprogesterone acetate', 'Megestrol', 'Quinestrol')   # never gated drugs
N_GATED = 3
PERMUTATION_SEED = 20261109
PERMUTATION_MAX_CORR = .1
SCORED_GENES = 1950   # genes of the synthetic-answer assay's readout
GENERATORS = {'tahoe': ROOT / 'tahoe-100m/F1/synthetic/generator.npz',
              'parse': ROOT / 'parse-pbmc/G1/synthetic/generator.npz'}   # the known-answer generators of beta_strong
CARRIERS = {'G1': (('Donor3',), -1.0), 'G3': (('Donor1', 'Donor4', 'Donor10'), 1.0)}   # group -> carriers, sign


def beta_strong(dataset):
    """The first coefficient of a dataset's known-answer generator."""
    with np.load(GENERATORS[dataset]) as g:
        return float(g['beta'][0])


def plant(values, rows, genes, shift):
    """A single-precision copy of ``values`` with max(value + shift, 0) on ``rows`` x ``genes`` (``shift``
    broadcast to that block, double precision)."""
    out = values.copy()
    block = values[np.ix_(rows, genes)].astype(np.float64) + shift
    out[np.ix_(rows, genes)] = np.maximum(block, 0).astype(np.float32)
    return out


def drug_changes(change, p_code, weight, drugs):
    """Per drug code (``drugs`` of them), the weighted mean over its units of ``change`` (units x genes), single
    precision."""
    return np.stack([np.average(change[p_code == p], axis=0, weights=weight[p_code == p])
                     for p in range(drugs)]).astype(np.float32)


def line_permutation(contexts, z, rng):
    """``z`` (one value per context 'line|plate') with the lines permuted: a context takes the value of the context of
    another line on its own plate. The first permutation drawn from ``rng`` whose correlation with ``z`` is at most
    ``PERMUTATION_MAX_CORR`` in absolute value."""
    line, plate = np.array([c.split('|')[0] for c in contexts]), np.array([c.split('|')[1] for c in contexts])
    names = sorted(set(line))
    lookup = {(l, p): v for l, p, v in zip(line, plate, z)}
    for _ in range(1000):
        order = rng.permutation(len(names))
        mapping = {names[i]: names[j] for i, j in enumerate(order)}
        permuted = np.array([lookup[(mapping[l], p)] for l, p in zip(line, plate)])
        if abs(np.corrcoef(permuted, z)[0, 1]) <= PERMUTATION_MAX_CORR:
            return permuted
    raise ValueError('no permutation within the correlation bound')


def gated_drugs(response, parent, p_code, receptor_z, names, contexts):
    """The ``N_GATED`` drug codes, among the non-steroid drugs measured in all ``contexts`` training contexts, whose
    target response (``response``: per training unit) depends least on the receptor across contexts: the smallest
    absolute least-squares slope on the receptor's z (``receptor_z``, per context)."""
    rows = []
    for p, name in enumerate(names):
        m = p_code == p
        if m.sum() != contexts or name in STEROIDS:
            continue
        rows.append((abs(np.polyfit(receptor_z[parent[m]], response[m], 1)[0]), p))
    return [p for _, p in sorted(rows)[:N_GATED]]


@cache
def _tahoe():
    """Fold 1's inputs of the planted tasks: training and held-out units, control-pool means, cues, genes, drugs."""
    f = tahoe.files(FOLD)
    genes = programs.panel(FOLD)
    sets = programs.gene_sets(programs.HALLMARK.read_text())
    score_genes = np.isin(genes, sorted(set().union(*(sets[n] for n in SCORE))))
    program = np.flatnonzero(programs.growth(FOLD)[1])
    index = {g: i for i, g in enumerate(genes)}
    receptor, targets = index[RECEPTOR], np.array([index[g] for g in TARGETS])
    with np.load(f['constants']) as z:
        keys = z['context_keys'].astype(str)
    C = np.empty((len(keys), len(genes)))
    with h5py.File(f['controls'], 'r') as bank:
        for key in bank:
            C[int(key)] = bank[key]['input/x'][:].astype(np.float64).mean(axis=0)
    with np.load(f['train_targets']) as z:
        train = {k: z[k] for k in ('target', 'parent', 'p_code', 'weight', 'p_ids')}
    train['parent'], train['p_code'] = train['parent'].astype(int), train['p_code'].astype(int)
    names = [ast.literal_eval(p)[0][0] for p in train['p_ids'].astype(str)]
    with np.load(f['outer']) as z:
        held = {k: z[k] for k in ('truth', 'context', 'p_code')}
    held_keys = sorted(set(held['context'].astype(str)))
    with h5py.File(f['inputs'], 'r') as bank:
        C_held = np.stack([bank['heldout'][k]['C'][:] for k in held_keys])
    held['parent'] = np.searchsorted(held_keys, held['context'].astype(str))
    score, score_held = C[:, score_genes].mean(axis=1), C_held[:, score_genes].mean(axis=1)
    z_train, z_held = (score - score.mean()) / score.std(), (score_held - score.mean()) / score.std()
    receptor_z = (C[:, receptor] - C[:, receptor].mean()) / C[:, receptor].std()
    response = (train['target'][:, targets] - C[train['parent']][:, targets]).mean(axis=1)
    drugs = gated_drugs(response, train['parent'], train['p_code'], receptor_z, names, len(C))
    rng = np.random.default_rng(PERMUTATION_SEED)
    null_train = line_permutation(keys, z_train, rng)
    null_held = line_permutation(held_keys, z_held, rng)
    return {'train': train, 'held': held, 'keys': keys, 'held_keys': np.array(held_keys), 'genes': program,
            'beta': beta_strong('tahoe') * np.sqrt(SCORED_GENES / len(program)), 'drugs': drugs,
            'cues': {'score': (z_train, z_held), 'null': (null_train, null_held)}}


@cache
def task(name):
    """A planted task of fold 1 (``TASKS``): 'genes' (panel indices), 'beta', 'drugs' (codes; empty for every drug),
    'r_change' (drugs x genes), 'cue_train' and 'cue_heldout' (per context of 'train_contexts', 'heldout_contexts'),
    'target' (the planted training targets, in the order of the fold's training units) and 'truth' (the planted
    held-out truth, in the order of the fold's held-out units)."""
    t = _tahoe()
    train, held, genes, beta = t['train'], t['held'], t['genes'], t['beta']
    cue_train, cue_held = t['cues']['null' if name in ('c', 'fnull') else 'score']
    drugs = np.array(t['drugs'] if name in ('f', 'fnull') else [], dtype=int)
    rows_train = np.isin(train['p_code'], drugs) if len(drugs) else np.ones(len(train['p_code']), bool)
    rows_held = np.isin(held['p_code'], drugs) if len(drugs) else np.ones(len(held['p_code']), bool)
    shift_train = beta * cue_train[train['parent'][rows_train]][:, None] * np.ones(len(genes))
    shift_held = beta * cue_held[held['parent'][rows_held]][:, None] * np.ones(len(genes))
    target = plant(train['target'], np.flatnonzero(rows_train), genes, shift_train)
    truth = plant(held['truth'], np.flatnonzero(rows_held), genes, shift_held)
    change = np.zeros((len(train['parent']), len(genes)))
    cells = np.ix_(rows_train, genes)
    change[rows_train] = target[cells].astype(np.float64) - train['target'][cells]
    r_change = np.zeros((len(train['p_ids']), target.shape[1]), dtype=np.float32)
    r_change[:, genes] = drug_changes(change, train['p_code'], train['weight'], len(train['p_ids']))
    return {'genes': genes, 'beta': beta, 'drugs': drugs, 'r_change': r_change, 'cue_train': cue_train,
            'cue_heldout': cue_held, 'train_contexts': t['keys'], 'heldout_contexts': t['held_keys'],
            'target': target, 'truth': truth}


class Known:
    """A planted task's block on one side (``side``: 'train', the fold's training units, or 'heldout', its held-out
    units): ``units`` (the real units), ``rows`` (the planted units), ``genes``, ``beta``, ``change`` (``r_change``
    per unit), ``cue`` (per unit) and ``dt``, the planted truth minus the real truth minus ``change``."""

    def __init__(self, name, side):
        planted = task(name)
        self.genes, self.beta = planted['genes'], float(planted['beta'])
        self.units = tahoe.units(FOLD, 'train' if side == 'train' else 'outer')
        contexts = planted['train_contexts' if side == 'train' else 'heldout_contexts']
        cue = dict(zip(contexts, planted['cue_train' if side == 'train' else 'cue_heldout']))
        drugs = planted['drugs']
        self.rows = np.isin(self.units['p_code'], drugs) if len(drugs) else np.ones(len(self.units['p_code']), bool)
        self.change = planted['r_change'].astype(np.float64)[self.units['p_code']]
        truth = planted['target' if side == 'train' else 'truth'].astype(np.float64)
        self.dt = truth - self.units['truth'] - self.change
        self.cue = np.array([cue[c] for c in self.units['context']])

    def slope(self, y):
        """The weighted least-squares slope of ``y`` (per planted unit) on the cue, over beta."""
        w = self.units['weight'][self.rows]
        x0 = self.cue[self.rows] - np.average(self.cue[self.rows], weights=w)
        return float(np.sum(w * x0 * y) / np.sum(w * x0 * x0)) / self.beta

    def read(self, net):
        """``net`` (a fit's change beyond its anchor change, units x genes) on the planted block: its mean over the
        planted genes regressed on the cue, over beta ('slope_over_beta'), and the same of the planted truth
        ('truth_slope_over_beta')."""
        block = np.ix_(self.rows, self.genes)
        return {'slope_over_beta': self.slope(net[block].mean(axis=1)),
                'truth_slope_over_beta': self.slope(self.dt[block].mean(axis=1))}


def carrier_cue(score, contexts):
    """The carrier cue of each training context ('donor|cell type'): ``score`` centred within cell type, divided by
    the SD of the centred values."""
    types = np.array([c.split('|')[1] for c in contexts])
    centred = score - np.array([score[types == t].mean() for t in types])
    return centred / centred.std()


@cache
def carrier(group):
    """The planted carrier task of a Parse PBMC donor group (``CARRIERS``): 'genes', 'cue_train' (per context of
    'train_contexts'), 'heldout_contexts', 'carriers', 'sign', 'beta', 'r_change', 'target' (the planted training
    targets, in the order of the group's training units), and the model constants recomputed from them, 'r_input' and
    'intercept' (``retraining.model_constants``)."""
    from .retraining import model_constants

    carriers, sign = CARRIERS[group]
    f = binding.files(group)
    _, W = parse_interferon.group_programs(group)
    alpha, gamma = W[parse_interferon.ALPHA], W[parse_interferon.GAMMA]
    genes = np.flatnonzero((alpha > 0) | (gamma > 0))
    with np.load(f['constants']) as z:
        keys = z['context_keys'].astype(str)
    with h5py.File(f['inputs'], 'r') as bank:
        held = sorted(bank['heldout'])
        C = np.stack([bank['train'][k]['C'][:] for k in keys]).astype(np.float64)
    with np.load(f['train']) as z:
        train = {k: z[k] for k in z.files}
    parent, p_code, w = train['parent'].astype(int), train['p_code'].astype(int), train['weight'].astype(np.float64)
    z_tr = carrier_cue(C @ ((alpha + gamma) / 2), keys)
    beta = beta_strong('parse') * np.sqrt(SCORED_GENES / len(genes))
    is_carrier = np.array([c.split('|')[0] in carriers for c in keys])
    shift = sign * beta * z_tr * is_carrier
    target = plant(train['target'], np.arange(len(parent)), genes, shift[parent][:, None])
    with h5py.File(binding.ARRAYS / f'parse_outer_{group}/controls.h5', 'r') as bank:
        pool = {int(k): bank[k]['input/x'][:].mean(axis=0, dtype=np.float64) for k in bank}
    r_input, _, intercept = model_constants(target, parent, p_code, w, pool, len(train['p_ids']))
    r_change = drug_changes(target.astype(np.float64) - train['target'], p_code, w, len(train['p_ids']))
    return {'genes': genes, 'cue_train': z_tr, 'train_contexts': keys, 'heldout_contexts': np.array(held),
            'carriers': np.array(carriers), 'sign': sign, 'beta': beta, 'r_change': r_change, 'target': target,
            'r_input': r_input, 'intercept': intercept}
