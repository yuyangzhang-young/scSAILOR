"""Retraining of the built models (summary-statistics, DeepSets and attention models) with replaced training targets:
the target interventions, the two-phase retraining and the planted tasks of Results 7.

Every run is the built models' natural-task recipe at Mean inputs, unchanged: the natural task's training stream
(``scsailor.natural.data``: eight units per update, each with 256 cells of its context's control pool and the carrier's
technical draw, all drawn from keys of the seed and the update), the 160,000-update schedule read out at ``STEPS`` =
40,000 updates (1,000 warm-up updates, cosine to a tenth; peak ``PEAKS``), AdamW (0.9, 0.999, eps 1e-8, no weight
decay), the gradient norm clipped at 1, the mean squared error, single precision without TF32
(``scsailor.models.set_runtime``). The model's baseline constants (the drug average ``r_input``, the training controls'
mean ``c_train`` and the intercept) are those of the run's targets (``constants``).

A design changes each drawn unit's target after it is drawn (``Stream``; values in double precision, the target kept in
single precision):

* ``original``: no change;
* ``input-relative``: target - (the context's control-pool mean - the mean of the unit's carrier rows), so that the
  target is relative to the input set instead of the whole pool;
* ``readable-only`` (Tahoe-100M): target - the unreadable part of the context's offset (``unreadable``);
* ``two-phase`` (Tahoe-100M): the original targets up to update ``SWITCH``; then target - (the context's level - the
  mean of the unit's carrier rows), the level being the pool mean plus the context's offset, the weighted mean over its
  units of target - pool mean - ``r_input`` (``context_levels``).

A planted task (``planted``) replaces the training targets before the run (Tahoe-100M fold 1: tasks ``c``, ``f`` and
``fnull`` with ``original`` and ``two-phase``; Parse PBMC groups 1 and 3: task ``carrier`` with ``original``).

After training, ``predict`` writes the prediction grid of every context of both sides and every perturbation from the
fold's evaluation inputs (one draw of the carrier at Mean, keys [41001, context index, draw, 777]): ``side_context``,
``p_code``, ``raw`` and ``anchor_raw`` (unclipped), the layout ``scsailor.resolution.grids.load`` reads (``fits``).

    python -m reproduce.memorization.retraining --dataset tahoe --fold F1 --design input-relative --task natural \\
        --model summary --seed 31002 --out runs/input-relative/summary-31002

``--updates N`` stops after N updates with the schedule unchanged and writes the weights and ``RUN.json`` (no
predictions). A full run writes ``step40000.pt``, ``predictions.npz`` and ``RUN.json`` (the arguments, the level, the
peak learning rate and the last update's loss).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scsailor.natural.data import NaturalData

from ..origin_resolution import parse as binding
from ..origin_resolution import tahoe

__all__ = ["LEVEL", "STEPS", "SWITCH", "PEAKS", "DESIGNS", "TASKS", "model_constants", "pool_means", "gap", "Stream",
           "input_relative", "readable_only", "context_levels", "two_phase", "unreadable", "constants", "setup",
           "build", "train", "predict", "main"]

LEVEL = 'L1'   # Mean inputs
STEPS = 40000
SWITCH = 20000   # two-phase: the last update with the original targets
PEAKS = {'summary': 1e-4, 'deepsets': 1e-4, 'attention_model': 3e-5}
DESIGNS = {'tahoe': ('original', 'input-relative', 'readable-only', 'two-phase'),
           'parse': ('original', 'input-relative')}
TASKS = {'tahoe': ('natural', 'c', 'f', 'fnull'), 'parse': ('natural', 'carrier')}


def pool_means(pools):
    """Each training context's control-pool mean (context code -> genes), double precision."""
    return {int(c): x.mean(axis=0, dtype=np.float64) for c, (x, _) in pools.items()}


def gap(level, parent, x):
    """Per unit (context codes ``parent``), its context's ``level`` (context code -> genes) minus the mean of its rows
    ``x`` (units x rows x genes), double precision."""
    return np.stack([level[int(c)] for c in parent]) - x.mean(axis=1, dtype=np.float64)


def model_constants(target, parent, p_code, weight, pool, drugs):
    """The baseline constants of a model trained on ``target`` (units x genes; unit context codes ``parent``, drug
    codes ``p_code``, weights ``weight``; ``pool``: context code -> control-pool mean): per drug the weighted mean of
    target - pool mean (``r_input``), the weighted mean of the units' pool means (``c_train``) and of the targets (the
    intercept), each in single precision."""
    t = target.astype(np.float64)
    pool_unit = np.stack([pool[int(p)] for p in parent])
    change = t - pool_unit
    r = np.stack([np.average(change[p_code == p], axis=0, weights=weight[p_code == p]) for p in range(drugs)])
    return (r.astype(np.float32), np.average(pool_unit, axis=0, weights=weight).astype(np.float32),
            np.average(t, axis=0, weights=weight).astype(np.float32))


class Stream(NaturalData):
    """The natural task's training stream of a training directory (``controls.h5``, ``train_targets.npz``), with the
    targets replaced by ``target`` and each drawn unit's target changed by ``change(step, parent, x, y)`` (the update,
    the units' context codes, their carrier rows and their targets)."""

    def __init__(self, directory, target=None, change=None):
        super().__init__(directory)
        if target is not None:
            self.target = target
        self.change = change

    def rows(self, step, seed):
        """The units drawn at an update (the draw of ``NaturalData.batch``)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, 0]))
        return rng.choice(len(self.target), size=8, replace=True, p=self.prob)

    def batch(self, step, m, level, seed):
        x, depth, p, y = super().batch(step, m, level, seed)
        if self.change is None:
            return x, depth, p, y
        return x, depth, p, self.change(step, self.parent[self.rows(step, seed)], x, y)


def input_relative(stream):
    """Target - (the context's pool mean - the mean of the unit's carrier rows)."""
    pool = pool_means(stream.pools)
    return lambda step, parent, x, y: (y - gap(pool, parent, x)).astype(np.float32)


def readable_only(E):
    """Target - ``E`` of the unit's context (context code -> genes)."""
    return lambda step, parent, x, y: (y - np.stack([E[int(c)] for c in parent])).astype(np.float32)


def context_levels(data, r_input, target=None):
    """Per training context (code) of ``data`` (a ``NaturalData``): its pool mean plus its offset, the weighted mean
    over its units of target - pool mean - ``r_input`` of the unit's drug; ``target`` (default ``data.target``),
    double precision."""
    target = data.target if target is None else target
    pool = pool_means(data.pools)
    rest = (np.asarray(target, np.float64) - np.stack([pool[int(c)] for c in data.parent])
            - r_input.astype(np.float64)[data.pcode])
    return {int(c): pool[int(c)] + np.average(rest[data.parent == c], axis=0, weights=data.prob[data.parent == c])
            for c in np.unique(data.parent)}


def two_phase(stream, r_input, switch=SWITCH):
    """The targets up to update ``switch``; after it, target - (the context's level - the mean of the unit's carrier
    rows)."""
    level = context_levels(stream, r_input)

    def change(step, parent, x, y):
        if step <= switch:
            return y
        return (y - gap(level, parent, x)).astype(np.float32)

    return change


def unreadable(fold):
    """The training contexts of a Tahoe-100M fold (sorted) and the unreadable part of their offsets on every gene: the
    offset minus the readers' offset of the context fitted on the other training lines (``storage``)."""
    from . import data, programs, readers, storage

    d = data.fold_data(fold)
    tr = d['train']
    _, G = programs.growth(fold)
    kernel_contexts, X, _ = readers.features(fold)
    contexts, offset, _ = storage.context_levels(tr['u'], tr['C'], tr['b'], d['r'])
    means = data.input_means(d['bags']['train'], contexts)
    return contexts, offset - storage.readable_training(offset, contexts, means, X @ X.T, kernel_contexts, G)


def constants(dataset, stream, stored, task=None):
    """``r_input``, ``c_train`` and the intercept of a run: on Tahoe-100M recomputed from the run's targets
    (``model_constants``); on Parse PBMC the group's stored constants (``stored``: ``training_constants.npz``; the
    intercept of ``train_targets.npz``), with ``r_input`` and the intercept of the planted carrier ``task``."""
    if dataset == 'tahoe':
        return model_constants(stream.target, stream.parent, stream.pcode, np.asarray(stream.prob, np.float64),
                               pool_means(stream.pools), stream.p_count)
    with np.load(stored) as z:
        r_input, c_train = z['r_input'], z['c_train']
    if task is not None:
        return task['r_input'], c_train, task['intercept']
    return r_input, c_train, stream.intercept


def setup(dataset, fold, design, task='natural'):
    """The stream of a run (``Stream``), its model constants (``constants``) and the evaluation inputs of its
    predictions. ``fold``: a Tahoe-100M fold or a Parse PBMC donor group."""
    if design not in DESIGNS[dataset] or task not in TASKS[dataset]:
        raise ValueError(f'{dataset} has no {design} run on task {task}')
    if task != 'natural' and (design not in ('original', 'two-phase') or fold not in ('F1', 'G1', 'G3')):
        raise ValueError(f'task {task} is trained with the original or two-phase design, on Tahoe-100M fold 1 or Parse '
                         'PBMC groups 1 and 3')
    from . import planted

    carrier = planted.carrier(fold) if task == 'carrier' else None
    if dataset == 'tahoe':
        files = tahoe.files(fold)
        directory, stored, inputs = files['controls'].parent, files['constants'], files['inputs']
        replaced = None if task == 'natural' else planted.task(task)['target']
    else:
        files = binding.files(fold)
        directory, stored, inputs = binding.ARRAYS / f'parse_outer_{fold}', files['constants'], files['inputs']
        replaced = None if carrier is None else carrier['target']
    stream = Stream(directory, replaced)
    values = constants(dataset, stream, stored, carrier)
    if design == 'input-relative':
        stream.change = input_relative(stream)
    elif design == 'readable-only':
        contexts, E = unreadable(fold)
        with np.load(stored) as z:
            code = {k: i for i, k in enumerate(z['context_keys'].astype(str))}
        stream.change = readable_only({code[c]: e for c, e in zip(contexts, E)})
    elif design == 'two-phase':
        stream.change = two_phase(stream, values[0])
    return stream, values, inputs


def build(name, values, drugs, seed, device='cpu'):
    """A built model (``scsailor.models.RawModel``) with the constants ``values`` (r_input, c_train, intercept),
    initialized after ``set_runtime(seed)``."""
    from scsailor.models import RawModel, set_runtime

    r_input, c_train, intercept = values
    set_runtime(seed)
    return RawModel(name, intercept, drugs, r_input, c_train).to(device).train()


def train(model, stream, name, seed, updates=STEPS, device='cpu', workers=3):
    """``updates`` updates of the recipe; returns each update's loss."""
    import torch
    from torch.nn import functional as F

    from scsailor.natural.data import lr_at, ordered_batches

    peak = PEAKS[name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak, betas=(.9, .999), eps=1e-8, weight_decay=0, fused=False)
    losses = []
    for step, batch, _ in ordered_batches(stream, 1, updates, 256, LEVEL, seed, workers):
        x, depth, p, y = (torch.from_numpy(v).to(device) for v in batch)
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, peak)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x, depth, p, LEVEL), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'nonfinite loss at update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def predict(model, inputs, device='cpu'):
    """The prediction grid over every context of both sides of the evaluation inputs (``inputs``: an
    ``evaluation_inputs.h5``) and every perturbation: 'side_context', 'p_code', 'raw' and 'anchor_raw'."""
    import h5py
    import torch

    from scsailor.carrier import make

    count = model.backbone.pert.num_embeddings
    model.eval()
    labels, raw, anchor = [], [], []
    with h5py.File(inputs, 'r') as bank, torch.inference_mode():
        for side in ('train', 'heldout'):
            for ci, context in enumerate(sorted(bank[side])):
                atoms, depth = bank[side][context]['x'][:], bank[side][context]['depth'][:]
                x, d = make(atoms, depth, LEVEL, np.random.default_rng(np.random.SeedSequence([41001, ci, 0, 777])))
                xt, dt = torch.from_numpy(x[None]).to(device), torch.from_numpy(d[None]).to(device)
                values, anchors = [], []
                for start in range(0, count, 8):
                    p = torch.arange(start, min(start + 8, count), device=device)
                    xx, dd = xt.expand(len(p), -1, -1), dt.expand(len(p), -1)
                    values.append(model(xx, dd, p, LEVEL).cpu().numpy())
                    anchors.append(model.anchor(xx, p, LEVEL).cpu().numpy())
                raw.append(np.concatenate(values).astype(np.float64))
                anchor.append(np.concatenate(anchors).astype(np.float64))
                labels += [f'{side}/{context}'] * count
    grid = {'side_context': np.asarray(labels), 'p_code': np.tile(np.arange(count), len(raw)),
            'raw': np.concatenate(raw), 'anchor_raw': np.concatenate(anchor)}
    if not (np.isfinite(grid['raw']).all() and np.isfinite(grid['anchor_raw']).all()):
        raise FloatingPointError('nonfinite prediction or anchor')
    return grid


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m reproduce.memorization.retraining',
                                     description=__doc__.split('\n\n')[0])
    parser.add_argument('--dataset', choices=tuple(DESIGNS), required=True)
    parser.add_argument('--fold', required=True, help='Tahoe-100M fold (F1-F3) or Parse PBMC donor group (G1, G3)')
    parser.add_argument('--design', choices=DESIGNS['tahoe'], default='original')
    parser.add_argument('--task', choices=('natural', 'c', 'f', 'fnull', 'carrier'), default='natural')
    parser.add_argument('--model', choices=tuple(PEAKS), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=STEPS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=3, help='threads that prepare batches ahead')
    args = parser.parse_args(argv)
    import torch

    stream, values, inputs = setup(args.dataset, args.fold, args.design, args.task)
    model = build(args.model, values, stream.p_count, args.seed, args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    losses = train(model, stream, args.model, args.seed, args.updates, args.device, args.workers)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.out / f'step{args.updates}.pt')
    record = {k: str(v) for k, v in vars(args).items()} | {'level': LEVEL, 'peak_lr': PEAKS[args.model],
                                                          'final_loss': losses[-1]}
    if args.updates == STEPS:
        with (args.out / 'predictions.npz').open('xb') as handle:
            np.savez(handle, **predict(model, inputs, args.device))
    (args.out / 'RUN.json').write_text(json.dumps(record, indent=1) + '\n')


if __name__ == '__main__':
    main()
