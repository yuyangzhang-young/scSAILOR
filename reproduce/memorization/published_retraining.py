"""Retraining of the published models (State, CPA, PrePR-CT) with replaced training targets: the target interventions,
the two-phase retraining and the planted tasks of Results 7.

Each run is a natural-task run of ``reproduce.published_models`` (model, optimizer, loss, schedule, fixed training
length, input sets and the pairing of each unit with 256 of its treated cells unchanged) whose treated cells are
changed after they are drawn, per unit of the update (``Retargeted``). The input set of a unit is its 256 carrier
atoms at the run's level (the built models' stream gives each atom twice: 512 carrier rows). Designs (``change``):

* ``original``: no change (the same-machine refit of the original recipe);
* ``input-relative``: every treated cell minus (the context's control-pool mean - the mean of the unit's input set),
  so that the target is relative to the input set instead of the whole pool;
* ``readable-only``: every treated cell minus the unreadable part of its context's offset (``retraining.unreadable``:
  the offset minus the readers' offset, the readers fitted on the other training lines, as in ``storage``);
* ``two-phase``: the original targets up to half the fixed length (``switch``); then every treated cell minus
  (the context's level - the mean of the unit's input set), the level being the pool mean plus the context's offset
  (``context_levels``), so that the second phase's targets carry no context-level correction beyond the input-set
  mean and the drug average.

A planted task (``--task``: ``c``, ``f`` or ``fnull`` of ``planted.task``, Tahoe-100M fold 1) adds each unit's change
of target to every one of its treated cells (``task_shift``), unclipped; with ``two-phase`` the context levels are those
of the task's targets, with the fold's drug average ``data.r`` (the one the model's anchor uses), not one recomputed
from the task's targets as ``retraining`` does for the built models. Values are computed in double precision and the
cells kept in single precision.

    python -m reproduce.memorization.published_retraining --model state --design input-relative \\
        --fold DATA/tahoe-100m/F1 --level L2 --seed 31002 --output runs/state/input-relative/F1/L2-31002

Run from the repository root; the outputs are those of ``reproduce.published_models.run`` (``PREDICTIONS_step<N>.npz``
on the training and held-out units). ``readable-only`` (the readers of ``--tahoe-fold``) and ``--task`` also read the
memorization data root (``SCSAILOR_DATA``).
"""
from ..published_models.common import runtime  # first: sets the thread variables before NumPy and PyTorch load

import argparse  # noqa: E402
import importlib  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from ..published_models.common.fit import fit  # noqa: E402
from ..published_models.common.natural import CELLS, PairedStream, TrainingSet, predict  # noqa: E402
from .retraining import context_levels, gap, pool_means, unreadable  # noqa: E402

__all__ = ["MODELS", "DESIGNS", "Retargeted", "shifted", "task_shift", "unreadable_shift", "input_relative",
           "two_phase", "change", "train"]

MODELS = {'state': 'state', 'cpa': 'cpa', 'prepr': 'prepr_ct'}   # model code -> ``reproduce.published_models`` package
DESIGNS = ('original', 'input-relative', 'readable-only', 'two-phase')


class Retargeted:
    """A training stream whose treated cells are changed per update: ``change(step, rows, atoms, target)`` receives the
    update's units (``rows``), their input sets (units x cells x genes) and treated cells (units x cells x genes) and
    returns the new treated cells. The stream state is the wrapped stream's."""

    def __init__(self, stream, change, cells=CELLS):
        self.stream, self.change, self.cells = stream, change, cells

    def batches(self, first, stop, workers):
        for step, batch in self.stream.batches(first, stop, workers):
            rows = self.stream.data.selected_rows(step, self.stream.seed)
            target = batch['target'].reshape(len(rows), self.cells, -1)
            batch['target'] = self.change(step, rows, batch['atoms'], target).reshape(batch['target'].shape)
            yield step, batch

    def state_dict(self):
        return self.stream.state_dict()

    def load_state_dict(self, state):
        self.stream.load_state_dict(state)


def shifted(shift):
    """Every treated cell of a unit plus the unit's ``shift`` (units x genes, single precision)."""
    return lambda step, rows, atoms, target: target + shift[rows][:, None]


def task_shift(natural, task):
    """A planted task's change of each unit's target: the task's training targets minus the natural ones."""
    return (task.astype(np.float64) - natural.astype(np.float64)).astype(np.float32)


def unreadable_shift(contexts, E, unit_contexts):
    """Per unit (its context in ``unit_contexts``), minus the unreadable part ``E`` of its context (rows
    ``contexts``)."""
    row = {c: i for i, c in enumerate(contexts)}
    return (-E[[row[c] for c in unit_contexts]]).astype(np.float32)


def input_relative(data):
    """Every treated cell minus (its context's pool mean - the mean of its unit's input set)."""
    pool = pool_means(data.pools)

    def change(step, rows, atoms, target):
        return (target - gap(pool, data.parent[rows], atoms)[:, None]).astype(np.float32)

    return change


def two_phase(data, switch, shift=None):
    """The targets (plus ``shift``) up to update ``switch``; after it, every treated cell minus (its context's level
    - the mean of its unit's input set), the levels (``retraining.context_levels``) of the targets plus ``shift`` with
    the drug average ``data.r``."""
    level = context_levels(data, data.r, None if shift is None else data.target.astype(np.float64) + shift)

    def change(step, rows, atoms, target):
        if shift is not None:
            target = target + shift[rows][:, None]
        if step <= switch:
            return target
        return (target - gap(level, data.parent[rows], atoms)[:, None]).astype(np.float32)

    return change


def change(design, data, steps, shift=None, tahoe_fold=None):
    """The target change of a design (``DESIGNS``) for a run of ``steps`` updates; None for the original targets."""
    if shift is not None and design not in ('original', 'two-phase'):
        raise ValueError(f'a planted task is trained with the designs original and two-phase, not {design}')
    if design == 'original':
        return None if shift is None else shifted(shift)
    if design == 'input-relative':
        return input_relative(data)
    if design == 'readable-only':
        contexts, E = unreadable(tahoe_fold)
        return shifted(unreadable_shift(contexts, E, data.contexts[data.parent]))
    return two_phase(data, steps // 2, shift)


def train(model, fold, level, seed, output, design='original', task='natural', tahoe_fold=None, device='cuda:0',
          workers=4):
    """Train one run (``model``: a code of ``MODELS``) for the model's fixed length and predict both sides
    (``reproduce.published_models.run``)."""
    module = importlib.import_module(f'reproduce.published_models.{MODELS[model]}.adapter')
    data = TrainingSet(fold, features=Path(fold) / 'drug_features.npz' if model == 'prepr' else None)
    if model == 'prepr':
        adapter = module.Adapter('natural', seed, data.r, data.features, device, feature_row=data.feature_row,
                                 r_row=data.r_row)
    else:
        adapter = module.Adapter('natural', seed, data.r, device)
    shift = None
    if task != 'natural':
        from . import planted
        # every drug of fold 1 has PrePR-CT descriptors, so ``data`` keeps all units, in the order of ``planted.task``
        shift = task_shift(data.target, planted.task(task)['target'])
    stream = PairedStream(data, level, seed)
    replaced = change(design, data, adapter.steps, shift, tahoe_fold)
    if replaced is not None:
        stream = Retargeted(stream, replaced)
    model_ = fit(adapter, stream, steps=adapter.steps, output=output, device=device, workers=workers)
    predict(adapter, model_.eval(), data, level=level, step=adapter.steps, output=output)


def main():
    parser = argparse.ArgumentParser(prog='python -m reproduce.memorization.published_retraining',
                                     description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--design', choices=DESIGNS, default='original')
    parser.add_argument('--fold', type=Path, required=True, help='natural-task fold directory (docs/data_contract.md)')
    parser.add_argument('--level', choices=('L1', 'L2'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=('natural', 'c', 'f', 'fnull'), default='natural',
                        help='the natural task or a planted task of Tahoe-100M fold 1 (planted.task)')
    parser.add_argument('--tahoe-fold', choices=('F1', 'F2', 'F3'), help='readable-only: the fold of the readers')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4, help='threads that prepare batches ahead')
    args = parser.parse_args()
    if args.design == 'readable-only' and args.tahoe_fold is None:
        parser.error('--tahoe-fold is required for readable-only')
    runtime.configure(args.device)
    train(args.model, args.fold, args.level, args.seed, args.output, args.design, args.task, args.tahoe_fold,
          args.device, args.workers)


if __name__ == '__main__':
    main()
