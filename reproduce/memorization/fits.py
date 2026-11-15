"""The saved predictions of the retraining runs behind Results 7 (``retraining`` trains the built models,
``published_retraining`` State, CPA and PrePR-CT), under the data root.

Tahoe-100M (``tahoe-100m-retraining/<design>/<fold>/<task>/<model>_<seed>_<level>.npz``), every model:

- design ``original``: the original recipe trained again on the same machine as the other retraining runs: the
  same-machine refits on the natural task (the built models' seeds in ``REFITS``; State at Mean and +Var, CPA and
  PrePR-CT at Mean) and the fits on the planted tasks;
- designs ``input-relative`` and ``readable-only``: the target interventions (fold 1, natural task);
- design ``two-phase``: the two-phase retraining (folds 1-3 on the natural task; fold 1 on the planted tasks);
- task ``natural`` or a planted task (``c``, ``f``, ``fnull``; ``planted``);
- level ``L1`` (Mean inputs), or ``L2`` (+Var) for State.

The models' fits on the ladder are not retraining runs: they are read with ``data.ladder_file``.

Parse PBMC (``parse-pbmc-retraining/<group>/<design>/<task>/<model>_<seed>_L1.npz``): designs ``original`` and
``input-relative``, tasks ``natural`` and ``carrier``.

A file holds, per side it is read on (``train_`` the training units, ``heldout_`` the held-out units), the run's raw
prediction (``raw``) and, where it is read, its anchor (``anchor_raw``), unclipped, as rows in the order of all units of
the side; ``scsailor.resolution.grids.load`` reads them with those units.
"""
from ..origin_resolution import ROOT

__all__ = ["TAHOE", "PARSE", "REFITS", "tahoe", "parse"]

TAHOE = ROOT / 'tahoe-100m-retraining'
PARSE = ROOT / 'parse-pbmc-retraining'
# the built models' seeds with a same-machine refit of the original recipe on fold 1's natural task; the other seeds'
# original is the ladder's fit (``data.ladder_file``)
REFITS = {'summary': (31002,), 'deepsets': (31002, 31003, 31004), 'attention_model': (31002, 31003)}


def tahoe(design, model, seed, fold='F1', task='natural', level='L1'):
    """The file of one Tahoe-100M retraining run."""
    return TAHOE / design / fold / task / f'{model}_{seed}_{level}.npz'


def parse(group, design, model, seed, task='natural'):
    """The file of one Parse PBMC retraining run (Mean inputs)."""
    return PARSE / group / design / task / f'{model}_{seed}_L1.npz'
