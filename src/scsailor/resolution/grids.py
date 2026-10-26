"""Origin resolution's prediction grids: a model's predictions for every context and perturbation of a fold.

A grid file holds the raw prediction (``raw``, unclipped) and the model's anchor (``anchor_raw``: the input bag's
mean plus the model's transferred response, unclipped) per unit. Three layouts occur:

- origin resolution's grid: ``side_context`` (``"train/<context>"`` or ``"heldout/<context>"``), ``p_code`` and the
  members, over every context of both sides;
- a scorer's file with its own keys: ``<side>_context``, ``<side>_p_code`` and ``<side>_<member>`` (official State);
- a scorer's file already in the scoring units' order: ``<side>_<member>`` only (the built models' fits).

``side`` is ``"train"`` (training units) or ``"outer"`` (held-out units).
"""
import numpy as np

__all__ = ["load"]


def load(path, side, scoring, member='raw'):
    """One member of a grid file as rows in the order of the units ``scoring``."""
    prefix = 'heldout' if side == 'outer' else 'train'
    wanted = [(c, int(p)) for c, p in zip(scoring['context'], scoring['p_code'])]
    with np.load(path) as saved:
        if 'side_context' in saved.files:
            keys = zip(saved['side_context'].astype(str), saved['p_code'].astype(int))
            wanted = [(prefix + '/' + c, p) for c, p in wanted]
            values = saved[member]
        else:
            values = saved[f'{prefix}_{member}']
            if f'{prefix}_context' not in saved.files:
                if len(values) != len(wanted):
                    raise ValueError(f'{path}: {len(values)} {prefix} rows against {len(wanted)} scoring units')
                return np.asarray(values, dtype=np.float64)
            keys = zip(saved[f'{prefix}_context'].astype(str), saved[f'{prefix}_p_code'].astype(int))
        lookup = {key: i for i, key in enumerate(keys)}
        return np.asarray(values, dtype=np.float64)[[lookup[key] for key in wanted]]
