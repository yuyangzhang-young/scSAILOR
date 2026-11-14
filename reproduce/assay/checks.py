"""The carrier check of the synthetic-answer assay (Supplementary Note S6.6), on a CPU from an instance: flipping a
planted state changes nothing in the model input but its cue.

For each evaluation input set (the fixed 256-cell bags of both sides), one base state, the same for every set
(``STATE_SEED``), has each planted state set to +1 and to -1, and both inputs are built at every level with one
shared technical draw (``DRAW_SEED``). Over all states and levels the check records the largest difference between
the paired inputs on the 1,950 output genes and on the cells' depths, the largest difference of the whole input at
levels below the cue's first level, and how many states are not read back (``decode_bit``) from either input at or
above that level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scsailor.assay.carrier import carrier, decode_bit
from scsailor.assay.constants import FIRST_SCORED_GENE, N_BITS
from scsailor.assay.design import LEVELS, visible
from scsailor.assay.instance import HELDOUT, SEEN

__all__ = ["STATE_SEED", "DRAW_SEED", "SIDES", "paired_inputs", "carrier_check"]

STATE_SEED = 20260926     # the check's base state
DRAW_SEED = 31001         # the technical draw both inputs of a pair share
SIDES = {HELDOUT: "held-out", SEEN: "training"}


def paired_inputs(x, depth, params):
    """The check on one input set (cells ``x`` with their ``depth``): ``largest_output_difference`` (on the 1,950 output
    genes), ``largest_depth_difference``, ``largest_difference_below_first_level`` (the whole input), ``pairs`` (states
    x levels), ``pairs_below_first_level`` and ``decoded_wrong`` (states not read back at or above the first level)."""
    state = np.random.default_rng(STATE_SEED).choice([-1, 1], N_BITS)
    out = {"largest_output_difference": 0., "largest_depth_difference": 0., "largest_difference_below_first_level": 0.,
           "pairs": N_BITS * len(LEVELS), "pairs_below_first_level": 0, "decoded_wrong": 0}
    for a in range(N_BITS):
        plus, minus = state.copy(), state.copy()
        plus[a], minus[a] = 1, -1
        for level in LEVELS:
            (xp, dp), (xm, dm) = (carrier(x, depth, s, level, np.random.default_rng(DRAW_SEED), params)
                                  for s in (plus, minus))
            difference = np.abs(xp.astype(np.float64) - xm)
            out["largest_output_difference"] = max(out["largest_output_difference"],
                                                   float(difference[:, FIRST_SCORED_GENE:].max()))
            out["largest_depth_difference"] = max(out["largest_depth_difference"], float(np.abs(dp - dm).max()))
            if visible(a, level):
                out["decoded_wrong"] += int(decode_bit(xp, dp, a) != 1) + int(decode_bit(xm, dm, a) != -1)
            else:
                out["pairs_below_first_level"] += 1
                out["largest_difference_below_first_level"] = max(out["largest_difference_below_first_level"],
                                                                  float(difference.max()))
    return out


def carrier_check(instance):
    """One row per evaluation input set of the instance (held-out, then training contexts)."""
    rows = []
    for side, name in SIDES.items():
        for bag in instance.evaluation_bags(side):
            rows.append({"side": name, "context": bag.context, **paired_inputs(bag.x, bag.depth, instance.params)})
    return pd.DataFrame(rows)
