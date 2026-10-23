"""The synthetic-answer design as tables: what each planted bit is and what each level shows.

Bit ``a`` (0-24) writes genes ``2a`` and ``2a + 1``. Its statistic is ``a // 5`` and its role is ``a % 5``.
The answer moves along response direction ``a % 5``, so the five bits that share a role (one per
statistic) share a direction: their flips change the answer by the same profile before clipping.
Recovery is still separate per bit, because a flip pair changes one bit only.

``ROLES`` is part of the construction: ``generator`` takes each bit's response weight and perturbation
dependence from it. The other tables describe the construction without being read by it; the tests check
that they agree with it (which genes a bit writes, from which level it is visible).
"""
from __future__ import annotations

from .constants import N_BITS

LEVELS = ("L0", "L1", "L2", "L3", "L3d", "L4")

# What the model's input keeps about the 256 control cells at each level (each adds to the previous one).
LEVEL_INFORMATION = {
    "L0": "cell depths only; expression is zero",
    "L1": "per-gene mean (every cell equals the bag mean)",
    "L2": "per-gene mean and variance (two- or three-point atoms rescaled to the bag variance)",
    "L3": "each gene's full marginal distribution (values shuffled across cells, gene by gene)",
    "L3d": "each gene's marginal distribution within depth bins (the L3 shuffle done inside each bin)",
    "L4": "the cells themselves, so joint expression across genes",
}

# Statistic a // 5: how the two genes of a bit carry it, and the first level whose input keeps it.
STATISTICS = (
    ("mean", "gene 2a is 3 + .5b and gene 2a+1 is 3 - .5b in every cell", "L1"),
    ("variance", "both genes are 3 +/- s in half the cells each, s = .5 + alpha*b and .5 - alpha*b", "L2"),
    ("skewness", "gene 2a takes 2 + .5*{1,1,1,3} (b = +1) or {0,2,2,2} (b = -1): same mean and variance", "L3"),
    ("depth_coupling", "gene 2a is 2.5 in the shallower half of cells and 3.5 in the deeper half, reversed for b = -1", "L3d"),
    ("gene_joint", "within each depth bin both genes take the same ramp, paired in the same (b = +1) or opposite order", "L4"),
)

# Role a % 5: response weight (times beta) and whether the response is multiplied by the perturbation code.
ROLES = (
    ("main_common", 1.0, False),
    ("main_conditional", 1.0, True),
    ("weak_common", 0.2, False),
    ("weak_conditional", 0.2, True),
    ("null", 0.0, False),
)

# Saved column names of a bit row, kept from the original module registry: the bit, its statistic, its role.
BIT, STATISTIC, ROLE = "module_id", "edge", "module_type"

__all__ = ["LEVELS", "LEVEL_INFORMATION", "STATISTICS", "ROLES", "bits", "visible"]


def bits():
    """One row per planted bit."""
    rows = []
    for a in range(N_BITS):
        statistic, written_as, first_visible = STATISTICS[a // 5]
        role, weight, perturbation_dependent = ROLES[a % 5]
        rows.append({BIT: a, "genes": (2 * a, 2 * a + 1), STATISTIC: statistic, ROLE: role,
                     "first_visible_level": first_visible, "response_weight": weight,
                     "perturbation_dependent": perturbation_dependent, "response_direction": a % 5, "written_as": written_as})
    return rows


def visible(bit, level):
    """Whether the input at ``level`` keeps the planted statistic of bit ``bit``."""
    return LEVELS.index(level) >= LEVELS.index(STATISTICS[bit // 5][2])
