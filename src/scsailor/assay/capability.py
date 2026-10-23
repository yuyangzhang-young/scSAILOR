"""Capability reading: from per-fit flip-pair scores to one table of what each model reads.

``seed_rows``          one row per model x level x seed x support x bit, from the fits' scores;
``pool_seeds``         the three-seed mean, minimum and maximum of Q (all seeds required), and the
                       null bits' leakage ratio;
``leakage_ratio``      a null bit's predicted contrast energy A over the mean true energy E of the same
                       statistic's two main bits;
``null_leakage_table`` that ratio per seed;
``judge``              Q against an attainable reference: the relative recovery Q / reference, and
                       descriptive labels whose thresholds were fixed before the scores were seen, kept
                       as an extra column;
``summarize``          per model and support, how many visible cells reached the reference or show a
                       model problem under those labels.

The continuous read-outs are Q (``recovery`` in code), the reference and their ratio. The labels use
thresholds frozen before the scores were seen and are not part of the reading. Rows keep the saved
column names (``design.BIT``, ``STATISTIC``, ``ROLE``; ``REFERENCE_COLUMN``).
"""
from __future__ import annotations

import numpy as np

from .design import BIT, LEVELS, ROLE, STATISTIC, bits, visible
from .instance import HELDOUT, SEEN

SUPPORTS = (("heldout", HELDOUT), ("seen", SEEN))
METRICS = ("E", "M", "A", "K", "Q", "kappa")
MAIN = ("main_common", "main_conditional")
# Saved column of the attainable reference's Q: the second of two attainable references, a ridge reader on
# the bag mean (level L1, Mean) with eight principal components.
REFERENCE_COLUMN = "R2_Q"
# Label thresholds, fixed before the scores were seen: reference >= .8 makes a cell judgeable.
FROZEN_JUDGEABLE, FROZEN_PROBLEM_Q, FROZEN_REACHED_SHARE, FROZEN_TASK_LIMITED = .8, .2, .8, .5
RATIO_MIN_REFERENCE = .5  # the relative recovery is shown only where the reference reaches .5

__all__ = ["seed_rows", "pool_seeds", "leakage_ratio", "null_leakage_table", "frozen_label", "judge", "summarize"]


def seed_rows(scores, models, seeds, levels=LEVELS):
    """``scores[(model, level, seed)] = (SCORES.json content, its path)``; a missing fit or a side
    without flips gives rows with status "missing"; a bit with no true contrast (E = 0: the null bits, and
    the mean bits when ``T_mean`` is False) gives status "NA"."""
    rows = []
    for model in models:
        for level in levels:
            for seed in seeds:
                found = scores.get((model, level, seed))
                for support, side in SUPPORTS:
                    flips = found[0].get(side, {}).get("modules", []) if found else []
                    for info in bits():
                        row = {"model": model, "level": level, "seed": seed, "support": support,
                               BIT: info[BIT], STATISTIC: info[STATISTIC], ROLE: info[ROLE],
                               "first_visible_level": info["first_visible_level"],
                               "source_path": found[1] if found else None}
                        if flips:
                            scored = flips[info[BIT]]
                            row.update({k: scored[k] for k in METRICS}, status="ok" if scored["E"] > 0 else "NA")
                        else:
                            row.update(dict.fromkeys(METRICS), status="missing")
                        rows.append(row)
    return rows


def _main_energies(rows):
    """E of each statistic's main_common and main_conditional bits, per fit and support."""
    energies = {}
    for row in rows:
        if row[ROLE] in MAIN:
            energies[(row["model"], row["level"], row["seed"], row["support"], row[STATISTIC], row[ROLE])] = row["E"]
    return energies


def leakage_ratio(A_null, E_main_common, E_main_conditional):
    """A null bit's predicted contrast energy over the mean true energy of the same statistic's two main bits."""
    return A_null / ((E_main_common + E_main_conditional) / 2)


def pool_seeds(rows):
    """One row per model x level x support x bit. A bit with no true contrast in any seed (the null bits, and
    the mean bits when ``T_mean`` is False) is "NA" without Q statistics; a null bit's leakage ratio is left
    undefined when its statistic's main bits have no true contrast either."""
    groups = {}
    for row in rows:
        groups.setdefault((row["model"], row["level"], row["support"], row[BIT]), []).append(row)
    energies = _main_energies(rows)
    pooled = []
    for (model, level, support, bit), group in groups.items():
        first = group[0]
        complete = all(row["status"] != "missing" for row in group)
        null = first[ROLE] == "null"
        no_contrast = null or all(row["status"] == "NA" for row in group)
        out = {k: first[k] for k in ("model", "level", BIT, STATISTIC, ROLE, "first_visible_level", "support")}
        out.update(n_seeds=sum(row["Q"] is not None for row in group),
                   n_scored=sum(row["status"] != "missing" for row in group),
                   missing_seeds=";".join(str(row["seed"]) for row in group if row["status"] == "missing"),
                   status="ok" if complete and not no_contrast else ("NA" if complete else "missing"),
                   source_path=";".join(str(row["source_path"]) for row in group),
                   Q_mean=None, Q_min=None, Q_max=None, direction_consistent=None, deviation=None,
                   null_ratio_mean=None, null_ratio_min=None, null_ratio_max=None)
        if complete and not no_contrast:
            recovery = np.array([row["Q"] for row in group], dtype=float)
            out.update(Q_mean=float(recovery.mean()), Q_min=float(recovery.min()), Q_max=float(recovery.max()),
                       direction_consistent=bool(np.all(np.sign(recovery) == np.sign(recovery[0]))))
            out["deviation"] = 1 - out["Q_mean"] if visible(bit, level) else abs(out["Q_mean"])
        if complete and null and all(energies[_key(row, MAIN[0])] + energies[_key(row, MAIN[1])] > 0 for row in group):
            ratios = [leakage_ratio(row["A"], energies[_key(row, MAIN[0])], energies[_key(row, MAIN[1])])
                      for row in group]
            out.update(null_ratio_mean=float(np.mean(ratios)), null_ratio_min=float(np.min(ratios)),
                       null_ratio_max=float(np.max(ratios)))
        pooled.append(out)
    return pooled


def _key(row, role):
    return row["model"], row["level"], row["seed"], row["support"], row[STATISTIC], role


def null_leakage_table(rows):
    """Per seed, each null bit's predicted contrast energy relative to its statistic's main bits."""
    energies = _main_energies(rows)
    out = []
    for row in rows:
        if row[ROLE] != "null":
            continue
        common, conditional = energies[_key(row, MAIN[0])], energies[_key(row, MAIN[1])]
        scored = None not in (row["A"], common, conditional)
        defined = scored and common + conditional > 0   # no ratio without main-bit contrast (T_mean False)
        out.append(row | {"E_main_common": common, "E_main_conditional": conditional,
                          "E_main": (common + conditional) / 2 if scored else None,
                          "A_over_E_main": leakage_ratio(row["A"], common, conditional) if defined else None})
    return out


def frozen_label(bit, level, status, recovery, reference):
    """The descriptive label of one pooled cell (an extra column; not a reading of its own)."""
    if bit % 5 == 4:
        return "null_na"
    if status != "ok":
        return "missing"
    if not visible(bit, level):
        return "invisible"
    if reference >= FROZEN_JUDGEABLE and recovery < FROZEN_PROBLEM_Q:
        return "model_problem"
    if reference >= FROZEN_JUDGEABLE and recovery >= FROZEN_REACHED_SHARE * reference:
        return "reached_reference"
    if reference < FROZEN_TASK_LIMITED:
        return "task_limited_reference"
    return "unclassified"


def judge(pooled, references):
    """Add the attainable reference ``references[(bit, support)]`` (its Q), the relative recovery
    Q / reference (shown only for visible scored cells with a reference of at least .5)
    and the descriptive label."""
    judged = []
    for row in pooled:
        bit, recovery, status = row[BIT], row["Q_mean"], row["status"]
        reference_Q = references.get((bit, row["support"]))
        if reference_Q is None and bit % 5 != 4 and status == "ok":
            raise ValueError(f"No attainable reference for bit {bit} on {row['support']}")
        shown = status == "ok" and visible(bit, row["level"]) and reference_Q >= RATIO_MIN_REFERENCE
        judged.append(row | {REFERENCE_COLUMN: reference_Q, "ratio": recovery / reference_Q if shown else None,
                             "classification": frozen_label(bit, row["level"], status, recovery, reference_Q)})
    return judged


def summarize(judged):
    """Per model and support: visible complete cells, and how many carry each descriptive label."""
    groups = {}
    for row in judged:
        groups.setdefault((row["model"], row["support"]), []).append(row)
    out = []
    for (model, support), group in groups.items():
        eligible = [row for row in group if row[BIT] % 5 != 4 and row["status"] == "ok" and row["n_seeds"] == 3
                    and visible(row[BIT], row["level"])]
        n = len(eligible)
        reached = sum(row["classification"] == "reached_reference" for row in eligible)
        problem = sum(row["classification"] == "model_problem" for row in eligible)
        out.append({"model": model, "support": support, "visible_complete_cells": n, "reached_cells": reached,
                    "model_problem_cells": problem, "reached_fraction": reached / n if n else None,
                    "model_problem_fraction": problem / n if n else None})
    return out
