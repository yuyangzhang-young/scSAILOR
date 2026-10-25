"""Readings: what the finished runs of a map say about where the target can be learned.

A map run (``starts``) trains one starting point of a parent at one cut and one rate on the synthetic task's
targets, and writes ``RUN_RECORD.json`` (its row: model, half, seed, case, cut, arm, rate multiple),
``EVAL_PARENT.json`` (the parent, scored before any training), and either ``EVAL.json`` (the end) with
``COMPLETE.json``, or ``DIVERGED.json``. ``tables(runs, settings, target)`` reads the finished runs it is given and
returns the tables below, keyed by their file names; ``write(folder, tables)`` writes them. ``Settings`` holds what
the rules need to know about the study: the map's seeds, each model's junction cut and the planted faults.
Everything is pure numpy on the saved evaluations; nothing is rescored.

The arms at a cut: ``a`` fine-tunes the parent behind the cut (at the cut ``input``, the whole parent: its own
continued training); ``d`` resets the drug embedding read behind the cut; ``f`` resets the back (behind the cut),
``e`` the front (before it, then frozen); for the summary-statistics model, ``e_mean`` / ``e_var`` reset only the
mean or the variance projection, ``interaction`` / ``common`` train one branch alone, and ``interaction_e_mean``
(the interaction branch with the mean projection reset) is the reference arm; ``fresh`` trains a re-initialized
network (cut ``input``). A rate is a multiple of the parent's own rate at 40k; the grid is x1, x3, x10, x30.

* **Cell outcome on the held-out panel** (``MAP_CELLS``). The target's recovery Q at the end, with a line-bootstrap
  interval (2,000 resamples of held-out cell lines, both plates together; the same resamples for every run, so an
  interval does not depend on what else is read), and the margins against the parent (the one reference for every
  arm): in the synthetic half the learned bits (0, and 5 where the parent has it at or above 0.5) end no lower than
  0.1 below the parent; in both halves the raw error on the target's endpoints ends no higher than twice the
  parent's. Outcomes: recovered (the interval's lower bound reaches L = 0.25), recovered with collateral damage
  (recovered outside the margins), not recovered (the upper bound below L), undetermined, diverged. The parent's
  own starting reading, from its saved evaluation, is kept beside it, so a diverged run has it too. Values keep
  full precision; they are rounded only when a table is written.
* **Arm readings** (``MAP_ARMS``) separate where the target can be learned from what learning it costs. Site: recovers (recovered at one or more rates, with or without
  collateral damage); does not recover (the whole grid run, an evaluable x1, and every evaluable rate not
  recovered: learning happens in a window of rates, so a negative needs the whole grid); not recovered at the rates
  run (the same with the grid incomplete, a bounded reading that carries no conclusion); undetermined. Cost, for an
  arm that recovers: clean (recovered within the margins at some rate) or with collateral damage (only beyond
  them). Rate-gated: recovered only above x1, with an evaluable x1 that is not recovered; an undetermined x1 is
  reported as such. A diverged run covers its rate but counts neither way.
* **Contrasts** (``MAP_CONTRASTS``), per model, half, seed, case and cut, each changing one component against arm
  a: drug side (d), back (f), front (e) and, for summary-statistics, the mean and the variance projection (e_mean,
  e_var). Arm a is read over all its runs: a recovery at any rate means no reset is needed, and its negative, the
  premise of a reset benefit, needs the whole grid. The reset arm's recovery and its cost count only at matched
  rates, where both arms have an evaluable run, x1 among them; a recovery at another rate is listed beside the
  contrast and leaves it undetermined. Outcome: reset benefit, kept arm recovers, neither recovers, neither
  recovers at the rates run (the reset arm's grid incomplete), undetermined. A case whose parent is recovered at
  the start is solved: its contrasts are intervention effects, and a loss after training is training-induced
  damage, not a failure of the parent. The summary-statistics single branches are read as sufficient or not.
* **Case** (``MAP_CASES``): whether any parent-derived arm recovers (the reference arm is read beside the labels,
  never in them), undetermined when none recovers and one is undetermined, with the fresh network's reading
  (synthetic half, same model and seed) beside it. Planted-fault cases (a parent that recovers the target, as it
  is and with one planted fault each) are kept apart by their ``case``.
* **Carried cell** (``MAP_CARRIED``, ``MAP_CARRIED_LINES``) per model and half, the change handed on to the real
  task and to origin resolution: among the cells behind a cut (not the cut input, not the fresh network or the
  reference arm) that recover the target on the validation panel (validation recovery at least L) within its
  margins in every map seed, the one with the best mean validation recovery: a qualified repair. If no cell
  qualifies, the fixed intervention is handed on instead, whatever its runs show: arm a at the model's junction
  cut at x1 (fixed by the architecture, not chosen from results), recorded as "diagnostic
  intervention (qualified repair: none)". Held-out results never decide. Its held-out outcome per seed is mapped
  to origin resolution's routing categories (recovered; recovered, cost flagged; partial for undetermined; not
  recovered; diverged excluded), in its own half and in the other (the synthetic choice read on the natural
  parent is the bridge), and its per-line recovery in both halves is written for the line alignment.
* **Planted-fault check** (``MAP_KNOWN_ANSWER``): each planted case against its admissible answer: pass (every
  required contrast reads reset benefit and every excluded one neither recovers), wrong (an excluded contrast
  reads reset benefit), fault not binding (arm a recovers despite the fault), incomplete (a contrast the verdict
  needs is undetermined, bounded or not run while runs are still scheduled; for the solved case, an undetermined
  start) or miss.
* **Rates to add** (``MAP_WAVE2``), read on the validation panel: an arm that learns the target at none of the rates
  run (validation recovery below L) gets the grid rates it lacks where its negative carries a conclusion. That is
  arm a at a cut where another parent-derived arm learns (the premise of a reset benefit), and every arm at a cut
  where none learns (the cut's negative); in the planted-fault cases, every arm that does not learn. The fresh
  network and the reference arm do not count as arms that learn, and the reference arm gets no rates. Other arms
  keep the reading "not recovered at the rates run". Held-out results never schedule
  runs. One row per run to add.
* **Selection picks** (``MAP_PICKS``) on the menu (every map run of a parent case except the fresh network), read
  on the training-line validation panel and evaluated on the held-out panel: target-aware (IL), the best
  validation recovery among runs that meet the margins on the validation panel; Cross-Val-style, the lowest
  validation raw error on the planted task's endpoints; fixed block, arm a at the junction cut at x1; continued
  training, arm a at the cut input at x1; and the menu mean of the held-out values, the expected value of a random
  pick.

Seeds that run only the carried cells (``Settings.carried_only_seeds``) enter ``MAP_CELLS`` and ``MAP_CARRIED``, not
the map's case tables. Two runs of one map cell stop the reading.

**The target** is bit 1 (the strong drug-specific mean) unless ``target`` names another scored bit: 0 (common
mean), 5 (common variance) or 6 (drug-specific variance). Every map run scores these four bits, so any of them is
read from the saved runs with the same rules. The learned-bit margins then cover the other bits among 0, 1 and 5
(bit 0 always, the others where the parent has them at or above 0.5). The raw-error margin keeps the endpoints the
runs saved, those of the bit-1 flip pairs. Tables for another target carry the suffix ``_bit<target>``. A table
with no rows is written as an empty file, so an earlier version never survives.
"""
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = ["RECOVERY_LEVEL", "GRID", "Settings", "tables", "write", "cell", "interval_reading", "margins",
           "arm_reading", "cost", "rate_gated", "contrast", "contrasts", "cases", "carried", "carried_lines",
           "planted_verdict", "known_answer", "second_wave", "picks"]

TARGETS = ("0", "1", "5", "6")   # the scored bits every map run saves
RECOVERY_LEVEL = 0.25   # recovery level L
LEARNED_MARGIN = 0.1   # learned bits may end at most this far below the parent's value
ERROR_FACTOR = 2.0     # raw target-endpoint error may end at most this many times the parent's
RESAMPLES = 2000
BOOTSTRAP_SEED = 20261017
GRID = (1.0, 3.0, 10.0, 30.0)   # multiples of the parent's own rate
LEARNS = ("recovered", "recovered with collateral damage")   # the target is learned; the margins are the cost
CONTRASTS = {"drug side": "d", "back": "f", "front": "e", "mean projection": "e_mean", "variance projection": "e_var"}
BRANCHES = ("interaction", "common")
FRESH = "fresh"
REFERENCE_ARM = "interaction_e_mean"   # read beside the labels, never in them
OR_CATEGORY = {"recovered": "recovered", "recovered with collateral damage": "recovered, cost flagged",
               "undetermined": "partial", "not recovered": "not recovered", "diverged": "excluded"}
KEY = ("model", "half", "seed", "case", "cut", "arm", "rate")   # one run per map cell
# Decimals when a table is written; everything is computed at full precision.
DIGITS = {"target_Q": 4, "target_low": 4, "target_high": 4, "common_mean_change": 4, "common_variance_change": 4,
          "validation_target_Q": 4, "mean_validation_target_Q": 4, "heldout_target_Q": 4,
          "error_ratio": 3, "validation_error_ratio": 3, "heldout_error_ratio": 3}


@dataclass(frozen=True)
class Settings:
    """What the reading rules need to know about the study's runs.

    ``map_seeds``: the map's seeds; a cell is a qualified repair only in every one of them.
    ``junction``: per model, the junction cut, where the fixed intervention and the fixed-block pick train arm a
    (fixed by the architecture, not chosen from results).
    ``carried_only_seeds``: seeds that run only the carried cells.
    ``planted``: per planted-fault case, its cut, the contrasts that must read "reset benefit", those that must not,
    and the contrasts for which "neither recovers" with the whole network recovering is also admissible. That holds
    for a front fault: a reset front is random and frozen, so it carries the cue only if a random projection keeps it
    readable, while a reset back is trained and must learn.
    ``planted_target``: the bit a planted case is read on, where it is not bit 1 (a fault that blocks the
    common-variance cue is read on bit 5).
    ``solved``: the planted negative controls, parents that recover the target at the start: their verdict is their
    start.
    """
    map_seeds: tuple
    junction: dict
    carried_only_seeds: tuple = ()
    planted: dict = field(default_factory=dict)
    planted_target: dict = field(default_factory=dict)
    solved: tuple = ()

    @property
    def carried_seeds(self):
        return tuple(self.map_seeds) + tuple(self.carried_only_seeds)


def per_line(bit):
    """Per held-out line: summed contrast error and truth energy of one bit's flip pairs."""
    lines = defaultdict(lambda: [0.0, 0.0])
    for row in bit["pair_losses"]:
        totals = lines[row["context"].split("|")[0]]
        totals[0] += row["contrast_error"]
        totals[1] += row["truth_energy"]
    return lines


@lru_cache
def resamples(n_lines):
    """The bootstrap's draws of line indices, fixed by the seed and the number of lines: every run of a panel is
    resampled the same way, whatever else is read and in whatever order."""
    return np.random.default_rng(BOOTSTRAP_SEED).integers(n_lines, size=(RESAMPLES, n_lines))


def line_interval(bit):
    """Q of one bit and its 95% line-bootstrap interval, from the saved per-pair losses."""
    lines = per_line(bit)
    errors, energies = np.array([lines[line] for line in sorted(lines)]).T
    draws = resamples(len(errors))
    boot = 1 - errors[draws].sum(1) / energies[draws].sum(1)
    return float(1 - errors.sum() / energies.sum()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def interval_reading(low, high):
    """Recovered when the lower bound reaches L, not recovered when the upper bound is below it, else undetermined."""
    return "recovered" if low >= RECOVERY_LEVEL else "not recovered" if high < RECOVERY_LEVEL else "undetermined"


def margins(parent, end, half, target):
    """Whether the run keeps the learned bits and the raw error within their margins against the parent."""
    kept = end["mse"]["raw_O50"] <= ERROR_FACTOR * parent["mse"]["raw_O50"]
    if half == "synthetic":
        for bit in (b for b in ("0", "1", "5") if b != target):   # bits 0 and 5 for the default target
            before = parent["bits"][bit]["Q"]
            if bit == "0" or before >= .5:
                kept = kept and end["bits"][bit]["Q"] >= before - LEARNED_MARGIN
    return kept


def cell(run, target):
    """One map run: its held-out outcome, its costs on both panels and the parent's starting reading."""
    run = Path(run)
    row = json.loads((run / "RUN_RECORD.json").read_text())["row"]
    keys = {"model": row["model"], "half": row["half"], "seed": row["seed"], "case": row.get("case", "parent"),
            "cut": row["cut"], "arm": row["arm"], "rate": float(row["rate_multiple"]), "run": str(run)}
    parent = json.loads((run / "EVAL_PARENT.json").read_text())["supports"]   # written before any training
    start = interval_reading(*line_interval(parent["heldout"]["bits"][target])[1:])
    if (run / "DIVERGED.json").is_file():
        return keys | {"outcome": "diverged", "parent_start": start}
    end = json.loads((run / "EVAL.json").read_text())["after"]["supports"]
    heldout, validation = end["heldout"], end["validation"]
    q, low, high = line_interval(heldout["bits"][target])
    kept = margins(parent["heldout"], heldout, row["half"], target)
    reading = interval_reading(low, high)
    return keys | {"outcome": "recovered with collateral damage" if reading == "recovered" and not kept else reading,
                   "parent_start": start, "target_Q": q, "target_low": low, "target_high": high, "margins_met": kept,
                   "error_ratio": heldout["mse"]["raw_O50"] / parent["heldout"]["mse"]["raw_O50"],
                   "common_mean_change": heldout["bits"]["0"]["Q"] - parent["heldout"]["bits"]["0"]["Q"],
                   "common_variance_change": heldout["bits"]["5"]["Q"] - parent["heldout"]["bits"]["5"]["Q"],
                   "validation_target_Q": validation["bits"][target]["Q"],
                   "validation_raw_error": validation["mse"]["raw_O50"],
                   "validation_error_ratio": validation["mse"]["raw_O50"] / parent["validation"]["mse"]["raw_O50"],
                   "validation_margins_met": margins(parent["validation"], validation, row["half"], target)}


def arm_reading(rows):
    """Site reading of one arm over the runs given: recovers; does not recover (the whole grid run, an evaluable x1,
    every evaluable rate not recovered); not recovered at the rates run (the same with the grid incomplete);
    undetermined (no evaluable x1, or an interval straddling L). A diverged run covers its rate, counting neither
    way."""
    evaluable = [r for r in rows if r["outcome"] != "diverged"]
    if any(r["outcome"] in LEARNS for r in evaluable):
        return "recovers"
    if not any(r["rate"] == 1 for r in evaluable) or any(r["outcome"] != "not recovered" for r in evaluable):
        return "undetermined"
    return "does not recover" if set(GRID) <= {r["rate"] for r in rows} else "not recovered at the rates run"


def cost(rows):
    """Cost of an arm that recovers: clean if it recovered within the margins at some rate."""
    learned = [r for r in rows if r["outcome"] in LEARNS]
    if not learned:
        return ""
    return "clean" if any(r["outcome"] == "recovered" for r in learned) else "with collateral damage"


def rate_gated(rows):
    """Recovered only above x1, with an evaluable x1 that is not recovered."""
    at_one = [r for r in rows if r["rate"] == 1 and r["outcome"] != "diverged"]
    if not any(r["outcome"] in LEARNS and r["rate"] > 1 for r in rows):
        return ""
    if not at_one:
        return "no evaluable x1"
    if any(r["outcome"] in LEARNS for r in at_one):
        return "no"
    return "yes" if all(r["outcome"] == "not recovered" for r in at_one) else "x1 undetermined"


def rate_list(rows):
    return ";".join(f"x{r['rate']:g}:{r['outcome']}" for r in sorted(rows, key=lambda r: r["rate"]))


def label(key):
    cut, arm, rate = key
    return f"{cut}/{arm}/x{rate:g}"


def by_case(cells):
    groups = defaultdict(lambda: defaultdict(list))
    for c in cells:
        groups[(c["model"], c["half"], c["seed"], c["case"])][(c["cut"], c["arm"])].append(c)
    return groups


def case_start(arms):
    """The parent's starting reading: one saved evaluation and fixed resamples give every run of a case the same."""
    return next(r["parent_start"] for rows in arms.values() for r in rows)


def arms_table(map_runs):
    out = []
    for (model, half, seed, case), arms in sorted(by_case(map_runs).items()):
        for (cut, arm), rows in sorted(arms.items()):
            out.append({"model": model, "half": half, "seed": seed, "case": case, "cut": cut, "arm": arm,
                        "rates": rate_list(rows), "reading": arm_reading(rows), "cost": cost(rows),
                        "rate_gated": rate_gated(rows)})
    return out


def evaluable_rates(rows):
    return {r["rate"] for r in rows if r["outcome"] != "diverged"}


def contrast(kept_rows, reset_rows):
    """One contrast against arm a: its outcome and its matched rates.

    Arm a is read over all its runs: a recovery at any rate means no reset is needed, and its negative, the premise
    of a reset benefit, needs the whole grid. The reset arm's recovery counts only at matched rates, where both arms
    have an evaluable run, x1 among them; one that recovers only elsewhere leaves the contrast undetermined."""
    matched = evaluable_rates(kept_rows) & evaluable_rates(reset_rows)
    kept = arm_reading(kept_rows)
    if kept == "recovers":
        return "kept arm recovers", matched
    if kept != "does not recover" or 1.0 not in matched:
        return "undetermined", matched
    if arm_reading([r for r in reset_rows if r["rate"] in matched]) == "recovers":
        return "reset benefit", matched
    negatives = {"does not recover": "neither recovers",
                 "not recovered at the rates run": "neither recovers at the rates run"}
    return negatives.get(arm_reading(reset_rows), "undetermined"), matched


def contrast_row(name, kept_rows, reset_rows):
    """A contrast's columns; a single summary-statistics branch is read as sufficient or not, not against arm a."""
    if name.endswith("branch alone"):
        reading = arm_reading(reset_rows)
        outcome, matched, counted = ("sufficient" if reading == "recovers" else reading), None, reset_rows
    else:
        outcome, matched = contrast(kept_rows, reset_rows)
        counted = [r for r in reset_rows if r["rate"] in matched]
    elsewhere = ([] if matched is None else
                 sorted(r["rate"] for r in reset_rows if r["rate"] not in matched and r["outcome"] in LEARNS))
    return {"contrast": name, "kept_arm_rates": rate_list(kept_rows), "reset_arm_rates": rate_list(reset_rows),
            "matched_rates": "" if matched is None else ";".join(f"x{rate:g}" for rate in sorted(matched)),
            "kept_arm": arm_reading(kept_rows), "reset_arm": arm_reading(reset_rows), "outcome": outcome,
            "kept_arm_cost": cost(kept_rows), "reset_arm_cost": cost(counted),
            "reset_arm_recovered_at_other_rates": ";".join(f"x{rate:g}" for rate in elsewhere)}


def contrasts(map_runs):
    out = []
    for (model, half, seed, case), arms in sorted(by_case(map_runs).items()):
        start = case_start(arms)
        solved = "intervention effect on a parent solved at the start" if start == "recovered" else ""
        for cut in sorted({cut for cut, _ in arms} - {"input"}):
            kept_rows = arms.get((cut, "a"))
            if not kept_rows:
                continue
            for name, arm in [*CONTRASTS.items(), *((f"{b} branch alone", b) for b in BRANCHES)]:
                reset_rows = arms.get((cut, arm))
                if reset_rows:
                    out.append({"model": model, "half": half, "seed": seed, "case": case, "parent_start": start,
                                "cut": cut} | contrast_row(name, kept_rows, reset_rows) | {"reading": solved})
    return out


def cases(map_runs):
    """Per case: whether any parent-derived arm recovers, with the fresh network's reading beside it."""
    groups = by_case(map_runs)
    out = []
    for (model, half, seed, case), arms in sorted(groups.items()):
        readings = [arm_reading(rows) for (_, arm), rows in arms.items() if arm not in (FRESH, REFERENCE_ARM)]
        start = case_start(arms)
        if start == "recovered":
            derived = "solved at the start"
        elif "recovers" in readings:
            derived = "some recover"
        elif "undetermined" in readings:
            derived = "undetermined"
        elif all(reading == "does not recover" for reading in readings):
            derived = "none recovers"
        else:
            derived = "none recovers at the rates run"
        fresh = groups.get((model, "synthetic", seed, "parent"), {}).get(("input", FRESH), [])
        out.append({"model": model, "half": half, "seed": seed, "case": case, "parent_start": start,
                    "parent_derived_arms": derived, "fresh_network": arm_reading(fresh) if fresh else "not run"})
    return out


def qualifies(runs, seeds):
    """A qualified repair: the target recovered on the validation panel within the margins in every map seed."""
    return all(seed in runs and runs[seed]["outcome"] != "diverged" and runs[seed]["validation_margins_met"]
               and runs[seed]["validation_target_Q"] >= RECOVERY_LEVEL for seed in seeds)


def mean_validation_q(runs, seeds):
    values = [runs[seed]["validation_target_Q"] for seed in seeds
              if seed in runs and runs[seed]["outcome"] != "diverged"]
    return float(np.mean(values)) if values else ""


def seed_outcome(run):
    """One seed's held-out outcome of a handed-on cell, in origin resolution's categories."""
    if run is None:
        return "not run"
    if run["outcome"] == "diverged":
        return f"diverged ({OR_CATEGORY['diverged']})"
    q, low, high = (round(run[key], DIGITS[key]) for key in ("target_Q", "target_low", "target_high"))
    return f"{q} [{low}, {high}] {OR_CATEGORY[run['outcome']]}"


def or_category(runs):
    """One category over a cell's evaluable seeds: recovered (cost flagged when any seed's is), not recovered, or
    partial or mixed."""
    categories = [OR_CATEGORY[r["outcome"]] for r in runs if r["outcome"] != "diverged"]
    if not categories:
        return "no evaluable seed"
    if all(category.startswith("recovered") for category in categories):
        return OR_CATEGORY["recovered with collateral damage"] if OR_CATEGORY["recovered with collateral damage"] in categories \
            else "recovered"
    return "not recovered" if all(category == "not recovered" for category in categories) else "partial or mixed"


def cost_flagged(runs):
    """Evaluable seeds that recovered only with collateral damage, out of all evaluable seeds."""
    evaluable = [r for r in runs if r["outcome"] != "diverged"]
    return f"{sum(r['outcome'] == 'recovered with collateral damage' for r in evaluable)} of {len(evaluable)}"


def carried_menus(cells):
    """(model, half) -> (cut, arm, rate) -> seed -> cell, over the cells that may be handed on."""
    menus = defaultdict(lambda: defaultdict(dict))
    for c in cells:
        if c["case"] == "parent" and c["cut"] != "input" and c["arm"] not in (FRESH, REFERENCE_ARM):
            menus[(c["model"], c["half"])][(c["cut"], c["arm"], c["rate"])][c["seed"]] = c
    return menus


def carried(cells, settings):
    """The cell handed on per model and half, chosen on the validation panel only, with its per-seed
    outcomes in its own half and in the other. Returns the ``MAP_CARRIED`` rows and, per model and handed-on cell,
    the halves that selected it (``carried_lines`` reads their per-line recovery)."""
    menus = carried_menus(cells)
    chosen = {}
    for (model, half), menu in menus.items():
        starts = {c["parent_start"] for runs in menu.values() for seed, c in runs.items() if seed in settings.map_seeds}
        if starts == {"recovered"}:   # every map seed's parent recovers the target at the start: nothing to locate
            chosen[(model, half)] = None, "solved at the start (no obstacle to locate)"
            continue
        qualified = [key for key, runs in menu.items() if qualifies(runs, settings.map_seeds)]
        if qualified:
            chosen[(model, half)] = (max(qualified, key=lambda key: mean_validation_q(menu[key], settings.map_seeds)),
                                     "qualified repair")
        else:
            chosen[(model, half)] = ((settings.junction[model], "a", 1.0),
                                     "diagnostic intervention (qualified repair: none)")
    out, selected_in = [], defaultdict(list)
    for (model, half), (key, kind) in sorted(chosen.items(), key=lambda item: item[0]):
        if key is None:
            out.append({"model": model, "half": half, "kind": kind, "carried": ""})
            continue
        other = "natural" if half == "synthetic" else "synthetic"
        own, across = menus[(model, half)].get(key, {}), menus.get((model, other), {}).get(key, {})
        out.append({"model": model, "half": half, "kind": kind, "carried": label(key),
                    "mean_validation_target_Q": mean_validation_q(own, settings.map_seeds),
                    **{f"seed_{seed}": seed_outcome(own.get(seed)) for seed in settings.carried_seeds},
                    "or_category": or_category(own.values()), "cost_flagged": cost_flagged(own.values()),
                    "other_half": other,
                    **{f"other_half_seed_{seed}": seed_outcome(across.get(seed)) for seed in settings.carried_seeds},
                    "other_half_or_category": or_category(across.values())})
        selected_in[(model, key)].append(half)
    return out, selected_in


def carried_lines(cells, selected_in, target):
    """Per-line held-out recovery of each handed-on cell, once per cell, read in both halves from its runs' saved
    ``EVAL.json``."""
    menus = carried_menus(cells)
    out = []
    for (model, key), halves in selected_in.items():
        for read_in in ("natural", "synthetic"):
            for seed, run in sorted(menus.get((model, read_in), {}).get(key, {}).items()):
                if run["outcome"] == "diverged":
                    continue
                end = json.loads((Path(run["run"]) / "EVAL.json").read_text())["after"]["supports"]
                for line, (error, energy) in sorted(per_line(end["heldout"]["bits"][target]).items()):
                    out.append({"model": model, "half": read_in, "selected_in": "; ".join(halves), "seed": seed,
                                "carried": label(key), "line": line,
                                "target_Q": 1 - error / energy if energy > 0 else "",
                                "contrast_error": error, "truth_energy": energy})
    return out


def planted_verdict(found, answer, whole_network, scheduled):
    """A planted case's verdict from its contrasts at its cut, against its admissible ``answer``
    (``Settings.planted``). An open contrast reads incomplete while runs are scheduled for the case (or a contrast
    was not run at all), and a miss under limited coverage once nothing more is scheduled, so that the check always
    concludes."""
    _, must, must_not, alternative = answer
    if any(found.get(name) == "reset benefit" for name in must_not):
        return "wrong"
    if any(found.get(name) == "kept arm recovers" for name in must | must_not):
        return "fault not binding"
    by_alternative = {name for name in must & alternative if found.get(name) == "neither recovers"}

    def met(name):
        return found.get(name) == "reset benefit" or (name in by_alternative and whole_network == "recovers")

    if all(met(name) for name in must) and all(found.get(name) == "neither recovers" for name in must_not):
        return "pass"
    needed = [found.get(name, "not run") for name in must | must_not]
    open_ = (any(outcome not in ("reset benefit", "neither recovers") for outcome in needed)
             or (by_alternative and whole_network not in ("recovers", "does not recover")))
    if "not run" in needed or (scheduled and open_):
        return "incomplete"
    return "miss (limited coverage)" if open_ else "miss"


def known_answer(contrast_rows, case_rows, wave_rows, arm_rows, target, settings):
    """Every planted-fault case present against its admissible answer: pass, miss, wrong, fault not binding or
    incomplete. Missing evidence reads incomplete, never wrong."""
    planted = settings.planted
    out = []
    for row in case_rows:
        if row["case"] in settings.solved:
            start = row["parent_start"]
            result = ("pass" if start == "recovered" else "wrong: read as not recovered at the start"
                      if start == "not recovered" else "incomplete: the start is undetermined")
            out.append({"case": row["case"], "seed": row["seed"], "result": result})
    found, costs = defaultdict(dict), defaultdict(dict)
    for row in contrast_rows:
        if row["case"] in planted and row["cut"] == planted[row["case"]][0]:
            found[(row["case"], row["seed"])][row["contrast"]] = row["outcome"]
            costs[(row["case"], row["seed"])][row["contrast"]] = row["reset_arm_cost"]
    whole = {(row["case"], row["seed"]): row["reading"] for row in arm_rows if (row["cut"], row["arm"]) == ("input", "a")}
    for case, seed in sorted({(row["case"], row["seed"]) for row in case_rows if row["case"] in planted
                              and settings.planted_target.get(row["case"], "1") == target}):   # read on its own bit
        cut, must, must_not, _ = planted[case]
        outcomes = found[(case, seed)]
        scheduled = any(row["case"] == case and row["seed"] == seed and row["cut"] in (cut, "input") for row in wave_rows)
        out.append({"case": case, "seed": seed,
                    "result": planted_verdict(outcomes, planted[case], whole.get((case, seed), "not run"), scheduled),
                    **{name: outcomes.get(name, "not run") for name in sorted(must | must_not)},
                    **{f"{name} cost": costs[(case, seed)].get(name, "") for name in sorted(must)},
                    "whole network": whole.get((case, seed), "not run")})
    return out


def learns(runs):
    """The target is learned on the validation panel at some rate run (validation recovery at least L)."""
    return any(c["outcome"] != "diverged" and c["validation_target_Q"] >= RECOVERY_LEVEL for c in runs.values())


def second_wave(map_runs, solved):
    """Runs to add, read on the validation panel; one row per (arm, rate). ``solved`` as ``Settings.solved``."""
    cuts = defaultdict(lambda: defaultdict(dict))   # (model, half, seed, case, cut) -> arm -> rate -> cell
    for c in map_runs:
        if c["case"] not in solved:   # its verdict is its start: no negative of it carries a conclusion
            cuts[(c["model"], c["half"], c["seed"], c["case"], c["cut"])][c["arm"]][c["rate"]] = c
    rows = []
    for (model, half, seed, case, cut), arms in sorted(cuts.items()):
        not_learning = sorted(arm for arm, runs in arms.items() if not learns(runs))
        if case != "parent":
            # the wording of the saved tables
            add, reason = not_learning, "known-answer case: every arm that does not learn"
        elif not any(learns(runs) for arm, runs in arms.items() if arm not in (FRESH, REFERENCE_ARM)):
            add = [arm for arm in not_learning if arm != REFERENCE_ARM]
            reason = "no arm at this cut learns: the cut's negative"
        else:
            add = [arm for arm in not_learning if arm == "a"]
            reason = "another arm at this cut learns: the premise of a reset benefit"
        rows += [{"model": model, "half": half, "seed": seed, "case": case, "cut": cut, "arm": arm, "rate": rate,
                  "reason": reason} for arm in add for rate in GRID if rate not in arms[arm]]
    return rows


def arm_a_at(menu, cut):
    """Arm a at a cut at x1, the fixed block or the continued training, if it was run."""
    return next((c for c in menu if (c["cut"], c["arm"], c["rate"]) == (cut, "a", 1)), None)


def picks(map_runs, junction):
    """Selection picks on the menu, every map run of a parent case except the fresh network, read on the
    validation panel and evaluated on the held-out panel; ``junction`` as ``Settings.junction``."""
    menus = defaultdict(list)
    for c in map_runs:
        if c["case"] == "parent" and c["arm"] != FRESH and c["outcome"] != "diverged":
            menus[(c["model"], c["half"], c["seed"])].append(c)
    out = []
    for (model, half, seed), menu in sorted(menus.items()):
        acceptable = [c for c in menu if c["validation_margins_met"]]
        chosen = {"target-aware (IL)": max(acceptable, key=lambda c: c["validation_target_Q"], default=None),
                  "Cross-Val-style": min(menu, key=lambda c: c["validation_raw_error"]),
                  "fixed block": arm_a_at(menu, junction[model]), "continued training": arm_a_at(menu, "input")}
        for rule, c in chosen.items():
            out.append({"model": model, "half": half, "seed": seed, "rule": rule, "menu": len(menu),
                        "pick": "" if c is None else label((c["cut"], c["arm"], c["rate"])),
                        "heldout_target_Q": "" if c is None else c["target_Q"],
                        "heldout_outcome": (c["outcome"] if c is not None
                                            else "no acceptable candidate" if rule.startswith("target") else "not run"),
                        "heldout_error_ratio": "" if c is None else c["error_ratio"]})
        out.append({"model": model, "half": half, "seed": seed, "rule": "random (menu mean)", "menu": len(menu),
                    "pick": "", "heldout_target_Q": float(np.mean([c["target_Q"] for c in menu])),
                    "heldout_outcome": f"{sum(c['outcome'] == 'recovered' for c in menu)}/{len(menu)} recovered",
                    "heldout_error_ratio": float(np.mean([c["error_ratio"] for c in menu]))})
    return out


def tables(runs, settings, target="1"):
    """The reading tables of the finished runs among ``runs`` (run folders) under ``settings``, keyed by file name.

    A run is finished when it holds ``COMPLETE.json`` or ``DIVERGED.json``; the others are skipped. Runs are read in
    sorted path order, which fixes the row order and the ties of the carried choice and the picks."""
    if target not in TARGETS:
        raise ValueError(f"The target must be a scored bit ({', '.join(TARGETS)}), not {target!r}")
    cells = [cell(run, target) for run in sorted(map(Path, runs))
             if (run / "COMPLETE.json").is_file() or (run / "DIVERGED.json").is_file()]
    repeated = [key for key, count in Counter(tuple(c[k] for k in KEY) for c in cells).items() if count > 1]
    if repeated:
        raise ValueError(f"More than one run of the same map cell: {repeated}")
    map_runs = [c for c in cells if c["case"] != "parent" or c["seed"] in settings.map_seeds]
    contrast_rows, case_rows = contrasts(map_runs), cases(map_runs)
    carried_rows, selected_in = carried(cells, settings)
    wave_rows = second_wave(map_runs, settings.solved)
    arm_rows = arms_table(map_runs)
    named = {"MAP_CELLS": cells, "MAP_ARMS": arm_rows, "MAP_CONTRASTS": contrast_rows, "MAP_CASES": case_rows,
             "MAP_CARRIED": carried_rows, "MAP_CARRIED_LINES": carried_lines(cells, selected_in, target),
             "MAP_KNOWN_ANSWER": known_answer(contrast_rows, case_rows, wave_rows, arm_rows, target, settings),
             "MAP_WAVE2": wave_rows, "MAP_PICKS": picks(map_runs, settings.junction)}
    suffix = "" if target == "1" else f"_bit{target}"
    return {f"{name}{suffix}.csv": rows for name, rows in named.items()}


def write(folder, tables):
    """Write each table into ``folder``, replacing any earlier version even when it has no rows; numbers are rounded
    only here."""
    for name, rows in tables.items():
        with (Path(folder) / name).open("w", newline="") as handle:
            if rows:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows({key: round(value, DIGITS[key]) if key in DIGITS and isinstance(value, float) else value
                                  for key, value in row.items()} for row in rows)

