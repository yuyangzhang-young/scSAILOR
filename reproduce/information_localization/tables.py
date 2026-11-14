"""The tables of information localization on Tahoe-100M, rebuilt from the saved run records under the data root: the
map tables of the six models, the training-longer table and the cue-profile tables.

* **Map tables** (``MAP_*.csv``), per map folder (``MAPS``: the built models on folds 1-3, the planted faults, State
  on folds 1-3, CPA and PrePR-CT): the folder's runs (``<folder>/map/<run>/``:
  ``RUN_RECORD.json``, ``EVAL_PARENT.json``, then ``EVAL.json`` with ``COMPLETE.json``, or ``DIVERGED.json``) read
  by ``scsailor.localization.readings`` under the paper's settings (``READINGS``), for the selective shift (bit 1)
  and, where the folder lists them, the shared shift (bit 0) and the shared spread (bit 5). A table's ``run`` column
  is the run's folder relative to the data root.
* **Training longer** (``TIME_AXIS.csv``): the scores of the parents' saved checkpoints,
  ``EVAL_<model>_<seed>_<step>.json`` in ``TIME_AXIS``'s ``checkpoint-scores/`` (``time_and_presence``).
* **Cue profiles** (``PRESENCE_TABLE_seen.csv``, ``PRESENCE_TABLE_heldout.csv``): the saved profiles
  ``PRESENCE_<half>_<model>_<seed>[_<case>].json`` in ``PRESENCE``'s ``profiles/`` (a ``_<case>`` suffix names a
  planted-fault parent).
* **Onset** (``ONSET_TABLE.csv`` in ``PRESENCE``): at each key cut, the energy coefficient of the selective-shift cue
  (bit 1), of the shared-shift cue (bit 0) and of the null test of the shift cue (bit 4, no response), and the first
  divided by the other two (``selective_to_shared``, ``selective_to_null``: the quotients ``PRESENCE_TABLE``'s
  ``ratio_to_learned`` and ``ratio_to_null`` give, here at full precision), beside the checkpoint's selective-shift
  recovery from the training-longer table. Rows: the synthetic-trained parents of the three built models at 40,000
  updates (``profiles/``) and the later checkpoints of the attention model and the summary-statistics model
  (``checkpoint-profiles/PRESENCE_synthetic_<model>_<seed>_step<updates>.json``). Fig. 4d's cue ratio is 100 x
  ``selective_to_shared`` on the held-out support, at the inner cuts (``INNER_CUTS``); Supplementary Note S8.5's ratio
  at the first trained layer reads ``FIRST_TRAINED_LAYER`` at 40,000 updates on both supports.
* **Planted faults** (``PLANTED_FAULTS_TABLE.csv`` in ``PLANTED``, Supplementary Fig. 2): per planted fault and
  unfaulted control and per starting point, held-out recovery of the tested response for the original model (the
  original models' reading, mean over seeds) and after retraining (the best over cuts and rates of the mean over
  seeds), the lowest rate whose best reached the recovery level of 0.25, and the rates run (``asterisk``: run only at
  x1 and x30); read from the rebuilt ``MAP_CELLS`` tables and the runs' ``EVAL_PARENT.json``.
* **GPU hours of the light version** (``LIGHT_VERSION_GPU_HOURS.csv`` in ``LIGHT``, Supplementary Note S8.7): per
  model and fold of the Tahoe-100M maps (``gpu_hours``, ``estimated_hours``): the GPU (``GPU``); the full procedure's
  runs at the map seeds and their recorded training hours; the light runs (the menu, ``light.menus.runs``, and its
  second-seed checks, ``light.selection.light_cases``), the light runs present in the map and their recorded hours. Hours
  are given where every run had a GPU to itself; for the built models, whose runs shared one workstation GPU, the
  median hours per run (training seconds over the runs sharing the GPU) times the light runs estimates them.

The saved files, and so these tables, name the attention model ``state`` and State ``official_state``; files written
under the public name ``attention_model`` are read too, and tabled under ``state``.

    python -m reproduce.information_localization.tables --data DATA --out OUT [--compare DATA]

``--compare`` prints, for each written table, ``equal`` or ``DIFFERS`` against the table at the same path under the
given root, cell for cell, or ``not deposited``. It leaves out the ``run`` column, since a deposited table names the
runs that are not deposited ``not-deposited:<run>``, and the saved ``MAP_CELLS`` rows of fold 1's finer DeepSets cut
(``NOT_IN_MAP``), which the rebuild does not read.
"""
import argparse
import csv
import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from scsailor.localization import presence, readings, training_time
from scsailor.localization.light import maps, menus, rules, selection
from scsailor.localization.starts import PARENT_STEP

from ..assay.study import SAVED_NAMES, SEEDS

__all__ = ["READINGS", "MAPS", "TIME_AXIS", "PRESENCE", "PLANTED", "LIGHT", "map_tables", "score_name", "profile_case",
           "time_axis", "presence_tables", "onset_rows", "onset_table", "planted_fault_cell", "planted_faults", "gpu_hours",
           "estimated_hours", "light_hours", "main"]

# Each model's designated cut under its saved name (the attention model ``state``, State ``official_state``) and the
# attention model's public name.
READINGS = readings.Settings(
    map_seeds=SEEDS[:2],
    carried_only_seeds=SEEDS[2:],
    junction={"summary": "projections", "deepsets": "phi", "state": "head", "attention_model": "head",
              "official_state": "head", "prepr": "prediction", "cpa": "decoder"},
    planted={
        "dead-interaction": ("projections", {"back"}, {"drug side", "front", "mean projection", "variance projection"},
                             set()),
        "blocked-mean-cue": ("projections", {"front", "mean projection"}, {"variance projection", "back", "drug side"},
                             {"front", "mean projection"}),
        "ds-dead-back": ("phi", {"back"}, {"drug side", "front"}, set()),
        "ds-blocked-cue": ("phi", {"front"}, {"back", "drug side"}, {"front"}),
        "blocked-variance-cue": ("projections", {"front", "variance projection"},
                                 {"mean projection", "back", "drug side"}, {"front", "variance projection"}),
    },
    planted_target={"blocked-variance-cue": "5"},
    solved=("solved", "ds-solved"),
)
MAPS = {   # map folder -> the targets it is read on
    "information-localization-maps-fold-1": ("1", "0", "5"),
    "information-localization-maps-fold-2": ("1", "0", "5"),
    "information-localization-maps-fold-3": ("1", "0", "5"),
    "information-localization-attention-model": ("1", "0", "5"),
    "information-localization-state": ("1", "0", "5"),
    "information-localization-folds-2-3/F2": ("1", "0", "5"),
    "information-localization-folds-2-3/F3": ("1", "0", "5"),
    "information-localization-published-models/cpa": ("1", "0", "5"),
    "information-localization-published-models/prepr": ("1", "0", "5"),
    "information-localization-planted-faults-deepsets": ("1",),
    "information-localization-planted-faults-variance": ("5",),
}
NOT_IN_MAP = "-phi_mid-"   # fold 1 holds two runs of a finer DeepSets cut, which is not one of the map's cuts
TIME_AXIS = "information-localization/recovery-by-training-length"
PRESENCE = "information-localization/presence"
PLANTED = "information-localization/planted-faults"
LIGHT = "information-localization/light-version"
MODEL_NAME = {"summary": "Summary-statistics", "deepsets": "DeepSets", "state": "Attention", "official_state": "State",
              "cpa": "CPA", "prepr": "PrePR-CT"}   # saved names -> the paper's
ONSET_MODELS = ("state", "summary", "deepsets")
FIRST_TRAINED_LAYER = {"summary": "backbone.sproj", "deepsets": "backbone.phi.0:kernel", "state": "backbone.input:kernel"}
INNER_CUTS = {"state": ("backbone.blocks.0:kernel", "backbone.blocks.3:kernel", "backbone.blocks.7:kernel", "backbone.head.input"),
              "summary": ("backbone.sproj", "backbone.common_hidden", "backbone.inter_hidden", "backbone.w0", "backbone.w1")}
# Supplementary Fig. 2's rows: (group, model, case, the case's name, map folder, target bit)
FOLD_1, DEEPSETS_FAULTS = "information-localization-maps-fold-1", "information-localization-planted-faults-deepsets"
UNFAULTED = "Unfaulted control"
PLANTED_ROWS = (("Fault in back", "summary", "dead-interaction", "Disabled back", FOLD_1, "1"),
                ("Fault in back", "deepsets", "ds-dead-back", "Disabled back", DEEPSETS_FAULTS, "1"),
                ("Fault in front", "summary", "blocked-mean-cue", "Blocked shift cue", FOLD_1, "1"),
                ("Fault in front", "summary", "blocked-variance-cue", "Blocked spread cue",
                 "information-localization-planted-faults-variance", "5"),
                ("Fault in front", "deepsets", "ds-blocked-cue", "Blocked shift cue", DEEPSETS_FAULTS, "1"),
                (UNFAULTED, "summary", "solved", UNFAULTED, FOLD_1, "1"),
                (UNFAULTED, "deepsets", "ds-solved", UNFAULTED, DEEPSETS_FAULTS, "1"),
                (UNFAULTED, "official_state", "parent", UNFAULTED, "information-localization-state", "0"))
TARGET_NAME = {"0": "Shared shift", "1": "Selective shift", "5": "Shared spread"}
STARTS = {"orig": "Original model", "ft": "Fine-tuning", "back": "Back reset", "front": "Front reset",
          "whole": "Whole network"}
START_RUNS = {"ft": lambda r: r["arm"] == "a" and r["cut"] != "input", "back": lambda r: r["arm"] == readings.CONTRASTS["back"],
              "front": lambda r: r["arm"] == readings.CONTRASTS["front"], "whole": lambda r: r["arm"] == "a" and r["cut"] == "input"}
ONLY_X1_X30 = {1., 30.}   # Supplementary Fig. 2 marks a cell run only at these rates with an asterisk
# The Tahoe-100M map folders of the light version's comparison: every map but the planted faults'
LIGHT_MAPS = tuple(f"{folder}/map" for folder in MAPS if "planted-faults" not in folder)
PUBLIC_NAME = {"state": "attention_model", "official_state": "state"}   # saved -> the light package's names
# The GPU every run of a model's map trained on, by saved model name and fold (the run records do not name it): the
# attention model and State on H200s and CPA and PrePR-CT on V100s, one run per GPU; the summary-statistics model and
# DeepSets on one RTX 5090 workstation GPU, several runs at a time.
GPU = {("state", "F1"): "H200", ("state", "F2"): "H200", ("state", "F3"): "H200",
       ("official_state", "F1"): "H200", ("official_state", "F2"): "H200", ("official_state", "F3"): "H200",
       ("cpa", "F1"): "V100", ("prepr", "F1"): "V100",
       ("summary", "F1"): "RTX 5090", ("summary", "F2"): "RTX 5090", ("summary", "F3"): "RTX 5090",
       ("deepsets", "F1"): "RTX 5090", ("deepsets", "F2"): "RTX 5090", ("deepsets", "F3"): "RTX 5090"}


def map_tables(root, folder, target):
    """The map tables of one folder for one target, keyed by file name."""
    runs = [run for run in sorted((root / folder / "map").iterdir()) if NOT_IN_MAP not in run.name]
    tables = readings.tables(runs, READINGS, target)
    for rows in tables.values():
        for row in rows:
            if "run" in row:
                row["run"] = Path(row["run"]).relative_to(root).as_posix()
    return tables


SAVED_NAME = {public: saved for saved, public in SAVED_NAMES.items()}   # attention_model -> state


def score_name(stem):
    """``(model, seed, step)`` of a score file ``EVAL_<model>_<seed>_<step>``, the model under its saved name."""
    model, seed, step = stem.removeprefix("EVAL_").rsplit("_", 2)
    return SAVED_NAME.get(model, model), seed, step


def profile_case(stem, seed):
    """The case of a profile file ``PRESENCE_<half>_<model>_<seed>[_<case>]``: ``parent`` without a suffix."""
    return stem.partition(f"_{seed}")[2].removeprefix("_") or "parent"


def time_axis(folder):
    """The rows of ``TIME_AXIS.csv`` from every ``EVAL_<model>_<seed>_<step>.json`` in ``folder``."""
    return training_time.time_axis([(score_name(path.stem), json.loads(path.read_text()))
                                    for path in sorted(folder.glob("EVAL_*.json"))])


def presence_tables(folder):
    """``{support: rows}`` of the cue-profile tables from every saved profile in ``folder``; the attention model's
    profiles are read under its public name and tabled under the saved one."""
    profiles = []
    for path in sorted(folder.glob("PRESENCE_*.json")):
        result = json.loads(path.read_text())
        result["model"] = SAVED_NAMES.get(result["model"], result["model"])
        profiles.append((profile_case(path.stem, result["seed"]), result))
    tables = {}
    for support in ("seen", "heldout"):
        tables[support] = presence.table(profiles, support)
        for row in tables[support]:
            row["model"] = SAVED_NAME.get(row["model"], row["model"])
    return tables


def onset_rows(model, seed, step, result, time_axis):
    """The onset rows of one cue profile (``model`` under its saved name): per support and key cut, the energy
    coefficients of the selective-shift cue (bit 1), the shared-shift cue (bit 0) and the null test of the shift cue
    (bit 4) and the first divided by the other two (blank where the other is not positive), beside the checkpoint's
    selective-shift recovery (``time_axis``: (model, seed, step, support) -> Q1)."""
    rows = []
    for support, block in result["supports"].items():
        bits = block["bits"]
        for cut in presence.KEY_CUTS[SAVED_NAMES.get(model, model)]:
            if cut not in bits["1"]["cuts"]:
                continue
            e1, e0, e4 = (bits[str(b)]["cuts"][cut]["energy_coefficient"]["value"] for b in (1, presence.LEARNED[1], presence.NULL[1]))
            rows.append({"Model": MODEL_NAME[model], "seed": seed, "step": step, "support": support, "cut": cut,
                         "selective_Q1": time_axis.get((model, seed, step, support), ""), "energy_selective": e1,
                         "energy_shared": e0, "energy_null": e4, "selective_to_shared": e1 / e0 if e0 > 0 else "",
                         "selective_to_null": e1 / e4 if e4 > 0 else ""})
    return rows


def onset_table(folder, time_rows):
    """``ONSET_TABLE.csv``'s rows from the synthetic-trained parents' profiles at 40,000 updates (``profiles/``) and the
    later checkpoints' (``checkpoint-profiles/``; a profile at 40,000 updates there repeats the parent's and is left
    out)."""
    time = {(r["model"], r["seed"], r["step"], r["support"]): r["Q1"] for r in time_rows}
    rows = []
    for path in sorted((folder / "profiles").glob("PRESENCE_synthetic_*.json")):
        result = json.loads(path.read_text())
        model = SAVED_NAME.get(result["model"], result["model"])
        if model in ONSET_MODELS and profile_case(path.stem, result["seed"]) == "parent":
            rows += onset_rows(model, result["seed"], PARENT_STEP, result, time)
    for path in sorted((folder / "checkpoint-profiles").glob("PRESENCE_synthetic_*_step*.json")):
        model, seed, step = path.stem.removeprefix("PRESENCE_synthetic_").rsplit("_", 2)
        step = int(step.removeprefix("step"))
        if step != PARENT_STEP:
            rows += onset_rows(SAVED_NAME.get(model, model), int(seed), step, json.loads(path.read_text()), time)
    return rows


def planted_fault_cell(rows, start, parent=None):
    """One cell of Supplementary Fig. 2 from a case's ``MAP_CELLS`` rows. Original model (``orig``): the mean over
    seeds of each seed's original model (``parent``: a run -> its original model's held-out recovery, read from the
    seed's first run). A starting point: per cut and rate the mean over seeds of the finished runs' held-out recovery;
    the best of these, the lowest rate whose best over cuts reached the recovery level, and the rates run."""
    if start == "orig":
        first = {}
        for r in rows:
            first.setdefault(r["seed"], r["run"])
        return {"recovery": float(np.mean([parent(run) for run in first.values()])), "asterisk": False,
                "lowest_rate_reaching_0.25": "", "rates_run": ""}
    found = {}
    for r in rows:
        if START_RUNS[start](r) and r["outcome"] != "diverged" and r["target_Q"] != "":
            found.setdefault((r["cut"], float(r["rate"])), []).append(float(r["target_Q"]))
    means = {key: float(np.mean(values)) for key, values in found.items()}
    best = {}
    for (_, rate), value in means.items():
        best[rate] = max(best.get(rate, -np.inf), value)
    return {"recovery": max(means.values()), "asterisk": set(best) <= ONLY_X1_X30,
            "lowest_rate_reaching_0.25": min((rate for rate, v in best.items() if v >= readings.RECOVERY_LEVEL), default=""),
            "rates_run": " ".join(f"{rate:g}" for rate in sorted(best))}


def planted_faults(root, out):
    """Supplementary Fig. 2's table from the rebuilt ``MAP_CELLS`` tables under ``out`` (synthetic-trained runs) and the
    runs' ``EVAL_PARENT.json`` under ``root``."""
    table = []
    for group, model, case, name, folder, bit in PLANTED_ROWS:
        suffix = "" if bit == "1" else f"_bit{bit}"
        with (out / folder / f"MAP_CELLS{suffix}.csv").open(newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if (r["model"], r["case"], r["half"]) == (model, case, "synthetic")]
        parent = lambda run: json.loads((root / run / "EVAL_PARENT.json").read_text())["supports"]["heldout"]["bits"][bit]["Q"]
        for start, start_name in STARTS.items():
            table.append({"group": group, "model": MODEL_NAME[model], "case": name, "target": TARGET_NAME[bit],
                          "starting_point": start_name, **planted_fault_cell(rows, start, parent)})
    return table


def run_records(runs):
    """One dict per finished run of ``runs`` (``maps.read_map``'s rows): its identity, training seconds (complete runs
    only; a diverged run has none), device and the runs it shared a GPU with. The saved records list those runs
    (``lockstep_group``); ``scsailor.localization.starts`` does not write that field, so a run it records reads as having
    had a GPU to itself."""
    records = []
    for r in runs.itertuples():
        execution = json.loads((Path(r.run) / "EXECUTION.json").read_text())
        seconds = json.loads((Path(r.run) / "EVAL.json").read_text())["train_seconds"] if r.status == "complete" else None
        records.append({"model": r.model, "fold": r.fold, "half": r.half, "seed": r.seed, "case": r.case, "cut": r.cut,
                        "arm": r.arm, "rate": r.rate, "seconds": seconds, "device": execution.get("device"),
                        "group": execution.get("lockstep_group") or []})
    return records


def gpu_hours(runs, light, checks):
    """One model and fold (``runs``: its map's records): the full procedure's runs at the map seeds and their training
    hours; the light runs (``light``, the menu's runs, and ``checks``, the second-seed checks: (half, seed, cut, arm,
    rate)), those present in the map and their training hours. Hours, to 0.1, are given where no run shared its GPU; a
    diverged run adds none."""
    full = [r for r in runs if r["case"] == "parent" and r["seed"] in rules.MAP_SEEDS]
    light = set(light) | set(checks)
    used = [r for r in full if (r["half"], r["seed"], r["cut"], r["arm"], r["rate"]) in light]
    alone = not any(r["group"] for r in full)
    hours = lambda part: round(sum(r["seconds"] or 0. for r in part) / 3600, 1) if alone else ""
    return {"full_runs": len(full), "full_hours": hours(full), "light_runs": len(light), "light_runs_in_map": len(used),
            "light_hours": hours(used)}


def estimated_hours(runs, light_runs):
    """For runs that shared a GPU: the GPU runs with recorded training seconds, the median hours per run (seconds over
    the runs sharing the GPU) and that median times ``light_runs``."""
    share = [r["seconds"] / 3600 / max(1, len(r["group"])) for r in runs if r["seconds"] and r["device"] == "cuda"]
    middle = median(share)
    return len(share), middle, middle * light_runs


def light_check_runs(runs):
    """(model, fold) -> the light version's second-seed checks (primary seed, fixed window) on the maps' runs
    (``maps.read_map``'s rows, models under the light package's names), as (half, seed, cut, arm, rate)."""
    found = {}
    for r in selection.light_cases(runs, rules.cells(runs)).itertuples():
        for cell in filter(None, r.checked.split(";")):
            cut, arm, rate = cell.split("/")
            found.setdefault((r.model, r.fold), []).append((r.half, menus.SECOND_SEED, cut, arm, float(rate[1:])))
    return found


def light_hours(root):
    """``LIGHT_VERSION_GPU_HOURS.csv``'s rows: per model and fold of the Tahoe-100M maps, ``gpu_hours``; for the runs that
    shared a GPU, ``estimated_hours`` over every finished run of the model's map. The finer DeepSets cut of fold 1
    (``NOT_IN_MAP``) is left out, as from the map tables."""
    runs = pd.concat([maps.read_map(root / folder, "tahoe") for folder in LIGHT_MAPS], ignore_index=True)
    runs = runs[~runs.run.str.contains(NOT_IN_MAP)]
    checks = light_check_runs(runs.assign(model=runs.model.replace(PUBLIC_NAME)))
    groups = {}
    for r in run_records(runs):
        groups.setdefault((r["model"], r["fold"]), []).append(r)
    rows = []
    for (model, fold), mine in sorted(groups.items()):
        public = PUBLIC_NAME.get(model, model)
        row = {"Model": MODEL_NAME[model], "fold": fold, "gpu": GPU[(model, fold)],
               **gpu_hours(mine, menus.runs(public), checks.get((public, fold), []))}
        if row["full_hours"] == "":
            row["runs_with_records"], row["median_hours_per_run"], row["estimated_light_hours"] = estimated_hours(mine, row["light_runs"])
        rows.append(row)
    return rows


def read_rows(path, drop=("run",)):
    with Path(path).open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return []
    keep = [i for i, name in enumerate(rows[0]) if name not in drop]
    return [[row[i] for i in keep] for row in rows]


def written(root, out):
    """Every table, written under ``out`` at its path under the data root; returns the paths written."""
    paths = []
    for folder, targets in MAPS.items():
        (out / folder).mkdir(parents=True, exist_ok=True)
        for target in targets:
            tables = map_tables(root, folder, target)
            readings.write(out / folder, tables)
            paths += [Path(folder) / name for name in tables]
    (out / TIME_AXIS).mkdir(parents=True, exist_ok=True)
    time_rows = time_axis(root / TIME_AXIS / "checkpoint-scores")
    training_time.write_time_axis(out / TIME_AXIS / "TIME_AXIS.csv", time_rows)
    paths.append(Path(TIME_AXIS) / "TIME_AXIS.csv")
    (out / PRESENCE).mkdir(parents=True, exist_ok=True)
    for support, rows in presence_tables(root / PRESENCE / "profiles").items():
        presence.write_table(out / PRESENCE / f"PRESENCE_TABLE_{support}.csv", rows)
        paths.append(Path(PRESENCE) / f"PRESENCE_TABLE_{support}.csv")
    for folder, name, rows in ((PRESENCE, "ONSET_TABLE.csv", onset_table(root / PRESENCE, time_rows)),
                               (PLANTED, "PLANTED_FAULTS_TABLE.csv", planted_faults(root, out)),
                               (LIGHT, "LIGHT_VERSION_GPU_HOURS.csv", light_hours(root))):
        (out / folder).mkdir(parents=True, exist_ok=True)
        readings.write(out / folder, {name: rows})
        paths.append(Path(folder) / name)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rebuild information localization's tables on Tahoe-100M from the "
                                                 "saved run records.")
    parser.add_argument("--data", type=Path, required=True, help="the data root")
    parser.add_argument("--out", type=Path, required=True, help="where the tables are written, at their data-root paths")
    parser.add_argument("--compare", type=Path, help="a root holding the saved tables at the same paths")
    args = parser.parse_args(argv)
    paths = written(args.data, args.out)
    print(f"{len(paths)} tables written under {args.out}")
    if args.compare:
        for path in paths:
            saved = args.compare / path
            if not saved.is_file():
                print(f"not deposited  {path}")
                continue
            old, new = read_rows(saved), read_rows(args.out / path)
            if path.name.startswith("MAP_CELLS"):   # a saved table may list the runs of the finer DeepSets cut
                runs = read_rows(saved, drop=())
                old = [row for row, full in zip(old, runs) if NOT_IN_MAP not in ",".join(full)]
            print(f"{'equal' if old == new else 'DIFFERS'}  {path}")


if __name__ == "__main__":
    main()
