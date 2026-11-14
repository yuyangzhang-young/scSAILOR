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

from scsailor.localization import presence, readings, training_time

from ..assay.study import SAVED_NAMES, SEEDS

__all__ = ["READINGS", "MAPS", "TIME_AXIS", "PRESENCE", "map_tables", "score_name", "profile_case", "time_axis",
           "presence_tables", "main"]

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
    training_time.write_time_axis(out / TIME_AXIS / "TIME_AXIS.csv", time_axis(root / TIME_AXIS / "checkpoint-scores"))
    paths.append(Path(TIME_AXIS) / "TIME_AXIS.csv")
    (out / PRESENCE).mkdir(parents=True, exist_ok=True)
    for support, rows in presence_tables(root / PRESENCE / "profiles").items():
        presence.write_table(out / PRESENCE / f"PRESENCE_TABLE_{support}.csv", rows)
        paths.append(Path(PRESENCE) / f"PRESENCE_TABLE_{support}.csv")
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
