"""The paper's information-localization runs of the built models on Tahoe-100M: settings, parents and the run entry.

The map is the full grid of cuts, starting points and learning rates on which information localization retrains a
parent (``scsailor.localization.starts``). A run is one row (JSON) of it:

- ``name``: a label recorded with the run; ``out``: the run's output folder;
- ``stage``: ``map`` (40,000 updates), ``pilot`` (a feasibility run: 2,000 updates, read on the training panel) or
  ``time`` (a separate trajectory: the whole parent retrained for 120,000 updates along its own schedule);
- ``model`` (``summary``, ``deepsets`` or ``attention_model``), ``fold`` (``F1`` to ``F3``), ``seed``;
- ``cut`` (``scsailor.localization.contract.CUTS``), ``arm`` (the starting point: ``starts.arm_parts`` and the term
  table of ``scsailor.localization``), ``rate_multiple`` (1, 3, 10 or 30 times the parent's own rate at 40,000
  updates);
- ``half``: ``synthetic`` or ``natural``, the parent was trained on the synthetic task or on measured responses;
  ``case``: ``parent``, or a planted fault's case (``planted_faults``);
- ``parent``: the parent's checkpoint; ``inputs``: the fold's synthetic-answer instance.

``run`` loads the row's task and parent, checks that they belong together and hands them to ``starts.run``, which
writes the run's files into ``out``.

Settings:

- the parents' own learning rate (``parent_schedule``): the assay's recipe (``reproduce.assay.study``), a warm-up
  over 1,000 updates to the model's peak, then a cosine to a tenth of the peak at 160,000 updates; every parent
  stopped at 40,000 updates. The parents trained on measured responses used the same schedule
  (``scsailor.natural.data.lr_at``) and name the instance ``natural-<fold>-v1.1``;
- the training stream ``STREAM_ROOT`` + seed and the re-initialization seed ``RESET_ROOT`` + seed;
- the panels: the training panel (32 flip pairs on the first eight training contexts) and the held-out panel (56
  pairs, two per held-out context) and, except in a feasibility run, the validation panel (two pairs per training
  context), on which the restoring change is chosen.

    python -m reproduce.information_localization.tahoe --row ROW.json [--device cuda]
"""
from __future__ import annotations

import functools
import json
from pathlib import Path

from scsailor.assay.instance import HELDOUT
from scsailor.assay.train import learning_rate
from scsailor.localization import starts
from scsailor.localization.constants import LEVEL
from scsailor.localization.recovery import PanelSpec
from scsailor.localization.runtime import set_precision
from scsailor.localization.task import Task

from ..assay.study import PEAK_LR, SAVED_NAMES, SCHEDULE

__all__ = ["parent_schedule", "natural_instance", "STREAM_ROOT", "RESET_ROOT", "PANEL_ROOTS", "VALIDATION_ROOT",
           "PANEL_NAME", "panels", "evaluation_panels", "network", "load_parent", "check_synthetic",
           "fresh_parameters", "run", "main"]

STREAM_ROOT, RESET_ROOT = 2170000, 2180000
PANEL_ROOTS = {"seen": 1071101, "heldout": 1071102}   # the training and the held-out panel
VALIDATION_ROOT = 1151000
PANEL_NAME = "origin_parent_bridge_v1"   # the panels' label, saved in every evaluation


def parent_schedule(name, step):
    """The parents' own learning rate at ``step``."""
    return learning_rate(step, PEAK_LR[name], SCHEDULE)


def natural_instance(fold):
    """The instance named by a fold's parents trained on measured responses."""
    return f"natural-{fold}-v1.1"


def panels(task, stage):
    """The training and held-out panels and, except in a feasibility run (training panel only), the validation
    panel."""
    seen = PanelSpec(PANEL_ROOTS["seen"], "train", 32, task.train_contexts[:8])
    if stage == "pilot":
        return {"seen": seen}
    contexts = sorted(task.train_contexts)
    return {"seen": seen, "heldout": PanelSpec(PANEL_ROOTS["heldout"], HELDOUT, 56, task.heldout_contexts),
            "validation": PanelSpec(VALIDATION_ROOT, "train", 2 * len(contexts), contexts)}


def evaluation_panels(task):
    """The training and held-out panels, without the validation panel."""
    return {support: spec for support, spec in panels(task, "map").items() if support != "validation"}


def network(name, weights):
    """A built model ``name`` with the sizes and fixed buffers of the saved ``weights`` (``scsailor.models``)."""
    from scsailor.models import RawModel

    return RawModel(name, weights["backbone.intercept"].cpu(), weights["backbone.pert.weight"].shape[0],
                    weights["r_pool"].cpu(), weights["c_train"].cpu())


def load_parent(checkpoint, name, seed, device="cpu", step=starts.PARENT_STEP):
    """A built model's saved checkpoint at +Var after ``step`` updates (a parent: 40,000) and its record (instance,
    parameters, model name, seed, level and step, as saved), after checking its model, seed, level and step.

    The saved checkpoints name the attention model ``state`` (``reproduce.assay.study.SAVED_NAMES``)."""
    import torch

    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    found = {"model_name": SAVED_NAMES.get(saved.get("model_name"), saved.get("model_name")), "seed": saved.get("seed"),
             "level": saved.get("level"), "step": saved.get("step")}
    expected = {"model_name": name, "seed": seed, "level": LEVEL, "step": step}
    if found != expected:
        raise ValueError(f"{checkpoint} is not the checkpoint {expected}: {found}")
    model = network(name, saved["model"])
    model.load_state_dict(saved["model"], strict=True)
    metadata = {key: saved[key] for key in ("instance", "parameters", "model_name", "seed", "level", "step") if key in saved}
    return model.to(device), metadata


def check_synthetic(metadata, task, checkpoint):
    """Raise unless the checkpoint record ``metadata`` names the instance and parameters of ``task``."""
    expected = {"instance": task.record["instance"], "parameters": task.record["parameters"]}
    if {key: metadata.get(key) for key in expected} != expected:
        raise ValueError(f"{checkpoint} was not trained on the synthetic task {expected}")


def fresh_parameters(model, name, seed):
    """Parameters of a fresh network with the parents' initialization, built under ``seed`` (on the CPU)."""
    import torch

    torch.manual_seed(seed)
    return dict(network(name, model.state_dict()).named_parameters())


def run(row, device="cuda", segment_seconds=0.):
    """One map, feasibility or separate-trajectory run of ``row``: the synthetic task's planted responses learned from
    a parent trained on the synthetic task or on measured responses. ``segment_seconds`` as for
    ``starts.RunCheckpoint``."""
    name, half, cut, arm, stage = row["model"], row["half"], row["cut"], row["arm"], row["stage"]
    seed = int(row["seed"])
    set_precision(threads=2, interop_threads=1)
    task = Task(row["inputs"])
    model, metadata = load_parent(row["parent"], name, seed, device)
    if not task.record["instance"].startswith(f"semi-{row['fold']}-"):   # the saved instances' names
        raise ValueError(f"{row['inputs']} holds {task.record['instance']}, not fold {row['fold']}")
    if half == "synthetic":
        check_synthetic(metadata, task, row["parent"])
    elif metadata.get("instance") != natural_instance(row["fold"]):
        raise ValueError(f"{row['parent']} is not a {row['fold']} parent trained on measured responses "
                         f"({metadata.get('instance')})")
    starts.arm_parts(name, cut, arm)   # an invalid arm fails before any output is written
    if stage == "time" and (half, cut, arm) != ("natural", "input", "a"):
        raise ValueError("A separate trajectory retrains the whole parent trained on measured responses "
                         "(cut 'input', arm 'a')")
    starts.run(row, task, model, metadata, panels(task, stage), PANEL_NAME,
               fresh=fresh_parameters(model, name, RESET_ROOT + seed), reset_seed=RESET_ROOT + seed,
               stream_root=STREAM_ROOT + seed, parent_schedule=functools.partial(parent_schedule, name), device=device,
               segment_seconds=segment_seconds)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="One Tahoe-100M run of information localization: map, feasibility "
                                                 "run or separate trajectory (the row's stage).")
    parser.add_argument("--row", type=Path, required=True, help="the run's row (JSON)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment-seconds", type=float, default=0.,
                        help="pause after a checkpoint once this much time has passed (0: never); a rerun resumes")
    args = parser.parse_args(argv)
    run(json.loads(args.row.read_text()), args.device, args.segment_seconds)


if __name__ == "__main__":
    main()
