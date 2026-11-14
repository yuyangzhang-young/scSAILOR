"""Cue profiles and training longer on Tahoe-100M fold 1: which parents and saved checkpoints they read, on which
supports and panels, and the entries that write their files.

**Cue profiles** (``presence``): how strongly each planted state still moves each cut of a parent, without training
(``scsailor.localization.presence``). The parents are the fold-1 parents of the three built models (seeds 31002 to
31004), trained on the synthetic task or on measured responses, and the planted-fault parents. Two supports of the
synthetic task are profiled (``PRESENCE_SUPPORTS``): ``seen``, eight flip pairs on each of the first eight training
contexts, and ``heldout``, four pairs on each held-out context; the paper's cue ratios are read on the held-out
support. The saved field ``decision_support`` keeps its saved wording. Bits (``PRESENCE_BITS``): for the shift cue
(bits 0, 1, 3, 4) and the spread cue (5, 6, 8, 9), the shared, selective, weak selective and null assignments; the
selective ones (``PRODUCT_BITS``) also get the readings weighted by the perturbation code. Profiles run on the CPU.

**Training longer** (``time_axis_result``): a parent's saved checkpoints from 40,000 updates on, along the same
schedule (``TIME_AXIS_STEPS``), each scored on the training and held-out panels (``tahoe.evaluation_panels``). The
summary-statistics parents (seeds 31002-31004) and the attention-model parents of seeds 31002 and 31003 saved their
full training state every 20,000 updates; the third attention-model seed saved its weights every 40,000. The saved
time-axis scores also hold the DeepSets model's and State's checkpoints, which these entries do not score; the
later-checkpoint cue profiles are not made here either.

A parent or checkpoint is loaded after checking that it is the expected model, seed, level and step and, for the
synthetic task, that it belongs to the fold's instance. A finished file is not recomputed:

    python -m reproduce.information_localization.time_and_presence presence --instance DATA/tahoe-100m/F1/synthetic \\
        --half synthetic --model summary --seed 31002 --parent PT --out RESULTS
    python -m reproduce.information_localization.time_and_presence time-axis --instance DATA/tahoe-100m/F1/synthetic \\
        --model summary --seed 31002 --step 60000 --checkpoint PT --out RESULTS

The tables of both are written by ``reproduce.information_localization.tables``.
"""
import argparse
import json
import time
from pathlib import Path

from scsailor.assay.instance import HELDOUT
from scsailor.localization import presence as presence_profile
from scsailor.localization import training_time
from scsailor.localization.contract import PRESENCE_SITES
from scsailor.localization.runtime import set_precision
from scsailor.localization.task import Task

from ..assay.study import SEEDS
from . import tahoe

__all__ = ["PRESENCE_BITS", "PRODUCT_BITS", "PRESENCE_SUPPORTS", "TIME_AXIS_STEPS", "presence", "time_axis_result",
           "main"]

PRESENCE_BITS = (0, 1, 3, 4, 5, 6, 8, 9)
PRODUCT_BITS = (1, 3, 6, 8)
PRESENCE_SUPPORTS = {"seen": (2010171, 8), "heldout": (2010172, 4)}   # root and flip pairs per context

# (model, seed) -> the saved steps the time axis scores
TIME_AXIS_STEPS = {**{(model, seed): (40000, 60000, 80000, 100000, 120000, 140000)
                      for model, seed in (("summary", 31002), ("summary", 31003), ("summary", 31004),
                                          ("attention_model", 31002), ("attention_model", 31003))},
                   ("attention_model", 31004): (40000, 80000, 120000, 160000)}


def presence(task, half, model, seed, parent, *, threads=4):
    """The cue profile of one fold-1 parent on both supports (with ``seconds``, the profile's wall time)."""
    set_precision(threads=threads)
    network, metadata = tahoe.load_parent(parent, model, seed)
    if half == "synthetic":
        tahoe.check_synthetic(metadata, task, parent)
    network.eval()
    started = time.monotonic()
    result = {"half": half, "model": model, "seed": seed, "parent": str(parent), "bits": list(PRESENCE_BITS),
              "drug_factor": presence_profile.drug_factor(network, task),
              "decision_support": "seen (training contexts); heldout is reported only", "supports": {}}
    for support, (root, per_context) in PRESENCE_SUPPORTS.items():
        contexts = task.train_contexts[:8] if support == "seen" else task.heldout_contexts
        group = "train" if support == "seen" else HELDOUT
        result["supports"][support] = {"root": root, "pairs_per_context": per_context, "contexts": list(contexts),
                                       **presence_profile.profile(network, PRESENCE_SITES[model], task,
                                                                  contexts, group, root, per_context,
                                                                  list(PRESENCE_BITS), PRODUCT_BITS)}
    result["seconds"] = time.monotonic() - started
    return result


def time_axis_result(task, name, seed, step, checkpoint, *, threads=4, device="cpu"):
    """One saved checkpoint of a synthetic parent's own training, scored on the training and held-out panels."""
    set_precision(threads=threads)
    model, metadata = tahoe.load_parent(checkpoint, name, seed, device, step=step)
    tahoe.check_synthetic(metadata, task, checkpoint)
    result = training_time.score(model.eval(), task, tahoe.evaluation_panels(task), tahoe.PANEL_NAME, device)
    result["checkpoint"] = str(checkpoint)
    result["step"] = step
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cue profiles and training longer on Tahoe-100M fold 1.")
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("presence", help="profile one parent: PRESENCE_<half>_<model>_<seed><tag>.json")
    profile.add_argument("--half", choices=("synthetic", "natural"), required=True)
    profile.add_argument("--model", choices=tuple(PRESENCE_SITES), required=True)
    profile.add_argument("--parent", type=Path, required=True, help="the parent's checkpoint")
    profile.add_argument("--tag", default="", help="appended to the file name: _<case> for a planted-fault parent")
    axis = commands.add_parser("time-axis", help="score one saved checkpoint: EVAL_<model>_<seed>_<step>.json")
    axis.add_argument("--model", choices=("summary", "attention_model"), required=True)
    axis.add_argument("--step", type=int, required=True)
    axis.add_argument("--checkpoint", type=Path, required=True)
    axis.add_argument("--device", default="cpu")
    for command in (profile, axis):
        command.add_argument("--seed", type=int, choices=SEEDS, required=True)
        command.add_argument("--instance", type=Path, required=True, help="Tahoe-100M fold 1's synthetic-answer instance")
        command.add_argument("--threads", type=int, default=4)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "presence":
        target = args.out / f"PRESENCE_{args.half}_{args.model}_{args.seed}{args.tag}.json"
        if not target.exists():
            result = presence(Task(args.instance), args.half, args.model, args.seed, args.parent, threads=args.threads)
            args.out.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result) + "\n")
    else:
        target = args.out / f"EVAL_{args.model}_{args.seed}_{args.step}.json"
        if not target.exists():
            result = time_axis_result(Task(args.instance), args.model, args.seed, args.step, args.checkpoint,
                                      threads=args.threads, device=args.device)
            args.out.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
