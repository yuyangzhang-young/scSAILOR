"""The handed-on change on the real task: one map cell applied unchanged to a parent trained on real responses.

A cell (model, cut, arm, rate) of the map is trained on real labels instead of the synthetic targets:

- the arm is built as in the map (``starts.build_arm``): the same resets and the same trained copy of the drug
  embedding where a frozen part also reads it; the copy becomes the model's submodule
  ``drug_embedding_copy``, so that it is saved and reloaded with the weights;
- the schedule is the map's (``starts.learning_rate``): a 1,000-update warm-up and a cosine to a tenth at 40,000
  updates, at the cell's multiple of the parent's rate at 40k, with 8 bags per update, and the map's update
  (``contract.training_update``);
- the data are the parent's own training stream continued: the first ``n_lines`` lines of the fold's fixed
  order (fewer for a nested parent, which saw only those), native batches from ``FIRST_NATIVE_INDEX`` on, stream
  root = seed, and the natural parent's own loss (MSE on all genes). No held-out expression value is read.

The caller passes the loaded stream (its ``batch`` returns the four arrays and the context codes, which are
dropped) and ``predict(model, path)``, which writes the prediction grid of every training and held-out context
in origin resolution's format and returns its receipt. Outputs in ``row["out"]``:

- ``predictions-parent.npz`` (the parent, before any change) and ``predictions-step40000.npz`` (the end, from
  the saved end weights). The arm's start is not predicted: it equals the parent for the arms that keep the
  function, and a re-initialized start is not one of origin resolution's quantities;
- ``step00000.pt`` and ``step40000.pt``: the weights;
- ``RUN_RECORD.json``, ``TRAIN_CURVE.jsonl``, ``EXECUTION.json``, ``COMPLETE.json`` (or ``DIVERGED.json``).

The run checkpoints and resumes as a map run does (``starts.RunCheckpoint``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .constants import ADAMW, BAGS_PER_NATIVE_BATCH, LEVEL
from .draws import training_batch
from .runtime import peak_gpu_mib
from .starts import (BATCH, PARENT_STEP, SCHEDULE, WARMUP, RunCheckpoint, build_arm, learning_rate, train,
                     write_execution, write_json)

__all__ = ["FIRST_NATIVE_INDEX", "run"]

FIRST_NATIVE_INDEX = 40001   # the parents trained on native batches 1-40,000 of their stream


class _Continued:
    """The natural stream from native index ``offset + 1`` on, without its appended context codes."""

    def __init__(self, stream, offset):
        self.stream, self.offset = stream, offset

    def batch(self, index, *args):
        return self.stream.batch(self.offset + index, *args)[:4]


def _weights(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def run(row, model, metadata, *, lines, menu_access, stream, predict, fresh, reset_seed, parent_schedule,
        device="cuda", updates=None, segment_seconds=0.):
    """Train ``row``'s cell (``model``, ``cut``, ``arm``, ``seed``, ``rate_multiple``, ``n_lines``, ``out``; the
    whole row is recorded) on the real task from the loaded natural parent ``model``.

    ``lines`` are the lines the stream draws from, ``menu_access`` the record of what the prediction menus read;
    ``fresh``, ``reset_seed``, ``parent_schedule`` and ``segment_seconds`` as for ``starts.run``. ``updates`` stops
    early with the schedule unchanged (checks only).
    """
    import torch

    name, cut, arm = row["model"], row["cut"], row["arm"]
    seed, multiple = int(row["seed"]), float(row["rate_multiple"])
    updates = SCHEDULE if updates is None else updates
    out = Path(row["out"])
    checkpoint = RunCheckpoint(out, row, updates, segment_seconds)
    per_update = BATCH // BAGS_PER_NATIVE_BATCH
    parent_rate = parent_schedule(PARENT_STEP)
    peak = multiple * parent_rate
    if checkpoint.resuming:
        record = json.loads((out / "RUN_RECORD.json").read_text())
    else:
        record = {"status": "training", "setting": "natural task", "row": row, "parent_metadata": metadata,
                  "updates": updates, "schedule_length": SCHEDULE, "bags_per_update": BATCH,
                  "n_lines": int(row["n_lines"]), "lines": lines,
                  "native_indices": [FIRST_NATIVE_INDEX, FIRST_NATIVE_INDEX - 1 + per_update * updates],
                  "stream_root": seed, "loss": "MSE on all genes", "level": LEVEL, "peak_lr": peak,
                  "parent_rate_at_40k": parent_rate, "rate_multiple": multiple, "heldout_truth_loaded": False,
                  "menu_access": menu_access}
    save = lambda: write_json(out / "RUN_RECORD.json", record)  # noqa: E731

    def grid(path):
        receipt = predict(model, out / path)
        record.setdefault("predictions", []).append({"file": path, "model_calls": receipt["model_calls"]})
        save()

    if not checkpoint.resuming:
        grid("predictions-parent.npz")
    parameters, copy, fields = build_arm(model, name, cut, arm, fresh, reset_seed, device)
    if copy is not None:
        model.drug_embedding_copy = copy   # a submodule, so that it is saved and reloaded with the weights
    record.update(fields)
    record["learning_rate_at"] = {str(u): learning_rate(peak, u) for u in sorted({1, WARMUP, WARMUP + 1, updates})}
    if not checkpoint.resuming:
        save()
        torch.save({"model": _weights(model), "step": 0}, out / "step00000.pt")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate(peak, 1), **ADAMW)
    continued = _Continued(stream, FIRST_NATIVE_INDEX - 1)
    model.train()
    started = train(model, optimizer, parameters, lambda u: training_batch(continued, u, seed, BATCH),
                    lambda u: learning_rate(peak, u), updates, 0, out, checkpoint, device)
    end = out / f"step{updates:05d}.pt"
    torch.save({"model": _weights(model), "step": updates}, end)
    record.update(status="predicting", fitting_seconds=time.monotonic() - started)
    save()
    # the end grid comes from the saved end weights
    model.load_state_dict(torch.load(end, map_location="cpu", weights_only=False)["model"], strict=True)
    grid(f"predictions-step{updates:05d}.npz")
    record.update(status="complete", elapsed_seconds=time.monotonic() - started, peak_gpu_MiB=peak_gpu_mib(device))
    save()
    write_execution(out, device)
    write_json(out / "COMPLETE.json", {"status": "complete", "row": row})
    checkpoint.close()
