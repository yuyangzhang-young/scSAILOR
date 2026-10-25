"""One starting point at one cut: train the part of a parent behind the cut on the known answer.

A run takes a parent (trained on the synthetic task, or on real data) and trains the part behind one cut
(``contract.CUTS``) on the synthetic task's labels, with the front frozen. The arms differ only in where
training starts:

- ``a``: the parent's front and the parent's weights behind the cut (fine-tune); at the cut ``input`` this
  trains the whole parent, its own continued training on these labels;
- ``f``: the parent's front, the part behind the cut re-initialized; it reads the parent's drug embedding;
- ``e``: the front re-initialized and frozen, the parent's weights behind the cut;
- ``e_mean`` / ``e_var`` (summary-statistics): as ``e``, re-initializing only the mean or only the variance
  projection;
- ``d``: as ``a``, with the drug embedding read behind the cut re-initialized;
- ``interaction`` / ``common`` (summary-statistics, cut ``projections``): only one branch is trained, with its
  own copy of the parent's drug embedding (the other branch keeps reading the original, frozen);
  ``interaction_e_mean`` is ``interaction`` with the mean projection re-initialized;
- ``fresh``: the whole model re-initialized and trained (cut ``input``).

Re-initialization follows the parents' own scheme: the caller builds a fresh network under a fixed seed
(``fresh``) and the chosen parameters are copied in; buffers stay the parent's. Where a frozen part also reads
the drug embedding (the attention token entry; the summary-statistics branch that is not trained), the trained
part gets its own copy (``contract.drug_embedding_copy``): the parent's in a, e, f and the single-branch arms,
a fresh one in d.

Training, the same in every arm: AdamW (0.9, 0.999, eps 1e-8, no weight decay), gradient clip 1, a linear
warm-up over 1,000 updates and a cosine to a tenth at 40,000 updates (``learning_rate``), at a multiple of the
parent's own rate at 40k, with 8 bags per update as the parents were trained. The stages:

- ``map``: those 40,000 updates (the parents' own budget), read every 4,000 updates and at the end;
- ``pilot`` (rate pilot): the first 2,000 of them, on the same schedule, read every 500 updates;
- ``time`` (time axis): the whole natural parent on these labels along the parent's own schedule from step
  40,000 to 160,000, after a fresh warm-up (``continued_rate``), read every 20,000 updates.

A reading is the recovery of bits 0, 1, 5 and 6 (target bit 1) on the caller's panels.

Outputs in ``row["out"]``: RUN_RECORD.json, EVAL_PARENT.json, EVAL_START.json, TRAIN_CURVE.jsonl (update 1 and
every 100th: loss, gradient norm before clipping, rate, seconds), EVAL_TRACE.jsonl, EVAL.json, FINAL.pt
(weights, optimizer state and drug-embedding copy; not for a pilot), EXECUTION.json and COMPLETE.json; or
DIVERGED.json at a nonfinite loss or gradient norm.

Every run checkpoints (``RunCheckpoint``): CHECKPOINT.pt every 1,000 updates, at the end and when a stop is
requested (SIGTERM, SIGUSR1, or, with ``segment_seconds``, that many seconds of training in this process). A stopped
run exits with status ``PAUSED``; the same call resumes it from its last checkpoint, and its outputs equal a straight
run's. torch is imported only by the functions that use it.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import platform
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..assay.constants import FIRST_SCORED_GENE
from .constants import ADAMW, BITS, CURVE_EVERY, TARGET_BIT
from .contract import (BRANCH_ARMS, CUTS, FRONT_PARTS, FRONT_READS_EMBEDDING, Nonfinite, drug_embedding_copy,
                       training_update)
from .draws import training_batch
from .recovery import evaluate

__all__ = ["ARMS", "WARMUP", "BATCH", "SCHEDULE", "PARENT_STEP", "UPDATES", "EVAL_EVERY", "BITS", "TARGET_BIT",
           "PAUSED", "learning_rate", "continued_rate", "arm_parts", "build_arm", "write_json", "write_execution",
           "RunCheckpoint", "train", "run"]

ARMS = ("a", "d", "e", "f", "e_mean", "e_var", "interaction", "common", "interaction_e_mean", "fresh")
WARMUP, BATCH, SCHEDULE = 1000, 8, 40000
PARENT_STEP = 40000         # every parent stopped at 40,000 updates of its own schedule
UPDATES = {"map": 40000, "pilot": 2000, "time": 120000}
EVAL_EVERY = {"map": 4000, "pilot": 500, "time": 20000}
WORKERS, AHEAD = 3, 6       # batch threads and batches drawn ahead; a batch does not depend on them
CHECKPOINT_EVERY = 1000
MAP_ROW = ("model", "half", "cut", "arm", "seed", "stage", "rate_multiple")   # the row fields FINAL.pt keeps
PAUSED = 75                 # exit status of a paused run


def learning_rate(peak, update, schedule=SCHEDULE):
    """Linear warm-up over 1,000 updates from a hundredth of the peak, then a cosine to a tenth at ``schedule``
    (the map's 40,000 updates unless given)."""
    if not 1 <= update <= schedule:
        raise ValueError(f"Update {update} is outside the {schedule}-update schedule (a continuation holds the floor rate)")
    if update <= WARMUP:
        return peak * (.01 + .99 * (update - 1) / (WARMUP - 1))
    return peak * (.1 + .45 * (1 + math.cos(math.pi * (update - WARMUP) / (schedule - WARMUP))))


def continued_rate(parent_schedule, update):
    """Time axis: the parent's own schedule (``parent_schedule(step)``) from step 40,000 to 160,000, after a fresh
    1,000-update warm-up."""
    if not 1 <= update <= UPDATES["time"]:
        raise ValueError(f"Update {update} is outside the time axis")
    rate = parent_schedule(PARENT_STEP + update)
    return rate * (.01 + .99 * (update - 1) / (WARMUP - 1)) if update <= WARMUP else rate


def matches(name, prefixes):
    return prefixes is None or any(name.startswith(prefix) for prefix in prefixes)


def select_trainable(model, prefixes):
    """Freeze every parameter except those matching ``prefixes``; return the trained ``(name, parameter)`` pairs."""
    trainable = [(name, p) for name, p in model.named_parameters() if matches(name, prefixes)]
    for name, p in model.named_parameters():
        p.requires_grad_(matches(name, prefixes))
    return trainable


def copy_in(model, fresh, prefixes):
    """Copy the fresh values of every parameter matching ``prefixes`` into ``model``; return their names."""
    import torch

    copied = []
    with torch.no_grad():
        for name, value in model.named_parameters():
            if matches(name, prefixes):
                value.copy_(fresh[name].to(value.device))
                copied.append(name)
    return copied


def arm_parts(name, cut, arm):
    """The frozen front, the trained prefixes and the summary-statistics branch of an arm; raises for an invalid arm."""
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; the arms are {', '.join(ARMS)}")
    front, trained = CUTS[name][cut]
    branch = BRANCH_ARMS.get(arm)
    if branch is not None:
        if (name, cut) != ("summary", "projections"):
            raise ValueError(f"Arm {arm} is a summary-statistics arm of the cut 'projections'")
        trained = branch[1]
    if arm == "fresh" and cut != "input":
        raise ValueError("The fresh network is an arm of the cut 'input'")
    return front, trained, branch


def build_arm(model, name, cut, arm, fresh, reset_seed, device):
    """Turn a loaded parent into an arm's starting point; return the trained parameters, the drug-embedding copy (or
    None) and the arm's record fields. ``fresh`` holds the parameters of a fresh network with the parents'
    initialization, built under ``reset_seed``. Shared by the map runs and the real task (``real_task``)."""
    front, trained, branch = arm_parts(name, cut, arm)
    parent_embedding = model.backbone.pert.weight.detach().clone()   # before any reset
    reset = []
    if arm == "fresh":
        reset += copy_in(model, fresh, trained)
    if arm == "f":   # the part behind the cut re-initialized, its drug embedding kept
        reset += copy_in(model, fresh, tuple(prefix for prefix in trained if prefix != "backbone.pert"))
    if arm == "e":
        reset += copy_in(model, fresh, front)
    if arm in FRONT_PARTS:
        reset += copy_in(model, fresh, FRONT_PARTS[arm])
    if branch is not None:
        copy_site = f"summary_{branch[0]}"
    else:
        copy_site = FRONT_READS_EMBEDDING.get(name) if cut != "input" else None
    if arm == "d" and copy_site is None:
        reset += copy_in(model, fresh, ("backbone.pert",))
    fields = {"front": front, "trained_prefixes": trained, "reset_seed": reset_seed, "reset_parameters": reset}
    trainable = select_trainable(model, trained)
    copy = None
    if copy_site is not None:
        own = arm in ("a", "e", "f") or branch is not None
        start = parent_embedding if own else fresh["backbone.pert.weight"].to(device)
        copy = drug_embedding_copy(model, start.to(device), copy_site).to(device)
        fields["drug_embedding_copy"] = {"site": copy_site, "start": "parent's" if own else "fresh"}
    parameters = [p for _, p in trainable] + ([] if copy is None else list(copy.parameters()))
    fields["trainable_parameters"] = int(sum(p.numel() for p in parameters))
    fields["trainable_names"] = [n for n, _ in trainable] + ([] if copy is None else ["drug_embedding_copy.weight"])
    return parameters, copy, fields


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_execution(out, device):
    """Where and how the run executed: host, interpreter, torch, device, threads and precision."""
    import torch

    write_json(Path(out) / "EXECUTION.json", {
        "host": platform.node(), "python": sys.executable, "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": device, "threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")})


class RunCheckpoint:
    """Resumable state of one run: one writer, the row's identity, restart at a completed optimizer update.

    CHECKPOINT.pt holds the weights, the optimizer, a separately trained module (a drug-embedding copy), the
    generator states and the committed sizes of the two logs; it is replaced only after a complete write. An
    output folder that exists without a checkpoint, or with DIVERGED.json, is refused. With ``segment_seconds``
    above zero, a stop is requested after that many seconds of training in this process.
    """

    def __init__(self, out, row, updates, segment_seconds=0.):
        import torch

        self.out, self.row, self.updates = Path(out), row, updates
        self.path = self.out / "CHECKPOINT.pt"
        self.resuming = self.out.exists()
        if (self.out / "DIVERGED.json").exists():
            raise RuntimeError(f"Numerical failure requires inspection: {self.out}")
        if self.resuming and not self.path.is_file():
            raise RuntimeError(f"Partial output has no resumable checkpoint; preserved: {self.out}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / ".run.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close()
            raise RuntimeError(f"Another writer owns {self.out}") from error
        self.saved = None
        if self.resuming:
            self.saved = torch.load(self.path, map_location="cpu", weights_only=False)
            if self.saved["row"] != row or self.saved["updates"] != updates:
                self.close()
                raise ValueError(f"Checkpoint identity does not match the run's row: {self.path}")
            if not 0 <= self.saved["update"] <= updates:
                self.close()
                raise ValueError(f"Invalid checkpoint update: {self.path}")
        self.stop_requested = False
        self.began = time.monotonic()
        self.limit = float(segment_seconds)
        self.handlers = {}
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            self.handlers[sig] = signal.signal(sig, self.request_stop)

    def request_stop(self, signum, frame):
        self.stop_requested = True

    def stopping(self):
        return self.stop_requested or (self.limit > 0 and time.monotonic() - self.began >= self.limit)

    def due(self, update, updates):
        return update % CHECKPOINT_EVERY == 0 or update == updates or self.stopping()

    def restore(self, model, optimizer, extra_model=None):
        """Load the saved state and cut the logs back to its committed sizes; return (update, elapsed seconds)."""
        import numpy as np
        import torch

        if self.saved is None:
            return 0, 0.
        saved = self.saved
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        if extra_model is not None:
            extra_model.load_state_dict(saved["extra_model"], strict=True)
        for name, size in saved["log_sizes"].items():
            path = self.out / name
            if size and (not path.is_file() or path.stat().st_size < size):
                raise RuntimeError(f"Checkpoint log is missing or shorter than its committed boundary: {path}")
            with path.open("a+b") as handle:
                handle.truncate(size)
        random.setstate(saved["random"])
        np.random.set_state(saved["numpy"])
        torch.set_rng_state(saved["torch"])
        if saved["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda"])
        print(f"RESUMED {self.row['name']} update={saved['update']}", flush=True)
        self.saved = None
        return saved["update"], saved["elapsed_s"]

    def save(self, model, optimizer, update, elapsed_s, extra_model=None):
        """Write the checkpoint (replacing the previous one only after a complete write); return whether to stop."""
        import numpy as np
        import torch

        logs = {p.name: p.stat().st_size if p.is_file() else 0
                for p in (self.out / "TRAIN_CURVE.jsonl", self.out / "EVAL_TRACE.jsonl")}
        saved = dict(row=self.row, updates=self.updates, update=update, elapsed_s=elapsed_s,
                     model=model.state_dict(), optimizer=optimizer.state_dict(),
                     extra_model=None if extra_model is None else extra_model.state_dict(),
                     random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, log_sizes=logs)
        temporary = self.path.with_suffix(".pt.tmp")
        torch.save(saved, temporary)
        os.replace(temporary, self.path)
        receipt = dict(status="paused" if self.stopping() else "checkpoint", row=self.row["name"],
                       update=update, updates=self.updates, elapsed_s=elapsed_s)
        receipt_path = self.out / "CHECKPOINT.json"
        receipt_tmp = receipt_path.with_suffix(".json.tmp")
        receipt_tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        os.replace(receipt_tmp, receipt_path)
        return self.stopping()

    def close(self):
        for sig, handler in getattr(self, "handlers", {}).items():
            signal.signal(sig, handler)
        self.lock.close()


def train(model, optimizer, parameters, batch, rate, updates, first_gene, out, checkpoint, device, copy=None,
          every=None, score=None):
    """The training loop of every start, resumed from ``checkpoint`` and paused at its request.

    ``batch(update)`` gives the numpy arrays (x, depth, p, y) of one update, drawn ahead by worker threads;
    ``rate(update)`` its learning rate; the loss reads genes ``first_gene`` onward. A curve row at update 1 and
    every ``CURVE_EVERY``-th update; with ``every``, ``score(model)`` is appended to EVAL_TRACE.jsonl every ``every``
    updates before the last. A nonfinite loss or gradient norm writes DIVERGED.json and raises. A checkpoint
    when ``checkpoint.due``; after a requested stop's checkpoint the process exits with ``PAUSED``. Returns the
    monotonic time at which the run started, counting earlier segments.
    """
    import torch

    completed, elapsed = checkpoint.restore(model, optimizer, copy)
    if not checkpoint.resuming:
        checkpoint.save(model, optimizer, 0, 0., copy)
    started = time.monotonic() - elapsed
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, \
            (out / "TRAIN_CURVE.jsonl").open("a" if checkpoint.resuming else "x") as curve:
        pending = {u: pool.submit(batch, u) for u in range(completed + 1, min(updates, completed + AHEAD) + 1)}
        for update in range(completed + 1, updates + 1):
            if update + AHEAD <= updates:
                pending[update + AHEAD] = pool.submit(batch, update + AHEAD)
            arrays = tuple(torch.from_numpy(v).to(device) for v in pending.pop(update).result())
            lr = rate(update)
            try:
                loss, grad_norm = training_update(model, optimizer, parameters, arrays, lr, first_gene)
            except Nonfinite as error:
                write_json(out / "DIVERGED.json", {"update": update, "what": error.what, "value": str(error.value)})
                raise RuntimeError(f"Nonfinite {error.what} at update {update}") from None
            if update == 1 or update % CURVE_EVERY == 0:
                curve.write(json.dumps({"update": update, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                                        "lr": lr, "elapsed_s": time.monotonic() - started}) + "\n")
                curve.flush()
            if every is not None and update % every == 0 and update < updates:
                with (out / "EVAL_TRACE.jsonl").open("a") as trace:
                    trace.write(json.dumps({"update": update, "eval": score(model)}) + "\n")
            if checkpoint.due(update, updates):
                curve.flush()
                if checkpoint.save(model, optimizer, update, time.monotonic() - started, copy):
                    checkpoint.close()
                    raise SystemExit(PAUSED)
    return started


def run(row, task, model, metadata, panels, panel_name, *, fresh, reset_seed, stream_root, parent_schedule,
        device="cuda", segment_seconds=0.):
    """One map, rate-pilot or time-axis run of ``row`` (``model``, ``cut``, ``arm``, ``stage``, ``rate_multiple``,
    ``out``; the whole row is recorded) from the loaded parent ``model`` and its checkpoint ``metadata``.

    ``task`` (``task.Task``) gives the training stream (``task.stream()``, read from ``stream_root``) and the panels'
    pools; ``panels`` maps a support to its ``recovery.PanelSpec``, read under ``panel_name``; ``fresh`` and
    ``reset_seed`` are the re-initialization (``build_arm``); ``parent_schedule(step)`` is the parent's own learning
    rate; ``segment_seconds`` as for ``RunCheckpoint``.
    """
    import torch

    name, cut, arm, stage = row["model"], row["cut"], row["arm"], row["stage"]
    multiple = float(row["rate_multiple"])
    updates, every = UPDATES[stage], EVAL_EVERY[stage]
    out = Path(row["out"])
    checkpoint = RunCheckpoint(out, row, updates, segment_seconds)

    def score(current):
        return evaluate(current, task, panels, BITS, TARGET_BIT, device, panel_name=panel_name)

    if not checkpoint.resuming:
        write_json(out / "EVAL_PARENT.json", score(model))
    parent_rate = parent_schedule(PARENT_STEP)
    record = {"row": row, "updates": updates, "schedule_length": SCHEDULE, "bags_per_update": BATCH,
              "stream_root": stream_root, "eval_every": every, "panels": list(panels), "bits": list(BITS),
              "target_bit": TARGET_BIT, "peak_lr": multiple * parent_rate, "parent_rate_at_40k": parent_rate,
              "rate_multiple": multiple}
    rate = ((lambda u: continued_rate(parent_schedule, u)) if stage == "time"
            else (lambda u: learning_rate(record["peak_lr"], u)))
    parameters, copy, fields = build_arm(model, name, cut, arm, fresh, reset_seed, device)
    record.update(fields)
    record["learning_rate_at"] = {str(u): rate(u) for u in (1, WARMUP, WARMUP + 1, updates // 2, updates)}
    if not checkpoint.resuming:
        write_json(out / "RUN_RECORD.json", record)
        write_json(out / "EVAL_START.json", score(model))
    optimizer = torch.optim.AdamW(parameters, lr=rate(1), **ADAMW)
    stream = task.stream()
    started = train(model, optimizer, parameters, lambda u: training_batch(stream, u, stream_root, BATCH), rate,
                    updates, FIRST_SCORED_GENE, out, checkpoint, device, copy=copy, every=every, score=score)
    final = score(model)
    with (out / "EVAL_TRACE.jsonl").open("a") as trace:
        trace.write(json.dumps({"update": updates, "eval": final}) + "\n")
    if stage != "pilot":   # weights and optimizer state, so that a still-rising run can be continued
        saved = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 **{k: metadata[k] for k in ("instance", "parameters", "model_name", "seed", "level") if k in metadata},
                 "map_row": {key: row[key] for key in MAP_ROW},
                 "optimizer": optimizer.state_dict(), "updates_done": updates}
        if copy is not None:
            saved["drug_embedding_copy"] = {"site": fields["drug_embedding_copy"]["site"],
                                            "weight": copy.embedding.weight.detach().cpu()}
        torch.save(saved, out / "FINAL.pt")
    write_json(out / "EVAL.json", {"after": final, "train_seconds": time.monotonic() - started})
    write_execution(out, device)
    write_json(out / "COMPLETE.json", {"status": "complete", "row": row})
    checkpoint.close()
