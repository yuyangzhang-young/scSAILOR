"""The training loop of the synthetic-answer assay.

Each update takes the stream's next batch of eight bags, sets the learning rate
(linear warm-up over 1,000 updates from 1% of the peak, then a cosine decay to
10% of the peak at the end of the schedule), computes the MSE between the raw
output and the known answer on genes 50 onward, clips the gradient norm at 1
and takes one AdamW step (no weight decay). Batches are drawn ahead by worker
threads; their content does not depend on that (see ``stream``).

Outputs in ``output``: ``RUN_RECORD.json``, ``TRAIN_CURVE.jsonl`` (every 100th
update: loss, gradient norm before clipping, learning rate, time) and ``step{N}.pt``
checkpoints at the requested steps. torch is imported only by the functions that use it.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import FIRST_SCORED_GENE, N_CELLS

WARMUP = 1000

__all__ = ["Fit", "learning_rate", "train", "save_checkpoint"]


@dataclass(frozen=True)
class Fit:
    """One training run."""
    model: str              # "state" (the attention network), "deepsets" or "summary"
    seed: int               # model initialization and training stream
    level: str              # information level of the inputs
    peak_lr: float
    steps: int              # updates to run
    total: int              # last update of the cosine schedule
    save_at: tuple = ()     # updates after which a checkpoint is written


def learning_rate(step, peak, total):
    if step <= WARMUP:
        return peak * (.01 + .99 * (step - 1) / (WARMUP - 1))
    return peak * (.1 + .45 * (1 + math.cos(math.pi * (step - WARMUP) / (total - WARMUP))))


def batches(stream, steps, level, seed, workers):
    """Batches 1..steps in order, drawn ahead by ``workers`` threads; yields (step, batch, seconds waited)."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {step: executor.submit(stream.batch, step, N_CELLS, level, seed)
                   for step in range(1, min(steps, 2 * workers) + 1)}
        for step in range(1, steps + 1):
            before = time.perf_counter()
            batch = pending.pop(step).result()
            waited = time.perf_counter() - before
            if step + 2 * workers <= steps:
                pending[step + 2 * workers] = executor.submit(stream.batch, step + 2 * workers, N_CELLS, level, seed)
            yield step, batch, waited


def save_checkpoint(path, model, fit, step, instance):
    """Weights and the identity a checkpoint needs to be replayed or probed (no optimizer state)."""
    import torch

    torch.save({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "step": step,
                "model_name": fit.model, "level": fit.level, "seed": fit.seed,
                "instance": instance.name, "parameters": instance.record["parameters"]}, path)


def train(model, stream, fit, instance, output, device, workers=4):
    """Train ``model`` in place for ``fit.steps`` updates; returns the curve rows."""
    import torch
    from torch.nn import functional as F

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "RUN_RECORD.json").write_text(json.dumps(
        {"fit": asdict(fit), "instance": instance.name, "parameters": instance.record["parameters"],
         "torch": str(torch.__version__), "device": str(device),
         "matmul_precision": torch.get_float32_matmul_precision(),
         "tf32": [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32]}, indent=2) + "\n")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=fit.peak_lr, betas=(.9, .999), eps=1e-8,
                                  weight_decay=0, fused=False)
    curve = []
    started = time.monotonic()
    with (output / "TRAIN_CURVE.jsonl").open("x") as log:
        for step, (x, depth, p, y), waited in batches(stream, fit.steps, fit.level, fit.seed, workers):
            x, depth, p, y = (torch.from_numpy(v).to(device) for v in (x, depth, p, y))
            lr = learning_rate(step, fit.peak_lr, fit.total)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x, depth, p, fit.level)[:, FIRST_SCORED_GENE:], y[:, FIRST_SCORED_GENE:])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at update {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step in fit.save_at:
                save_checkpoint(output / f"step{step}.pt", model, fit, step, instance)
            if step % 100 == 0:
                row = {"step": step, "loss": float(loss), "gradient_norm": float(norm), "learning_rate": lr,
                       "elapsed_s": time.monotonic() - started, "input_wait_s": waited}
                curve.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
    return curve
