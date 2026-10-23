"""The training loop shared by the three published models and both tasks."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

SAVE_EVERY = 1000   # updates between resume points


def _random_states(cuda):
    return dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if cuda else [])


def _restore_random_states(state):
    torch.set_rng_state(state["torch_rng"].cpu())
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def fit(adapter, stream, *, steps, output, device, workers=4):
    """Train for ``steps`` updates and return the model.

    ``adapter`` builds the seeded model and its optimizers and performs one update; ``stream`` yields the
    batch of each update in order. A run that stops early resumes from ``latest.pt`` with every random
    state restored, so its result equals an uninterrupted run. Writes ``TRAIN_CURVE.jsonl`` (every 100th
    update), ``latest.pt`` (every ``SAVE_EVERY`` updates) and ``step{steps}.pt``, the model that is read out.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cuda = device.startswith("cuda")
    model = adapter.build().train()
    optimizers = adapter.optimizers(model)
    latest = output / "latest.pt"
    first = 1
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        stream.load_state_dict(state["stream"])
        _restore_random_states(state)
        first = state["step"] + 1
    begin = time.perf_counter()
    with (output / "TRAIN_CURVE.jsonl").open("a", buffering=1) as log:
        for step, batch in stream.batches(first, steps, workers):
            loss = float(adapter.update(model, optimizers, batch, step))
            if not np.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at update {step}")
            if step % 100 == 0 or step == steps:
                log.write(json.dumps(dict(step=step, loss=loss, seconds=time.perf_counter() - begin)) + "\n")
            if step % SAVE_EVERY == 0 or step == steps:
                _save(latest, dict(model=model.state_dict(), step=step, stream=stream.state_dict(),
                                   optimizers={name: optimizer.state_dict() for name, optimizer in optimizers.items()},
                                   **_random_states(cuda)))
    _save(output / f"step{steps}.pt", dict(model=model.state_dict(), step=steps))
    return model
