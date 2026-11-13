"""The paper's settings of the synthetic-answer assay, and its binding to the built models.

Where each dataset fold's instance lies under the data root, the training recipe, the technical draws and the replay
scopes of the paper's runs on Tahoe-100M and Parse PBMC, and the networks of ``scsailor.models``. The modules of
``scsailor.assay`` take all of these as arguments.

The study's saved checkpoints name the attention model ``state``; ``load_checkpoint`` reads them under its public name
``attention_model``. Importing this module does not import torch.
"""
from __future__ import annotations

from pathlib import Path

from scsailor.assay.instance import HELDOUT, SEEN, Instance
from scsailor.assay.train import Fit

__all__ = ["DATASETS", "load_instance", "ENDPOINT", "SCHEDULE", "PEAK_LR", "SAVE_AT", "SEEDS", "fit", "DRAWS",
           "replay_scopes", "SAVED_NAMES", "make_model", "load_checkpoint"]

DATASETS = {"tahoe": "tahoe-100m", "parse": "parse-pbmc"}


def load_instance(root, dataset, fold):
    """The instance of one fold (Tahoe-100M) or group (Parse PBMC), ``<root>/<DATASETS[dataset]>/<fold>/synthetic``:
    for example ``("tahoe", "F1")`` or ``("parse", "G1")``."""
    return Instance(Path(root) / DATASETS[dataset] / fold / "synthetic")


# Training: 40,000 updates of eight bags, on a learning-rate schedule defined over 160,000 updates (warm-up over 1,000
# updates, then a cosine that would reach 10% of the peak at 160,000); the peak learning rate depends on the model.
ENDPOINT = 40000
SCHEDULE = 160000
PEAK_LR = {"attention_model": 3e-5, "deepsets": 1e-4, "summary": 1e-4}
SAVE_AT = (5000, 10000, 20000, 30000, 40000)
SEEDS = (31002, 31003, 31004)


def fit(model, seed, level):
    """The training run of one model, seed and level: 40,000 updates of the 160,000-update schedule."""
    return Fit(model, seed, level, PEAK_LR[model], steps=ENDPOINT, total=SCHEDULE, save_at=SAVE_AT)


# Evaluation: technical draws per bag. The L0 and L1 inputs do not depend on the draw, so one is enough.
DRAWS = {"L0": 1, "L1": 1, "L2": 16, "L3": 16, "L3d": 16, "L4": 16}


def replay_scopes(model, level, step):
    """Which configurations the checkpoint at 40,000 updates replays: "full" = the base state and all 25 flips,
    "base" = the base state only.

    Every model replays the flips on the held-out backgrounds. On the seen backgrounds it replays them only for the
    summary-statistics model and the attention model at L0-L2 and the summary-statistics model at L3.
    """
    if step != ENDPOINT:
        raise ValueError(f"The paper replays only the checkpoint at {ENDPOINT} updates, not step {step}")
    seen_full = (model in ("summary", "attention_model") and level in ("L0", "L1", "L2")) or (model, level) == ("summary", "L3")
    return {HELDOUT: "full", SEEN: "full" if seen_full else "base"}


SAVED_NAMES = {"state": "attention_model"}   # the attention model's name in the study's saved checkpoints


def make_model(generator, name, seed, device):
    """A freshly initialized network (``"attention_model"``, ``"deepsets"`` or ``"summary"``) for one generator's
    perturbations, from ``scsailor.models``.

    ``set_runtime(seed)`` runs immediately before the network is built, and its effects last for the rest of the
    process: Python, NumPy and torch (CPU and CUDA) are seeded, TF32 is off, FP32 matmul precision is ``'highest'``,
    and torch is left with one CPU thread.
    """
    from scsailor.models import RawModel, set_runtime

    set_runtime(seed)
    return RawModel(name, generator.intercept, len(generator.p_ids), generator.r, generator.c_train).to(device)


def load_checkpoint(instance, path, device):
    """A trained network in eval mode and its checkpoint record, once the record names this instance. The record's
    ``model_name`` is the public name.

    The network is built by ``make_model``, so the same lasting runtime effects apply, with the checkpoint's seed.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["instance"] != instance.name or saved["parameters"] != instance.record["parameters"]:
        raise ValueError(f"{path} belongs to another synthetic-answer instance")
    saved["model_name"] = SAVED_NAMES.get(saved["model_name"], saved["model_name"])
    model = make_model(instance.generator, saved["model_name"], saved["seed"], device)
    model.load_state_dict(saved.pop("model"), strict=True)
    model.eval()
    return model, saved
