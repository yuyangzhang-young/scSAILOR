"""Replay a trained model on the fixed evaluation support.

For each evaluation background, base state (4 random ones) and configuration (the base
state and its 25 single-bit flips), the bag's carrier is rebuilt ``draws`` times with
technical draws keyed by ``(root, context_index, base, draw, 888)``. The key does not
depend on the configuration, so a flip pair shares every draw. The model's raw output
and its baseline (``model.anchor``: the bag mean plus the perturbation's response field,
the part the network's output is added to) are averaged over the draws in float64 and
stored once as float32. Clipping at zero happens only in scoring.

``write_predictions`` stores ``raw``, ``anchor_raw`` (the baseline; bases x 26 x
perturbations x genes) and ``done`` (bases x 26) per background in the original layout.
The number of draws is a setting (``DRAWS`` in ``reproduce/study.py``); torch is imported only to run the model.
"""
from __future__ import annotations

import json

import h5py
import numpy as np

from .carrier import carrier
from .constants import N_BITS, N_GENES, REPLAY_DRAW_KEY

CHUNK = 8  # perturbations per forward pass

__all__ = ["replay_bag", "write_predictions"]


def replay_bag(model, bag, level, params, draws, configs, device):
    """Draw-averaged raw output and baseline for every base state of one ``EvaluationBag`` and the given
    configurations.

    Returns two float32 arrays, bases x len(configs) x perturbations x genes.
    """
    import torch

    p_count = len(model.r_pool)  # the model's perturbations: one response-field row each
    raw = np.empty((len(bag.states), len(configs), p_count, N_GENES), dtype=np.float32)
    baseline = np.empty_like(raw)
    for base in range(len(bag.states)):
        for j, config in enumerate(configs):
            raw_sum, baseline_sum = np.zeros((p_count, N_GENES)), np.zeros((p_count, N_GENES))
            for draw in range(draws):
                rng = np.random.default_rng(np.random.SeedSequence([bag.root, bag.context_index, base, draw,
                                                                    REPLAY_DRAW_KEY]))
                rows, row_depth = carrier(bag.x, bag.depth, bag.states[base, config], level, rng, params)
                rows, row_depth = torch.from_numpy(rows[None]).to(device), torch.from_numpy(row_depth[None]).to(device)
                for start in range(0, p_count, CHUNK):
                    stop = min(start + CHUNK, p_count)
                    p = torch.arange(start, stop, dtype=torch.long, device=device)
                    xc, dc = model.canonical_rows(rows.expand(len(p), -1, -1), row_depth.expand(len(p), -1))
                    raw_sum[start:stop] += model(xc, dc, p, level, canonicalize=False).cpu().numpy().astype(np.float64)
                    baseline_sum[start:stop] += model.anchor(xc, p, level).cpu().numpy().astype(np.float64)
            raw[base, j] = (raw_sum / draws).astype(np.float32)
            baseline[base, j] = (baseline_sum / draws).astype(np.float32)
            if not np.isfinite(raw[base, j]).all():
                raise ValueError("Nonfinite raw prediction")
    return raw, baseline


def write_predictions(model, instance, level, step, scopes, draws, output, device, checkpoint=""):
    """Replay every evaluation background of each side in ``scopes`` ("full": 26 configurations, "base": 1),
    with ``draws`` technical draws per bag."""
    unknown = {side: scope for side, scope in scopes.items() if scope not in ("full", "base")}
    if unknown:
        raise ValueError(f"Unknown replay scope {unknown}; the valid scopes are 'full' and 'base'")
    import torch

    with h5py.File(output, "w-") as out, torch.inference_mode():
        out.attrs["checkpoint"] = str(checkpoint)
        out.attrs["instance"] = instance.name
        out.attrs["parameters"] = json.dumps(instance.record["parameters"], sort_keys=True)
        out.attrs["step"] = step
        for side, scope in scopes.items():
            configs = list(range(N_BITS + 1)) if scope == "full" else [0]
            for bag in instance.evaluation_bags(side):
                raw, baseline = replay_bag(model, bag, level, instance.params, draws, configs, device)
                group = out.create_group(f"{side}/{bag.context}")
                shape = (len(bag.states), N_BITS + 1, raw.shape[2], N_GENES)
                for field, values in (("raw", raw), ("anchor_raw", baseline)):
                    dataset = group.create_dataset(field, shape=shape, dtype="f4", chunks=(1, 1, shape[2], N_GENES),
                                                   compression="lzf")
                    dataset[:, configs[0]:configs[-1] + 1] = values
                done = np.zeros((len(bag.states), N_BITS + 1), dtype=bool)
                done[:, configs] = True
                group.create_dataset("done", data=done)
