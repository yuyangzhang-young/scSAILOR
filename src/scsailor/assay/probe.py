"""Linear read-out probes: is the planted state linearly readable inside the trained network?

A probe design (``ProbeDesign``; the paper's designs are ``PROBE`` in ``reproduce/study.py``) gives each training background
``bags_per_context`` probe bags (256 cells, a random state, one technical draw, perturbation 0). One
internal feature vector is taken per bag from the submodules ``contract.READOUT_MODULES`` names for the
network. Ridge readers on standardized features then predict the visible bits (5 per visible statistic,
``design``).

The backgrounds are split in their instance order (``ProbeDesign.split``: fit, tune, test): readers are
fit on the first group for six ridge strengths, each bit keeps the strength with the best sign accuracy
on the second group, the final readers are refit on both groups, and the third group is the test. On the
test backgrounds the probe also builds local flip pairs (each bag's state and its 25 single-bit flips,
same draws) and scores them like ``score.flip_pairs``, with perturbation 0 only.

torch is imported only by the functions that run the model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .carrier import carrier
from .constants import CELLS_KEY, N_BITS, N_GENES, STATE_KEY
from .contract import READOUT_MODULES
from .design import LEVELS
from .generator import Background, bag_indices, flip_set, known_answer
from .score import BIT_KEY, flip_pairs, pool_contexts

ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.)
MIN_SD = 1e-5

__all__ = ["ProbeDesign", "fit_ridge", "read", "fit_readers", "test_rows", "read_scores", "probe_bags", "readout",
           "run_probe"]


@dataclass(frozen=True)
class ProbeDesign:
    """Where a probe's random keys start, how many bags each training background gives and how the
    backgrounds are split."""
    root: int               # first entry of every probe key
    draw_key: int           # last entry of each bag's technical-draw key
    bags_per_context: int   # probe bags per training background
    split: tuple            # training backgrounds, in instance order, that fit the readers, tune their strength and test them


@dataclass
class Bag:
    x: np.ndarray        # carrier rows (512 x genes)
    depth: np.ndarray    # carrier depths (512)
    state: np.ndarray    # planted state (25)
    context_index: int   # background index in the instance's training order (saved key "q")
    rep: int             # bag index within the background


@dataclass
class Reader:
    mean: np.ndarray      # feature means
    sd: np.ndarray        # feature standard deviations (at least MIN_SD)
    center: np.ndarray    # mean target per bit
    weights: np.ndarray   # features x bits
    constant: bool        # all standardized features are zero (nothing to read)
    alpha: np.ndarray     # selected ridge strength per bit


@dataclass(frozen=True)
class _Run:
    """What the local flip pairs of one probe run need."""
    model: object
    modules: tuple
    instance: object
    level: str
    design: ProbeDesign
    device: object


def fit_ridge(x, y, alpha):
    """Ridge regression of ``y`` on standardized ``x``, solved in the bag (dual) space.

    The penalty is ``alpha`` times the mean squared norm of a standardized bag.
    Returns (mean, sd, center, weights, constant).
    """
    mean, sd = x.mean(axis=0), np.maximum(x.std(axis=0), MIN_SD)
    a, center = (x - mean) / sd, y.mean(axis=0)
    gram = a @ a.T
    scale = np.trace(gram) / len(gram)
    if scale == 0:
        return mean, sd, center, np.zeros((x.shape[1], y.shape[1])), True
    return mean, sd, center, a.T @ np.linalg.solve(gram + alpha * scale * np.eye(len(gram)), y - center), False


def read(reader, values):
    """Reader outputs for feature rows."""
    return (values - reader.mean) / reader.sd @ reader.weights + reader.center


def _rows(design):
    """Ends of the fit rows and of the tune rows; probe bags are in background order."""
    fit, tune, _ = design.split
    return fit * design.bags_per_context, (fit + tune) * design.bags_per_context


def test_rows(features, design):
    """Rows of the test backgrounds."""
    return slice(_rows(design)[1], len(features))


def fit_readers(features, states, level, design):
    """Readers for the bits visible at ``level``, with the ridge strength chosen per bit on the tune rows."""
    visible = 5 * LEVELS.index(level)
    fit_end, tune_end = _rows(design)
    targets = states[:, :visible]
    accuracy = []
    for alpha in ALPHAS:
        candidate = Reader(*fit_ridge(features[:fit_end], targets[:fit_end], alpha), alpha=None)
        prediction = read(candidate, features[fit_end:tune_end])
        accuracy.append(np.mean(np.sign(prediction) == targets[fit_end:tune_end], axis=0))
    selected = np.argmax(accuracy, axis=0)
    mean, sd, center, weights, constant = fit_ridge(features[:tune_end], targets[:tune_end], ALPHAS[0])
    for index, alpha in enumerate(ALPHAS):
        columns = np.flatnonzero(selected == index)
        if len(columns):
            weights[:, columns] = fit_ridge(features[:tune_end], states[:tune_end, columns], alpha)[3]
    return Reader(mean, sd, center, weights, constant, np.asarray(ALPHAS)[selected])


def read_scores(prediction, true_states, reader, params):
    """Per visible bit: correct signs, squared error and the chosen ridge strength on the test bags."""
    return [{BIT_KEY: a, "correct": int(np.sum(np.sign(prediction[:, a]) == true_states[:, a])),
             "n": len(true_states), "alpha": float(reader.alpha[a]),
             "MSE": float(np.mean((prediction[:, a] - true_states[:, a]) ** 2)),
             "hidden_input_control": bool(a < 5 and not params.I_mean)} for a in range(prediction.shape[1])]


def _probe_bag(pool, depth, context_index, rep, design):
    """Cells, depths and random state of probe bag ``rep`` of one training background."""
    indices = bag_indices(len(pool), [design.root, context_index, rep, CELLS_KEY])
    state = np.random.default_rng(np.random.SeedSequence([design.root, context_index, rep, STATE_KEY])).choice(
        [-1, 1], N_BITS)
    return pool[indices], depth[indices], state


def _technical(context_index, rep, design):
    return np.random.default_rng(np.random.SeedSequence([design.root, context_index, rep, design.draw_key]))


def probe_bags(instance, level, design, contexts=None, reps=None):
    """Probe bags of the given training-background indices (default: all), in background then bag order."""
    bags = []
    names = instance.train_contexts
    for context_index in range(len(names)) if contexts is None else contexts:
        pool, depth = instance.pool("train", names[context_index])
        for rep in range(design.bags_per_context) if reps is None else reps:
            x, d, state = _probe_bag(pool, depth, context_index, rep, design)
            rows, row_depth = carrier(x, d, state, level, _technical(context_index, rep, design), instance.params)
            bags.append(Bag(rows, row_depth, state, context_index, rep))
    return bags


def _pooled(output):
    """One bag's feature from a hooked output: averaged over rows when it has a row axis."""
    return (output.mean(dim=1) if output.dim() == 3 else output).detach().cpu().numpy()[0]


def readout(model, modules, level, x, depth, device):
    """Internal feature (the outputs of ``modules``), raw output and baseline of one bag for perturbation 0."""
    import torch

    captured = {}
    handles = [model.get_submodule(path).register_forward_hook(
        lambda module, inputs, output, path=path: captured.__setitem__(path, output)) for path in modules]
    rows, row_depth = model.canonical_rows(torch.from_numpy(x[None]).to(device), torch.from_numpy(depth[None]).to(device))
    p = torch.zeros(1, dtype=torch.long, device=device)
    try:
        raw = model(rows, row_depth, p, level, canonicalize=False).cpu().numpy()[0]
    finally:
        for handle in handles:
            handle.remove()
    baseline = model.anchor(rows, p, level).cpu().numpy()[0]
    return np.concatenate([_pooled(captured[path]) for path in modules]), raw, baseline


def run_probe(model, instance, checkpoint, design, output, device):
    """The complete probe of one trained model (``checkpoint`` is its record); writes the four probe files."""
    name, level = checkpoint["model_name"], checkpoint["level"]
    if level == "L0":
        raise ValueError("Nothing is visible at L0, so there is nothing to probe")
    if sum(design.split) != len(instance.train_contexts):
        raise ValueError(f"The probe split {tuple(design.split)} (fit, tune, test) must cover all "
                         f"{len(instance.train_contexts)} training backgrounds of the instance, in order")
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    modules = READOUT_MODULES[name]
    contexts = instance.train_contexts
    n = design.bags_per_context
    features, states, raws, baselines = [], [], [], []
    with torch.inference_mode():
        for bag in probe_bags(instance, level, design):
            feature, raw, baseline = readout(model, modules, level, bag.x, bag.depth, device)
            features.append(feature)
            states.append(bag.state)
            raws.append(raw)
            baselines.append(baseline)
    features, states = np.asarray(features, dtype=np.float64), np.asarray(states)
    raws, baselines = np.asarray(raws), np.asarray(baselines)
    reader = fit_readers(features, states, level, design)
    tests = test_rows(features, design)
    prediction = read(reader, features[tests])
    np.savez(output / "PROBE_FEATURES.npz", features=features, states=states, raw=raws, anchor_raw=baselines,
             q=np.repeat(np.arange(len(contexts)), n), rep=np.tile(np.arange(n), len(contexts)),
             context=np.repeat(contexts, n), fixed_p=0)
    np.savez(output / "READER.npz", feature_mean=reader.mean, feature_sd=reader.sd, target_center=reader.center,
             weights=reader.weights, alpha=reader.alpha, test_prediction=prediction,
             test_state=states[tests, :5 * LEVELS.index(level)])
    run = _Run(model, modules, instance, level, design, device)
    tables = []
    with h5py.File(output / "LOCAL_PAIRS.h5", "w-") as local, torch.inference_mode():
        for context_index in range(sum(design.split[:2]), sum(design.split)):
            arrays, rows = _local_pairs(run, context_index, (features, raws, baselines), reader)
            group = local.create_group(contexts[context_index])
            for key, value in arrays.items():
                group.create_dataset(key, data=value, compression="lzf")
            tables.append(rows)
    summary = {"instance": instance.name, "model": name, "level": level, "split_contexts": list(design.split),
               "constant_representation": reader.constant,
               "modules": read_scores(prediction, states[tests], reader, instance.params),
               "local_modules": pool_contexts(tables)}
    (output / "PROBE_AND_LOCAL.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def _local_pairs(run, context_index, readouts, reader):
    """Flip pairs of one test background's probe bags: the arrays of its ``LOCAL_PAIRS.h5`` group and its
    flip rows. ``readouts`` are the probe's features, raw outputs and baselines of every probe bag."""
    pool = run.instance.pool("train", run.instance.train_contexts[context_index])
    background = Background(pool[0])
    n = run.design.bags_per_context
    bags = [_flip_bag(run, pool, context_index, rep, [values[context_index * n + rep] for values in readouts])
            for rep in range(n)]
    features, raw, baseline, states = (np.stack(part) for part in zip(*bags))
    truth = np.stack([[known_answer(run.instance.generator, background, [0], state[None]) for state in bag_states]
                      for bag_states in states])
    rows, _ = flip_pairs(np.maximum(raw, 0), truth, states, raw, baseline)
    return {"feature": features, "raw": raw, "anchor_raw": baseline, "truth": truth, "states": states,
            "reader_output": read(reader, features)}, rows


def _flip_bag(run, pool, context_index, rep, base_readout):
    """One test bag's state and its 25 single-bit flips, read out with the bag's draws: features
    (26 x features), raw output and baseline (26 x 1 x genes), all float64, and states (26 x 25).
    Configuration 0 reuses the bag's probe readout ``base_readout``."""
    x, depth, state = _probe_bag(*pool, context_index, rep, run.design)
    states = flip_set(state)
    features = np.empty((N_BITS + 1, len(base_readout[0])))
    raw, baseline = (np.empty((N_BITS + 1, 1, N_GENES)) for _ in range(2))
    features[0], raw[0, 0], baseline[0, 0] = base_readout
    for config in range(1, N_BITS + 1):
        rows, row_depth = carrier(x, depth, states[config], run.level, _technical(context_index, rep, run.design),
                                  run.instance.params)
        features[config], raw[config, 0], baseline[config, 0] = readout(run.model, run.modules, run.level, rows,
                                                                        row_depth, run.device)
    return features, raw, baseline, states
