"""Scores of replayed predictions against known answers, on the scored genes 50-1999.

Risk (base states only): for one background the error ``f - y`` has a part shared
by all perturbations of a base state (its mean over perturbations) and the rest:

    R = mean(error^2) = R_common + R_contrast.

Flip-pair recovery (base states and their 25 single-bit flips): for bit a the true
half-contrast is ``dy = s * (y(base) - y(flip a)) / 2`` and the model's is
``df = s * (f(base) - f(flip a)) / 2``, with ``s`` the bit's sign in the base, so
both read "bit +1 minus bit -1". Averaged over base states and perturbations:

    E = mean(dy^2)          true contrast energy
    M = mean((df - dy)^2)   contrast error
    A = mean(df^2)          predicted contrast energy
    K = 2 mean(df * dy)     agreement, so that M = E + A - K
    Q = 1 - M / E           recovery: 1 exact, 0 no better than predicting no contrast, < 0 wrong
    kappa = K / (2E)        slope of the predicted on the true contrast
    P = A - K^2 / (4E)      predicted energy not along the true contrast
    C_pair                  error of the pair's centre; R_pair = C_pair + M
    J_catalog, J_MC         spread of the true contrast across base states (MC: n / (n - 1) corrected)

The null bits (role ``null``) have E = 0, so Q is undefined; their A measures
leakage, a contrast the model makes where the answer has none (``capability.leakage_ratio``).

Saved keys keep their original names: ``anchor_R`` and ``d_anchor_*`` refer to the model's
baseline (``contract``), and score rows name the bit ``module`` (``BIT_KEY``).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .constants import FIRST_SCORED_GENE, N_BITS
from .instance import HELDOUT, SEEN

POOLED = ("E", "M", "A", "K", "C_pair", "R_pair", "J_catalog", "J_MC")
BIT_KEY = "module"   # saved key of the bit in score rows (SCORES.json, PROBE_AND_LOCAL.json)

__all__ = ["risk", "flip_pairs", "pool_contexts", "score_background", "score_predictions"]


def risk(prediction, truth):
    """Risk of one background: base states x perturbations x genes; perturbations weigh equally."""
    error = np.asarray(prediction, dtype=np.float64)[..., FIRST_SCORED_GENE:] - truth[..., FIRST_SCORED_GENE:]
    center = error.mean(axis=-2, keepdims=True)
    return {"R": float(np.mean(error ** 2)), "R_common": float(np.mean(center ** 2)),
            "R_contrast": float(np.mean((error - center) ** 2)),
            "row_RMSE": float(np.sqrt(np.mean(error ** 2, axis=-1)).mean())}


def _ratios(E, M, A, K):
    return {"Q": 1 - M / E if E > 0 else None, "kappa": K / (2 * E) if E > 0 else None,
            "P": A - K * K / (4 * E) if E > 0 else None, "identity_residual": M - E - A + K}


def flip_pairs(prediction, truth, states, raw=None, baseline=None):
    """Recovery rows of the 25 bits on one background, and their per base state x perturbation contributions.

    ``prediction`` and ``truth``: bases x 26 configurations x perturbations x genes (configuration 0
    is the base, a + 1 flips bit a); ``states``: bases x 26 x 25. With ``raw`` and ``baseline`` the
    predicted half-contrast is also split into the baseline's part, the network's part and clipping.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    rows, contributions = [], {}
    for a in range(N_BITS):
        sign = states[:, 0, a, None, None]
        dt = sign * (truth[:, 0] - truth[:, a + 1]) / 2
        df = sign * (prediction[:, 0] - prediction[:, a + 1]) / 2
        t, f = dt[..., FIRST_SCORED_GENE:], df[..., FIRST_SCORED_GENE:]
        e = np.mean(t * t, axis=-1)
        error = np.mean((f - t) ** 2, axis=-1)
        energy = np.mean(f * f, axis=-1)
        k = 2 * np.mean(f * t, axis=-1)
        centre = (prediction[:, 0] + prediction[:, a + 1] - truth[:, 0] - truth[:, a + 1]) / 2
        c = np.mean(centre[..., FIRST_SCORED_GENE:] ** 2, axis=-1)
        E, M, A, K, C = (float(v.mean()) for v in (e, error, energy, k, c))
        catalog = float(np.mean((t - t.mean(axis=0, keepdims=True)) ** 2))
        rows.append({BIT_KEY: a, "E": E, "M": M, "A": A, "K": K, "C_pair": C, "R_pair": C + M, **_ratios(E, M, A, K),
                     "J_catalog": catalog, "J_MC": catalog * len(states) / (len(states) - 1)})
        for name, value in (("E", e), ("M", error), ("A", energy), ("K", k), ("C_pair", c)):
            contributions[f"{name}_{a}"] = value
        contributions[f"d_truth_{a}"] = dt
        contributions[f"d_final_{a}"] = df
        if raw is not None and baseline is not None:
            da = sign * (baseline[:, 0].astype(np.float64) - baseline[:, a + 1].astype(np.float64)) / 2
            dr = sign * (raw[:, 0].astype(np.float64) - raw[:, a + 1].astype(np.float64)) / 2
            contributions[f"d_anchor_{a}"] = da
            contributions[f"d_delta_{a}"] = dr - da
            contributions[f"d_projection_{a}"] = df - dr
    return rows, contributions


def pool_contexts(tables):
    """Pool per-background rows: average E, M, A, K, ... over backgrounds, then recompute Q and the ratios."""
    pooled = []
    for a in range(N_BITS):
        row = {k: float(np.mean([table[a][k] for table in tables])) for k in POOLED}
        row.update({BIT_KEY: a}, **_ratios(row["E"], row["M"], row["A"], row["K"]))
        pooled.append(row)
    return pooled


def score_background(raw, baseline, truth, states):
    """Scores of one evaluation background: the base states' risks, and flip-pair rows when all 26
    configurations are given.

    ``raw`` and ``baseline`` (saved as ``raw`` and ``anchor_raw``) are bases x configurations x
    perturbations x genes, with one configuration (the base states) or all 26; ``truth`` is bases x 26 x
    perturbations x genes and ``states`` bases x 26 x 25. Returns the background's score row (with
    ``"modules"`` when flips are scored) and the flip contributions of ``flip_pairs`` (None without flips).
    """
    row = risk(np.maximum(raw[:, 0], 0), truth[:, 0])
    row["anchor_R"] = risk(np.maximum(baseline[:, 0], 0), truth[:, 0])["R"]
    row["negative_raw_fraction"] = float(np.mean(raw[:, 0] < 0))
    if raw.shape[1] == 1:
        return row, None
    rows, contributions = flip_pairs(np.maximum(raw, 0), truth, states, raw, baseline)
    row["modules"] = rows
    return row, contributions


def score_predictions(instance, predictions, output=None):
    """Score a predictions file against the instance's known answers.

    Returns ``(scores, contributions)``. ``scores[side]`` averages the risks over the side's
    backgrounds and, when every background has all 26 configurations, pools the flip-pair rows;
    ``Gap_seen`` is the held-out minus the seen risk. With ``output`` the scores are also written as
    ``SCORES.json`` and the contributions (without half-contrasts) as ``{side}_{context}_contributions.npz``.
    """
    scores, contributions = {}, {}
    with h5py.File(predictions, "r") as saved:
        if saved.attrs["instance"] != instance.name or json.loads(saved.attrs["parameters"]) != instance.record["parameters"]:
            raise ValueError(f"{predictions} belongs to another synthetic-answer instance")
        for side in saved:
            contexts, tables = {}, []
            for entry in instance.evaluation[side]:
                context = entry["context"]
                group = saved[f"{side}/{context}"]
                done = group["done"][:]
                if not done[:, 0].all():
                    raise ValueError(f"Incomplete base predictions: {side}/{context}")
                states, truth = instance.truth(side, context)
                configs = slice(None) if done.all() else slice(0, 1)
                row, parts = score_background(group["raw"][:, configs], group["anchor_raw"][:, configs], truth, states)
                if parts is not None:
                    tables.append(row["modules"])
                    contributions[f"{side}_{context}"] = {k: v for k, v in parts.items() if not k.startswith("d_")}
                contexts[context] = row
            summary = {k: float(np.mean([v[k] for v in contexts.values()]))
                       for k in ("R", "R_common", "R_contrast", "anchor_R", "row_RMSE")}
            summary["contexts"] = contexts
            if len(tables) == len(contexts):
                summary["modules"] = pool_contexts(tables)
            scores[side] = summary
    if HELDOUT in scores and SEEN in scores:
        scores["Gap_seen"] = scores[HELDOUT]["R"] - scores[SEEN]["R"]
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "SCORES.json").write_text(json.dumps({"instance": instance.name, "prediction": str(predictions),
                                                         **scores}, indent=2, allow_nan=False) + "\n")
        for key, values in contributions.items():
            np.savez(output / f"{key}_contributions.npz", **values)
    return scores, contributions
