"""Constructed predictions with known errors, scored by whole-profile and paired-response metrics (Fig. 3d,
Supplementary Note S6.7), computed on a CPU from an instance's synthetic targets; no model.

Targets (``targets``): on each held-out context, the targets of the four base states and of their flips of a spread
test (planted states 5, shared, and 6, selective), on the 1,950 scored genes, minus the context's full control-pool
mean. The training reference: over the evaluation's training contexts, the mean of each context's base-state
targets minus its control-pool mean, averaged over base states and perturbations (one gene vector).

Constructed predictions (``construct``) of a pair of profiles y0 (base state) and y1 (flipped), with pair center
c = (y0 + y1)/2 and half-difference h = (y0 - y1)/2:

* Exact: y0 and y1;
* Offset: y0 + e and y1 + e, the same error e on both states: each context's and base state's mean profile over
  perturbations, rolled by one gene (an error profile that does not lie along the truth's own mean) and scaled to
  mean square 4V, V = mean(h^2);
* Swapped: c' + h and c' - h, with the pair centers c' rolled by one perturbation;
* Halved, Absent, Reversed: c + g h and c - g h, g = 0.5, 0 and -1.

Whole-profile scores (``profile_scores``) score each state's profiles (rows: context, base state and perturbation) and
average the two states; paired scores (``paired_scores``) score the half-differences, oriented by the planted state's
sign in the base state. Recovery Q = 1 - M/V. Correlations are the mean over rows with a defined value.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from scsailor.assay.constants import SCORED
from scsailor.assay.instance import HELDOUT, SEEN
from scsailor.natural.score import row_correlation
from scsailor.resolution.metrics import effect_weights, r2_units

from .scores import describe

__all__ = ["SPREAD", "CASES", "GAINS", "OFFSET_SIZE", "METRICS", "interaction_part", "centroid_accuracy", "global_r2",
           "row_r2", "targets", "construct", "profile_scores", "paired_scores", "constructed_table", "summary"]

SPREAD = (5, 6)                                     # the shared and the selective spread tests
CASES = ("Exact", "Offset", "Swapped", "Halved", "Absent", "Reversed")
GAINS = {"Halved": .5, "Absent": 0., "Reversed": -1.}
OFFSET_SIZE = 4                                     # the offset's mean square, in units of V
METRICS = ("PearsonΔ", "interaction alignment", "centroid accuracy", "R²", "R²row", "R²w", "R²w,full", "Q",
           "squared error / V")


def _rows(x):
    return x.reshape(-1, x.shape[-1])


def _mean_defined(values):
    """The mean over rows whose value is not NaN; NaN when none is."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(values))


def interaction_part(y):
    """The interaction part of profiles y (contexts x base states x perturbations x genes): y minus, per base state,
    its grand mean and its context and perturbation main effects."""
    mu = y.mean((0, 2), keepdims=True)
    return y - mu - (y.mean(2, keepdims=True) - mu) - (y.mean(0, keepdims=True) - mu)


def centroid_accuracy(truth, prediction):
    """Per context and base state, for each perturbation: the fraction of the other perturbations' targets strictly
    farther (squared distance) from its prediction than its own target; the mean over perturbations, then over
    contexts and base states. Ties earn no credit."""
    contexts, bases, perturbations = truth.shape[:3]
    accuracy = []
    for b in range(contexts):
        for s in range(bases):
            distance = ((prediction[b, s][:, None, :] - truth[b, s][None, :, :]) ** 2).sum(-1)
            accuracy.append(((distance > np.diag(distance)[:, None]).sum(1) / (perturbations - 1)).mean())
    return float(np.mean(accuracy))


def global_r2(y, p):
    """1 - mean squared error / the variance of y over all its entries."""
    return 1 - float(np.mean((p - y) ** 2)) / float(np.mean((y - y.mean()) ** 2))


def row_r2(y, p):
    """The mean over rows of 1 - mean squared error / the row's variance over genes."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return _mean_defined(1 - np.mean((p - y) ** 2, -1) / np.mean((y - y.mean(-1, keepdims=True)) ** 2, -1))


def _weighted(y, p):
    """R²w (adapted weighted R² of the change from y's mean over perturbations) and R²w,full (the same, with the
    weights recomputed from the full response)."""
    mean = y.mean(2, keepdims=True)
    d = _rows(y - mean)
    return {"R²w": _mean_defined(r2_units(d, _rows(p - mean), effect_weights(d))),
            "R²w,full": _mean_defined(r2_units(_rows(y), _rows(p), effect_weights(_rows(y))))}


def targets(instance):
    """y0 (held-out contexts x base states x perturbations x scored genes), per spread test its flipped targets y1
    and its signs in the base states (contexts x base states x 1 x 1), and the training reference, all minus each
    context's control-pool mean."""
    y0, y1, signs = [], {a: [] for a in SPREAD}, {a: [] for a in SPREAD}
    for entry in instance.evaluation[HELDOUT]:
        control = instance.pool(HELDOUT, entry["context"])[0].mean(0)[SCORED]
        states, truth = instance.truth(HELDOUT, entry["context"])
        y0.append(truth[:, 0, :, SCORED] - control)
        for a in SPREAD:
            y1[a].append(truth[:, a + 1, :, SCORED] - control)
            signs[a].append(states[:, 0, a])
    reference = np.mean([(instance.truth(SEEN, e["context"])[1][:, 0, :, SCORED]
                          - instance.pool("train", e["context"])[0].mean(0)[SCORED]).mean((0, 1))
                         for e in instance.evaluation[SEEN]], axis=0)
    return (np.stack(y0), {a: np.stack(y1[a]) for a in SPREAD},
            {a: np.stack(signs[a])[:, :, None, None] for a in SPREAD}, reference)


def construct(case, y0, y1):
    """The constructed predictions (p0, p1) of the pair (y0, y1)."""
    if case == "Exact":
        return y0, y1
    center, half = (y0 + y1) / 2, (y0 - y1) / 2
    if case in GAINS:
        return center + GAINS[case] * half, center - GAINS[case] * half
    if case == "Swapped":
        wrong = np.roll(center, 1, axis=2)
        return wrong + half, wrong - half
    if case == "Offset":
        size = float(np.mean(half ** 2))
        error = np.broadcast_to(np.roll(y0.mean(2, keepdims=True), 1, axis=-1), y0.shape)
        error = np.sqrt(OFFSET_SIZE) * (error * np.sqrt(size / np.mean(error ** 2)))
        return y0 + error, y1 + error
    raise ValueError(f"Unknown constructed case {case}")


def profile_scores(y, p, reference):
    """Whole-profile scores of one state's predictions p of profiles y: PearsonΔ after the training reference,
    interaction alignment, centroid accuracy, R², R²row, R²w, R²w,full, and the squared error."""
    return {"PearsonΔ": _mean_defined(row_correlation(_rows(p - reference), _rows(y - reference))),
            "interaction alignment": _mean_defined(row_correlation(_rows(p), _rows(interaction_part(y)))),
            "centroid accuracy": centroid_accuracy(y, p), "R²": global_r2(y, p), "R²row": row_r2(y, p),
            **_weighted(y, p), "squared error": float(np.mean((p - y) ** 2))}


def paired_scores(y0, y1, p0, p1, signs):
    """Paired-response scores: the same scores of the oriented predicted half-differences against the true ones
    (Pearson without a reference), Q, and the squared error in units of V."""
    half = (y0 - y1) / 2
    size = float(np.mean(half ** 2))
    truth, prediction = half * signs, ((p0 - p1) / 2) * signs
    error = float(np.mean((prediction - truth) ** 2)) / size
    return {"PearsonΔ": _mean_defined(row_correlation(_rows(prediction), _rows(truth))),
            "interaction alignment": _mean_defined(row_correlation(_rows(prediction),
                                                                   _rows(interaction_part(truth)))),
            "centroid accuracy": centroid_accuracy(truth, prediction), "R²": global_r2(truth, prediction),
            "R²row": row_r2(truth, prediction), **_weighted(truth, prediction), "Q": 1 - error,
            "squared error / V": error}


def constructed_table(instance, fold):
    """One row per spread test, constructed case, assessment (whole profile, paired) and score on one fold."""
    y0, flips, signs, reference = targets(instance)
    rows = []
    for a in SPREAD:
        y1 = flips[a]
        size = float(np.mean(((y0 - y1) / 2) ** 2))
        for case in CASES:
            p0, p1 = construct(case, y0, y1)
            first, second = profile_scores(y0, p0, reference), profile_scores(y1, p1, reference)
            whole = {k: (first[k] + second[k]) / 2 for k in first if k != "squared error"}
            whole["squared error / V"] = (first["squared error"] + second["squared error"]) / 2 / size
            for assessment, scores in (("whole profile", whole), ("paired", paired_scores(y0, y1, p0, p1, signs[a]))):
                rows += [{"fold": fold, **describe(a), "case": case, "assessment": assessment, "metric": m,
                          "value": scores[m]} for m in METRICS if m in scores]
    return pd.DataFrame(rows)


def summary(table):
    """Fig. 3d: per case, score and assessment, the mean, minimum and maximum over the folds' spread tests, and the
    number of tests with a defined value."""
    groups = table.groupby(["case", "metric", "assessment"], sort=False).value
    return groups.agg(mean="mean", minimum="min", maximum="max", n_tests="count").reset_index()
