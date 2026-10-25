"""Recovery of a planted statistic: how much of a known contrast the model reproduces.

For a flip pair the true half-contrast is ``dy = (y+ - y-) / 2`` and the model's is
``df = (relu(f+) - relu(f-)) / 2``, both on output genes ``FIRST_SCORED_GENE`` onward. Recovery over a set of
pairs is ``Q = 1 - sum(mean((df - dy)^2)) / sum(mean(dy^2))``: 1 means the contrast is reproduced exactly,
0 means no better than predicting no contrast, and negative values mean a wrong contrast. ``Q_raw`` is the
same recovery with ``df`` taken from the unclipped outputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..assay.constants import FIRST_SCORED_GENE
from .constants import LEVEL
from .draws import flip_pairs

__all__ = ["PanelSpec", "predict", "pair_row", "recovery", "support_summary", "score_pairs", "evaluate"]


@dataclass
class PanelSpec:
    """Where a support's recovery panel comes from."""
    root: int
    pool_group: str
    pairs: int
    contexts: list


def predict(model, endpoint, device, level=LEVEL):
    """The model's raw output for one endpoint, as float64 on the CPU."""
    import torch

    p = torch.tensor([endpoint.p], device=device)
    output = model(endpoint.x.to(device), endpoint.depth.to(device), p, level, canonicalize=True)
    return output[0].double().cpu().numpy()


def pair_row(context, p, predictions, truths):
    """Contrast energy, contrast error and raw endpoint errors for one flip pair."""
    f_minus, f_plus = predictions
    y_minus, y_plus = truths
    dy = (y_plus - y_minus)[FIRST_SCORED_GENE:] / 2
    df = (np.maximum(f_plus, 0) - np.maximum(f_minus, 0))[FIRST_SCORED_GENE:] / 2
    raw = (f_plus - f_minus)[FIRST_SCORED_GENE:] / 2   # the same contrast without the clip at zero
    return {"context": context, "p": p, "truth_energy": float(np.mean(dy * dy)),
            "contrast_error": float(np.mean((df - dy) ** 2)),
            "prediction_energy": float(np.mean(df * df)),
            "endpoint_raw_mse": [float(np.mean((f[FIRST_SCORED_GENE:] - y[FIRST_SCORED_GENE:].astype(np.float64)) ** 2))
                                 for f, y in zip(predictions, truths)],
            "contrast_error_raw": float(np.mean((raw - dy) ** 2)),
            "prediction_energy_raw": float(np.mean(raw * raw))}


def recovery(rows):
    """Q and its parts over a set of pair rows; Q is undefined when the truth has no contrast.

    Rows from ``pair_row`` also give ``Q_raw``, the same recovery of the unclipped contrast.
    """
    energy = float(np.sum([row["truth_energy"] for row in rows]))
    error = float(np.sum([row["contrast_error"] for row in rows]))
    result = {"n_pairs": len(rows), "Q": 1 - error / energy if energy > 0 else None,
              "truth_halfcontrast_energy": energy / len(rows),
              "halfcontrast_error": error / len(rows),
              "predicted_halfcontrast_energy": float(np.mean([row["prediction_energy"] for row in rows])),
              "status": "defined" if energy > 0 else "null_target_report_absolute_leakage"}
    if rows and all("contrast_error_raw" in row for row in rows):
        raw = float(np.sum([row["contrast_error_raw"] for row in rows]))
        result.update(Q_raw=1 - raw / energy if energy > 0 else None, halfcontrast_error_raw=raw / len(rows),
                      predicted_halfcontrast_energy_raw=float(np.mean([row["prediction_energy_raw"] for row in rows])))
    return result


def support_summary(bit_rows, target_bit, contexts):
    """Recovery per bit and per context, plus raw error on the target bit's endpoints."""
    bits = {str(bit): recovery(rows) | {"pair_losses": rows,
                                        "per_context": {context: recovery([r for r in rows if r["context"] == context])
                                                        for context in contexts}}
            for bit, rows in bit_rows.items()}
    losses = [loss for row in bit_rows[target_bit] for loss in row["endpoint_raw_mse"]]
    return {"bits": bits, "mse": {"raw_O50": float(np.mean(losses)), "n_endpoints": len(losses),
                                  "target_bit": target_bit,
                                  "sampling": "Same endpoints as the support's target-bit pairs"}}


def score_pairs(model, pairs, device):
    """Pair rows for a list of flip pairs."""
    rows = []
    for pair in pairs:
        predictions = (predict(model, pair.minus, device), predict(model, pair.plus, device))
        rows.append(pair_row(pair.minus.context, pair.minus.p, predictions, (pair.minus.truth, pair.plus.truth)))
    return rows


def evaluate(model, task, panels, bits, target_bit, device, panel_name="repair"):
    """Recovery of every bit on every support panel; the model's train/eval mode is restored."""
    import torch

    was_training = model.training
    model.eval()
    result = {"panel": panel_name, "specifications": {support: asdict(spec) for support, spec in panels.items()},
              "bits": list(bits), "supports": {}}
    with torch.inference_mode():
        for support, spec in panels.items():
            bit_rows = {bit: score_pairs(model, flip_pairs(task, spec.root, bit, spec.pairs, spec.contexts,
                                                           spec.pool_group), device)
                        for bit in bits}
            result["supports"][support] = support_summary(bit_rows, target_bit, spec.contexts)
    model.train(was_training)
    return result
