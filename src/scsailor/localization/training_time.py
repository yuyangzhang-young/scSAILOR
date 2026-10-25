"""Training time: when, in a parent's own training, the planted statistics are learned.

The map reads a parent at one length (40k updates). Its own training run saved later checkpoints too; scoring
them on the same crossed panels shows whether the target is learned later under the parent's own recipe, and what
happens to the statistics it had already learned. ``score`` reads one checkpoint: recovery of bits 0 (common mean),
1 (drug-specific mean, the target), 5 (common variance) and 6 (drug-specific variance) on every panel, with the raw
error on the target's endpoints (``recovery.evaluate``).

``time_axis`` turns the scored checkpoints into the rows of ``TIME_AXIS.csv``: per model, seed, step and support,
Q of the four bits, the matched (a) and mismatched (o) amplitudes of the common-mean response, and the median line
Q of the common mean. With P the predicted and E the true half-contrast energy and M the contrast error, the
predicted contrast is a times the true one plus a part orthogonal to it of energy o^2 E:

    a = (P + E - M) / (2E),   o = sqrt(max(P / E - a^2, 0)),   so that Q = 1 - (1 - a)^2 - o^2.
"""
import csv
import math

from .constants import BITS, TARGET_BIT
from .readings import per_line
from .recovery import evaluate

__all__ = ["score", "split", "median_line", "time_axis", "write_time_axis"]


def score(model, task, panels, panel_name, device="cpu"):
    """Recovery of ``BITS`` on every panel of ``panels`` (``recovery.PanelSpec`` by support)."""
    return evaluate(model, task, panels, BITS, TARGET_BIT, device, panel_name=panel_name)


def split(bit):
    """The matched (a) and mismatched (o) amplitudes of one bit's predicted contrast."""
    P, E, M = bit["predicted_halfcontrast_energy"], bit["truth_halfcontrast_energy"], bit["halfcontrast_error"]
    a = (P + E - M) / (2 * E)
    return a, math.sqrt(max(P / E - a * a, 0.0))


def median_line(bit):
    """The median over held-out lines of one bit's per-line Q (both plates of a line pooled)."""
    values = sorted(1 - m / e for m, e in per_line(bit).values())
    n = len(values)
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2


def time_axis(evaluations):
    """The rows of ``TIME_AXIS.csv`` from ``evaluations``, pairs of ``(model, seed, step)`` and that checkpoint's
    ``score``, sorted by model, seed, support and step."""
    rows = []
    for (model, seed, step), result in evaluations:
        for support, values in result["supports"].items():
            bits = values["bits"]
            a, o = split(bits["0"])
            rows.append({"model": model, "seed": int(seed), "step": int(step), "support": support,
                         **{f"Q{b}": round(bits[b]["Q"], 4) for b in ("0", "1", "5", "6")},
                         "common_matched": round(a, 4), "common_mismatched": round(o, 4),
                         "common_median_line_Q": round(median_line(bits["0"]), 4)})
    rows.sort(key=lambda r: (r["model"], r["seed"], r["support"], r["step"]))
    return rows


def write_time_axis(path, rows):
    """Write ``TIME_AXIS.csv``."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
