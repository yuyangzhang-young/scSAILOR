"""The synthetic-answer assay for the published models: training stream, replay and scores.

The published models learn from cells, not from mean profiles. For each bag of the assay's training stream
(``scsailor.assay.stream``), the treated cells are 256 control cells of the bag's context, drawn with
seed key ``[seed, step, slot, 5]``, plus the bag's planted response (the generator's shift for its
perturbation and state), clipped at zero. The model input is every second of the bag's 512 carrier rows,
that is, its 256 distinct cells.

Replay writes the assay's prediction layout and ``scsailor.assay.score.score_predictions`` scores it.
"""
from concurrent.futures import ThreadPoolExecutor
import json

import h5py
import numpy as np
import torch

from scsailor.assay.carrier import carrier
from scsailor.assay.constants import N_GENES, REPLAY_DRAW_KEY
from scsailor.assay.instance import HELDOUT, SEEN
from scsailor.assay.stream import BAGS, TrainingStream

TREATED_CELLS_KEY = 5   # [seed, step, slot, 5]: which control cells become the bag's treated cells
CELLS = 256             # distinct cells per bag
DRAWS = {"L1": 1, "L2": 16, "L3": 16, "L3d": 16, "L4": 16}   # replay draws per level
CONFIGURATIONS = {HELDOUT: list(range(26)), SEEN: [0]}       # base state and its 25 flips; seen side: base only


class TreatedCellStream:
    """Training batches of one fit: control cells, treated cells and perturbation codes per bag."""

    def __init__(self, instance, level, seed):
        self.assay = TrainingStream(instance)
        self.level, self.seed = level, seed

    def _draws(self, step):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, step, 0]))
        contexts = rng.choice(len(self.assay.pools), size=BAGS, replace=True)
        codes = rng.choice(len(self.assay.generator.p_ids), size=BAGS, p=self.assay.generator.pi)
        states = np.stack([np.random.default_rng(np.random.SeedSequence([self.seed, step, slot, 3]))
                           .choice([-1, 1], size=25) for slot in range(BAGS)])
        return contexts, codes, states

    def batch(self, step):
        x, _, codes, _ = self.assay.batch(step, CELLS, self.level, self.seed)
        contexts, expected_codes, states = self._draws(step)
        if not np.array_equal(codes, expected_codes):
            raise RuntimeError("The assay stream and the treated cells disagree on the perturbations")
        shifts = self.assay.generator.shift(codes, states)
        targets = []
        for slot, context in enumerate(contexts):
            pool, _ = self.assay.pools[context]
            selected = np.random.default_rng(np.random.SeedSequence([self.seed, step, slot, TREATED_CELLS_KEY])) \
                .choice(len(pool), size=CELLS, replace=False)
            targets.append(np.maximum(pool[selected].astype(np.float64) + shifts[slot], 0).astype(np.float32))
        return dict(atoms=np.ascontiguousarray(x[:, ::2]), target=np.stack(targets), codes=codes)

    def batches(self, first, stop, workers):
        """``(step, batch)`` for ``first`` ... ``stop``, computed ahead by worker threads; batches depend only on
        the step, so their content does not depend on the number of workers."""
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {step: executor.submit(self.batch, step) for step in range(first, min(stop + 1, first + 2 * workers))}
            for step in range(first, stop + 1):
                batch = pending.pop(step).result()
                if step + 2 * workers <= stop:
                    pending[step + 2 * workers] = executor.submit(self.batch, step + 2 * workers)
                yield step, batch

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def replay_bag(predict, x, depth, states, root, q, level, params, n_codes, configurations):
    """Raw predictions and baselines (bases x configurations x codes x genes, float32) of one evaluation bag.

    For each base state and configuration, the carrier is drawn ``DRAWS[level]`` times with key
    ``[root, q, base, draw, 888]``; ``predict(cells)`` returns float64 (codes x genes) predictions and
    baselines for the bag's 256 distinct cells, and the draws are averaged in float64."""
    draws = DRAWS[level]
    raw = np.empty((len(states), len(configurations), n_codes, N_GENES), dtype=np.float32)
    baseline = np.empty_like(raw)
    with torch.inference_mode():
        for base in range(len(states)):
            for j, configuration in enumerate(configurations):
                raw_sum = np.zeros((n_codes, N_GENES), dtype=np.float64)
                baseline_sum = np.zeros_like(raw_sum)
                for draw in range(draws):
                    rng = np.random.default_rng(np.random.SeedSequence([root, q, base, draw, REPLAY_DRAW_KEY]))
                    rows, _ = carrier(x, depth, states[base, configuration], level, rng, params)
                    values, base_values = predict(np.ascontiguousarray(rows[::2]))
                    raw_sum += values
                    baseline_sum += base_values
                raw[base, j] = (raw_sum / draws).astype("f4")
                baseline[base, j] = (baseline_sum / draws).astype("f4")
    return raw, baseline


def write_predictions(predict, instance, *, level, seed, step, checkpoint, output, codes=None):
    """Replay every evaluation bag of both sides into one HDF5 file in the assay's layout.

    ``codes`` restricts the perturbations (they are then stored as a ``codes`` dataset); by default all
    perturbations are replayed."""
    n_codes = len(instance.generator.p_ids) if codes is None else len(codes)
    with h5py.File(output, "w-") as out:
        for key, value in dict(checkpoint=str(checkpoint), instance=instance.name, level=level, seed=seed,
                               parameters=json.dumps(instance.record["parameters"], sort_keys=True),
                               step=step, draws=DRAWS[level]).items():
            out.attrs[key] = value
        if codes is not None:
            out.create_dataset("codes", data=np.asarray(codes, dtype=np.int64))
        for side, configurations in CONFIGURATIONS.items():
            for entry in instance.evaluation[side]:
                x, depth = instance.bag(side, entry["context"])
                states = instance.states(side, entry["context"])
                raw, baseline = replay_bag(predict, x, depth, states, entry["root"], entry["q"], level,
                                           instance.params, n_codes, configurations)
                group = out.create_group(f"{side}/{entry['context']}")
                for field, values in (("raw", raw), ("anchor_raw", baseline)):
                    dataset = group.create_dataset(field, shape=(len(states), 26, n_codes, N_GENES), dtype="f4",
                                                   chunks=(1, 1, n_codes, N_GENES), compression="lzf")
                    dataset[:, configurations[0]:configurations[-1] + 1] = values
                done = np.zeros((len(states), 26), dtype=bool)
                done[:, configurations] = True
                group.create_dataset("done", data=done)
                out.flush()


def _ratios(E, M, A, K):
    return {"Q": 1 - M / E if E > 0 else None, "kappa": K / (2 * E) if E > 0 else None}


def score_subset(instance, predictions, output):
    """Scores of a replay restricted to the instance's ``EVALUATION_SUBSET.json``; writes ``SCORES.json``.

    For instances whose full table of synthetic targets would be too large to store (Replogle–Nadig: 1,966
    perturbations), the targets of the subset are recomputed one base state at a time from the context's pool.
    The formulas are those of ``scsailor.assay.score``: per planted state, the true half-contrast energy E, the
    error energy M, the predicted contrast energy A and twice the cross term K, averaged over base states and
    contexts, and Q = 1 - M / E. Also reported: the standard error of Q from scoring n of N perturbations,
    sqrt((1 - n/N) / n) * sd_p(m_p - (1 - Q) e_p) / mean_p(e_p)."""
    from scsailor.assay.generator import Background, known_answer

    first = 50
    subset = json.loads((instance.inputs / "EVALUATION_SUBSET.json").read_text())
    codes, n_all = np.asarray(subset["codes"]), subset["N"]
    scores = {}
    with h5py.File(predictions, "r") as handle:
        if not np.array_equal(handle["codes"][:], codes):
            raise ValueError(f"{predictions} was not replayed on the evaluation subset")
        for side in (HELDOUT, SEEN):
            per_context, per_perturbation = {}, []
            for entry in instance.evaluation[side]:
                group = handle[f"{side}/{entry['context']}"]
                states = instance.states(side, entry["context"])
                background = Background(instance.pool("train" if side == SEEN else HELDOUT, entry["context"])[0])
                full = bool(group["done"][:].all())
                sums = np.zeros((25, 5))
                pp = np.zeros((25, 2, len(codes)))
                base = {key: [] for key in ("R", "R_common", "R_contrast", "anchor_R")}
                for i in range(len(states)):
                    configurations = list(range(26)) if full else [0]
                    truth = np.stack([known_answer(instance.generator, background, codes,
                                                   np.tile(states[i, c], (len(codes), 1))) for c in configurations])
                    raw = group["raw"][i][configurations].astype(np.float64)
                    anchor = group["anchor_raw"][i, 0].astype(np.float64)
                    pred = np.maximum(raw, 0)
                    error = pred[0, :, first:] - truth[0, :, first:]
                    center = error.mean(axis=0, keepdims=True)
                    base["R"].append(np.mean(error ** 2))
                    base["R_common"].append(np.mean(center ** 2))
                    base["R_contrast"].append(np.mean((error - center) ** 2))
                    base["anchor_R"].append(np.mean((np.maximum(anchor, 0)[:, first:] - truth[0, :, first:]) ** 2))
                    if full:
                        for a in range(25):
                            s = states[i, 0, a]
                            t = s * (truth[0, :, first:] - truth[a + 1, :, first:]) / 2
                            f = s * (pred[0, :, first:] - pred[a + 1, :, first:]) / 2
                            c = (pred[0, :, first:] + pred[a + 1, :, first:] - truth[0, :, first:] - truth[a + 1, :, first:]) / 2
                            e_p, m_p = np.mean(t * t, axis=1), np.mean((f - t) ** 2, axis=1)
                            sums[a] += (e_p.mean(), m_p.mean(), np.mean(f * f), 2 * np.mean(f * t), np.mean(c * c))
                            pp[a, 0] += e_p
                            pp[a, 1] += m_p
                result = {key: float(np.mean(value)) for key, value in base.items()}
                if full:
                    means = sums / len(states)
                    result["modules"] = [{"module": a, "E": E, "M": M, "A": A, "K": K, "C_pair": C, **_ratios(E, M, A, K)}
                                         for a, (E, M, A, K, C) in enumerate(means)]
                    per_perturbation.append(pp / len(states))
                per_context[entry["context"]] = result
            summary = {key: float(np.mean([c[key] for c in per_context.values()]))
                       for key in ("R", "R_common", "R_contrast", "anchor_R")}
            if per_perturbation and len(per_perturbation) == len(per_context):
                pert = np.mean(per_perturbation, axis=0)   # 25 x (e, m) x n, contexts pooled with equal weights
                summary["modules"] = []
                for a in range(25):
                    row = {key: float(np.mean([c["modules"][a][key] for c in per_context.values()]))
                           for key in ("E", "M", "A", "K", "C_pair")}
                    row.update(_ratios(row["E"], row["M"], row["A"], row["K"]))
                    e, m = pert[a]
                    if row["Q"] is not None and len(codes) > 1:
                        d = m - (1 - row["Q"]) * e
                        row["Q_SE_subset"] = float(np.sqrt((1 - len(codes) / n_all) / len(codes)) * d.std(ddof=1) / e.mean())
                    summary["modules"].append({"module": a, **row})
            summary["contexts"] = per_context
            scores[side] = summary
    record = {"instance": instance.name, "subset": {key: subset[key] for key in ("n", "N", "rule", "seed")},
              "predictions": str(predictions), **scores}
    (output / "SCORES.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
