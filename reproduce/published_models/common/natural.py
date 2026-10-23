"""The natural task for the published models: training data, the paired training stream and prediction.

A fold directory holds the files of ``docs/data_contract.md`` (natural task, published models). Training
draws its control cells exactly as the built models' stream (``scsailor.natural.data``): eight units per
update by weight, 256 control cells per unit, and the level's carrier. The published models learn from
cells, so each unit is also paired with 256 of its treated cells, drawn without replacement (with
replacement when a unit has fewer than 256) by one generator seeded ``[seed, 9300]`` and consumed in
update order.

Prediction gives every context of the training menu and of the held-out menu one fixed evaluation bag. Its
carrier is drawn 16 times at +Var, +Dist and +DepCp and once otherwise (seed key
``[41001, context index, draw, 777]``); each perturbation's prediction is the mean over 256 predicted cells,
averaged over the draws.
"""
from pathlib import Path

import h5py
import numpy as np
import torch

from scsailor.carrier import make_atoms
from scsailor.natural.data import NaturalData, ordered_batches

PAIRING_KEY = 9300
CELLS = 256
CHUNK = 8   # perturbations per forward in prediction


class TrainingSet(NaturalData):
    """The training units of one fold, their control pools and their treated cells.

    With ``features`` (PrePR-CT's ``drug_features.npz``), perturbations without descriptors (``applicable``
    false) are left out: their units are dropped and the unit weights renormalized, as PrePR-CT cannot
    represent them. ``features``, ``feature_row`` and ``r_row`` then give the descriptor rows."""

    def __init__(self, fold, features=None):
        self.fold = Path(fold)
        with np.load(self.fold / "training_constants.npz", allow_pickle=False) as z:
            self.contexts = z["context_keys"].astype(str)
            self.r = z["r_input"]
        with np.load(self.fold / "training_targets.npz", allow_pickle=False) as z:
            target, parent, codes, weight = (z[key] for key in ("target", "parent", "p_code", "weight"))
            self.intercept = z["intercept"]
            self.p_ids = z["p_ids"].astype(str)
        self.p_count = len(self.p_ids)
        with np.load(self.fold / "treated_index.npz", allow_pickle=False) as z:
            start, count = z["start"], z["count"]
        units = np.arange(len(target))
        self.applicable = None
        if features is not None:
            with np.load(features, allow_pickle=False) as z:
                if not np.array_equal(z["p_ids"].astype(str), self.p_ids):
                    raise ValueError("drug_features.npz lists the perturbations in another order")
                applicable = z["applicable"]
                self.features = z["normalized"][applicable]
            self.feature_row = np.full(self.p_count, -1, dtype=np.int64)
            self.feature_row[applicable] = np.arange(applicable.sum())
            self.r_row = np.arange(self.p_count)
            if not applicable.all():
                self.applicable = applicable
                keep = applicable[codes]
                target, parent, codes, units = target[keep], parent[keep], codes[keep], units[keep]
                weight = weight[keep] / weight[keep].sum()
        self.target, self.parent, self.pcode, self.prob = target, parent, codes, weight
        self.treated_start, self.treated_count = start[units], count[units]
        cells = np.load(self.fold / "treated_cells.npy", mmap_mode="r")
        self.treated = cells.reshape(-1, cells.shape[-1])   # (units, 512, genes) files are read as rows
        with h5py.File(self.fold / "controls.h5", "r") as bank:
            self.pools = {int(c): (bank[str(c)]["input/x"][:], bank[str(c)]["input/depth"][:])
                          for c in np.unique(self.parent)}

    def selected_rows(self, step, seed):
        """The eight units of update ``step``, as drawn by ``NaturalData.batch``."""
        return np.random.default_rng(np.random.SeedSequence([seed, step, 0])).choice(
            len(self.target), size=8, replace=True, p=self.prob)

    def menus(self):
        """Prediction menus: per side, the contexts, perturbation codes and (training side) unit weights."""
        with np.load(self.fold / "heldout_prediction_units.npz", allow_pickle=False) as z:
            context, codes = z["context"].astype(str), z["p_code"]
        if self.applicable is not None:
            keep = self.applicable[codes] & np.isin(codes, np.unique(self.pcode))
            context, codes = context[keep], codes[keep]
        return {"train": (self.contexts[self.parent], self.pcode, self.prob), "heldout": (context, codes, None)}


class PairedStream:
    """Training batches: control cells (8 x 256 x genes), treated cells (2048 x genes) and codes (8)."""

    def __init__(self, data, level, seed):
        self.data, self.level, self.seed = data, level, seed
        self.pair_rng = np.random.default_rng(np.random.SeedSequence([seed, PAIRING_KEY]))

    def batches(self, first, stop, workers):
        for step, (x, _depth, codes, _labels), _waited in ordered_batches(self.data, first, stop, CELLS, self.level,
                                                                          self.seed, workers):
            rows = self.data.selected_rows(step, self.seed)
            target = np.concatenate([
                self.data.treated[start + self.pair_rng.choice(count, size=CELLS, replace=count < CELLS)]
                for start, count in zip(self.data.treated_start[rows], self.data.treated_count[rows])])
            yield step, dict(atoms=np.ascontiguousarray(x[:, ::2]), target=target, codes=np.asarray(codes))

    def state_dict(self):
        return dict(pair_rng=self.pair_rng.bit_generator.state)

    def load_state_dict(self, state):
        self.pair_rng.bit_generator.state = state["pair_rng"]


def predict(adapter, model, data, *, level, step, output):
    """Write ``PREDICTION_INPUT_IDS.npz`` and ``PREDICTIONS_step{step}.npz`` (per side: raw and clipped
    predictions, contexts, codes and the evaluation draws used)."""
    output = Path(output)
    menu = data.menus()
    draws = 16 if level in ("L2", "L3", "L3d") else 1
    contexts = np.concatenate([value[0] for value in menu.values()])
    codes = np.concatenate([value[1] for value in menu.values()])
    sides = np.concatenate([np.repeat(side, len(value[0])) for side, value in menu.items()])
    references = np.empty((len(contexts), draws), dtype=np.int64)
    records, cell_ids, keys = [], [], []
    with h5py.File(data.fold / "evaluation_inputs.h5", "r") as bank:
        for side, (side_contexts, _, _) in menu.items():
            bank_side = "heldout" if side == "heldout" else "train"
            order = sorted(bank[bank_side])
            for context in sorted(set(side_contexts)):
                rows = np.flatnonzero((sides == side) & (contexts == context))
                group = bank[bank_side][context]
                ordered = group["indices"][:][np.argsort(group["depth"][:], kind="stable")]
                for draw in range(draws):
                    references[rows, draw] = len(records)
                    records.append(f"{side}/{context}/draw{draw}")
                    cell_ids.append(ordered)
                    keys.append([41001, order.index(context), draw, 777])
    np.savez(output / "PREDICTION_INPUT_IDS.npz", evaluation_bag_draw=np.asarray(records),
             atom_indices=np.asarray(cell_ids), carrier_keys=np.asarray(keys), context=contexts, p_code=codes,
             side=sides, input_record=references,
             index_definition=np.asarray("Depth-sorted cell indices of each evaluation bag; the carrier rows of "
                                         "a draw follow from its carrier key"))
    raw = np.zeros((len(contexts), 2000), dtype=np.float64)
    with h5py.File(data.fold / "evaluation_inputs.h5", "r") as bank, torch.no_grad():
        for side, (side_contexts, _, _) in menu.items():
            bank_side = "heldout" if side == "heldout" else "train"
            for context in sorted(set(side_contexts)):
                rows = np.flatnonzero((sides == side) & (contexts == context))
                unique = np.unique(codes[rows])
                group = bank[bank_side][context]
                source, depth = group["x"][:], group["depth"][:]
                for draw in range(draws):
                    key = keys[references[rows[0], draw]]
                    atoms, _ = make_atoms(source, depth, level, np.random.default_rng(np.random.SeedSequence(key)))
                    prepared = adapter.prepare(atoms)
                    values = [adapter.predict_chunk(model, prepared, unique[start:start + CHUNK]).mean(1, dtype=np.float64)
                              for start in range(0, len(unique), CHUNK)]
                    raw[rows] += np.concatenate(values)[np.searchsorted(unique, codes[rows])]
    raw /= draws
    arrays = {}
    for side, (side_contexts, side_codes, _) in menu.items():
        rows = sides == side
        arrays.update({f"{side}_raw": raw[rows], f"{side}_projected": np.maximum(raw[rows], 0),
                       f"{side}_context": side_contexts, f"{side}_p_code": side_codes,
                       f"{side}_input_record": references[rows]})
    np.savez(output / f"PREDICTIONS_step{step}.npz", **arrays, technical_draws=np.asarray(draws))
