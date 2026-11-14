"""Planted faults: the positive control of information localization, built from parents that learned the selective
shift.

Each fault blocks learning at a known place, so the expected outcome of the localization is fixed by construction.
Five faulted parents and two unfaulted controls, all of seed 31002, on Tahoe-100M fold 1 (State's unfaulted control
is its own parent). Each file records its case and expected outcome under ``known_answer``; ``admissible_answer`` is a
saved description, and the verdict is read by ``tables.READINGS.planted``, which for the two front faults also admits
recovery by the whole network.

Summary-statistics model (``summary_parents``). The donor is the summary-statistics parent (40,000 updates) after
its interaction branch and perturbation embedding were retrained at 3e-3 for 8,000 updates of 64 input sets (a
retraining these commands do not include); it has learned the selective shift.

- ``solved``: the donor as it is (an unfaulted control);
- ``dead-interaction``: every hidden unit of the interaction branch switched off by a bias of -1000, so that no
  gradient reaches that branch. The obstacle is behind the cut ``projections``;
- ``blocked-mean-cue``: the mean projection's weights on the target's two cue genes (2 and 3; bit a's cue is written
  into genes 2a and 2a + 1) set to zero. The
  variance projection reads per-gene variance, which the planted mean cue does not change, so nothing behind
  the cut ``projections`` receives the cue. The obstacle is in the front, in the mean projection;
- ``blocked-variance-cue`` (``variance_parent``, from the saved ``solved`` parent): the variance projection's
  weights on the shared-spread cue genes (10 and 11, bit 5) set to zero. The variance bit leaves each gene's
  mean unchanged, so the mean projection does not see it; the obstacle is in the front, in the variance
  projection. Read on bit 5.

DeepSets model (``deepsets_parents``). The donor is the map run that learned the selective shift
(``map-deepsets-synthetic-31002-input-a-x3``, ``tahoe.run``: the whole network at x3).

- ``ds-solved``: the donor as it is (an unfaulted control);
- ``ds-dead-back``: every hidden unit of ``rho`` switched off by a bias of -1000 on ``rho.0`` (ReLU), so that no
  gradient reaches the back or anything before it. The obstacle is behind the cut ``phi``;
- ``ds-blocked-cue``: the per-cell encoder's weights on the target's two cue genes (2 and 3) set to zero.
  ``phi.0`` reads ``cat(x / U, log-depth)``, so its input columns 0-1999 are the genes in panel order; ``phi`` is
  the only path from genes to the network. The obstacle is in the front.

Each file keeps the donor's loader fields (model, seed, level, instance, parameters) with step 40000, the base
parent's step, so that ``tahoe.load_parent`` reads it. The map runs' ``EVAL_PARENT.json`` holds each parent's scores
on the map's panels.

    python -m reproduce.information_localization.planted_faults --summary-donor PT --summary-base PT \\
        --deepsets-donor PT --instance DATA/tahoe-100m/F1/synthetic --out FOLDER
"""
from __future__ import annotations

from pathlib import Path

from scsailor.localization.constants import LEVEL
from scsailor.localization.starts import PARENT_STEP

from . import tahoe

__all__ = ["SEED", "summary_parents", "deepsets_parents", "variance_parent", "write", "main"]

SEED = 31002
MEAN_CUE_GENES = (2, 3)        # bit 1, the selective shift
VARIANCE_CUE_GENES = (10, 11)  # bit 5, the shared spread
DEAD_BIAS = -1000.0
ADMISSIBLE = {
    "solved": "read as solved at the start from the saved evaluation of this parent, so no obstacle is located; a "
              "loss of the target after training is training-induced damage",
    "dead-interaction": "on matched rates, the back contrast at 'projections' reads 'reset benefit'; drug side, front, "
                        "mean projection and variance projection do not",
    "blocked-mean-cue": "on matched rates, the front and mean-projection contrasts at 'projections' read 'reset "
                        "benefit'; variance projection, back and drug side do not",
    "blocked-variance-cue": "read on the common variance (bit 5): on matched rates, the front and "
                            "variance-projection contrasts at 'projections' read 'reset benefit'; mean projection, "
                            "back and drug side do not",
    "ds-solved": "read as solved at the start from the saved evaluation of this parent, so no obstacle is located",
    "ds-dead-back": "on matched rates, the back contrast at 'phi' reads 'reset benefit'; drug side and front do not",
    "ds-blocked-cue": "no back or drug-side reset benefit at 'phi'; the front contrast reads 'reset benefit', or no "
                      "contrast at 'phi' reads 'reset benefit' while the whole network recovers",
}
FAULT = {"solved": None, "dead-interaction": f"backbone.inter_hidden.0.bias = {DEAD_BIAS}",
         "blocked-mean-cue": f"backbone.sproj.weight[:, {list(MEAN_CUE_GENES)}] = 0",
         "blocked-variance-cue": f"backbone.hproj.weight[:, {list(VARIANCE_CUE_GENES)}] = 0",
         "ds-solved": None, "ds-dead-back": f"backbone.rho.0.bias = {DEAD_BIAS}",
         "ds-blocked-cue": f"backbone.phi.0.weight[:, {list(MEAN_CUE_GENES)}] = 0"}


def _load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _dead(state, bias):
    """``state`` with every unit behind the bias ``bias`` switched off."""
    import torch

    return dict(state, **{bias: torch.full_like(state[bias], DEAD_BIAS)})


def _blocked(state, weight, genes):
    """``state`` with the input columns ``genes`` of the layer ``weight`` set to zero."""
    blocked = state[weight].clone()
    blocked[:, list(genes)] = 0
    return dict(state, **{weight: blocked})


def summary_parents(donor_path, base_path):
    """The three summary-statistics parents built from the retrained base parent, by case."""
    import torch

    donor = _load(donor_path)
    if donor["identity"]["checkpoint_identity"] != {"model_name": "summary", "seed": SEED, "level": LEVEL,
                                                    "step": PARENT_STEP}:
        raise ValueError(f"{donor_path} is not retrained from the seed-{SEED} summary-statistics parent")
    base = _load(base_path)
    trained = ("backbone.inter_hidden.", "backbone.w1.", "backbone.pert.")
    changed = [k for k, v in base["model"].items()
               if not k.startswith(trained) and not torch.equal(v, donor["model"][k])]
    if changed:
        raise ValueError(f"{donor_path} differs from {base_path} outside the retrained parts: {changed}")
    weights = {"solved": dict(donor["model"]),
               "dead-interaction": _dead(donor["model"], "backbone.inter_hidden.0.bias"),
               "blocked-mean-cue": _blocked(donor["model"], "backbone.sproj.weight", MEAN_CUE_GENES)}
    return {case: {"model": state, "model_name": "summary", "seed": SEED, "level": LEVEL, "step": PARENT_STEP,
                   "instance": base["instance"], "parameters": base["parameters"],
                   "known_answer": {"case": case, "admissible_answer": ADMISSIBLE[case], "donor": str(donor_path),
                                    "base_parent": str(base_path),
                                    "step_field": "the base parent's step; the repair added 8,000 updates of 64 bags",
                                    "fault": FAULT[case]}}
            for case, state in weights.items()}


def variance_parent(solved_path):
    """The blocked-variance-cue parent, built from the saved summary-statistics ``solved`` parent."""
    case = "blocked-variance-cue"
    donor = _load(solved_path)
    found = (donor["model_name"], donor["seed"], donor["level"], donor["known_answer"]["case"])
    if found != ("summary", SEED, LEVEL, "solved"):
        raise ValueError(f"{solved_path} is not the seed-{SEED} summary-statistics solved parent")
    saved = {key: donor[key] for key in ("model_name", "seed", "level", "step", "instance", "parameters")}
    saved["model"] = _blocked(donor["model"], "backbone.hproj.weight", VARIANCE_CUE_GENES)
    saved["known_answer"] = dict(donor["known_answer"], case=case, admissible_answer=ADMISSIBLE[case],
                                 fault=FAULT[case], donor=str(solved_path))
    return saved


def deepsets_parents(donor_path, instance):
    """The three DeepSets parents built from the map run that learned the selective shift, by case; ``instance`` is
    the fold's synthetic-answer instance directory."""
    from scsailor.assay.instance import Instance

    donor = _load(donor_path)
    if (donor["model_name"], donor["seed"], donor["level"]) != ("deepsets", SEED, LEVEL):
        raise ValueError(f"{donor_path} is not a seed-{SEED} DeepSets +Var run")
    tahoe.check_synthetic(donor, Instance(instance), donor_path)
    weights = {"ds-solved": dict(donor["model"]), "ds-dead-back": _dead(donor["model"], "backbone.rho.0.bias"),
               "ds-blocked-cue": _blocked(donor["model"], "backbone.phi.0.weight", MEAN_CUE_GENES)}
    return {case: {"model": state, "model_name": "deepsets", "seed": SEED, "level": LEVEL, "step": PARENT_STEP,
                   "instance": donor["instance"], "parameters": donor["parameters"],
                   "known_answer": {"case": case, "admissible_answer": ADMISSIBLE[case], "donor": str(donor_path),
                                    "step_field": "the base parent's step; the donor's map run added 40,000 updates "
                                                  "of 8 bags", "fault": FAULT[case]}}
            for case, state in weights.items()}


def write(folder, summary_donor, summary_base, deepsets_donor, instance):
    """Build the seven parents into ``folder``; the blocked-variance-cue parent from the solved parent written there."""
    import torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for case, saved in summary_parents(summary_donor, summary_base).items():
        torch.save(saved, folder / f"summary_{SEED}_{case}.pt")
    torch.save(variance_parent(folder / f"summary_{SEED}_solved.pt"),
               folder / f"summary_{SEED}_blocked-variance-cue.pt")
    for case, saved in deepsets_parents(deepsets_donor, instance).items():
        torch.save(saved, folder / f"deepsets_{SEED}_{case}.pt")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Build the five planted-fault parents and the two unfaulted controls.")
    parser.add_argument("--summary-donor", type=Path, required=True,
                        help="the summary-statistics parent with its interaction branch and embedding retrained")
    parser.add_argument("--summary-base", type=Path, required=True, help="the summary-statistics parent it came from")
    parser.add_argument("--deepsets-donor", type=Path, required=True,
                        help="the DeepSets map run that learned the selective shift (its FINAL.pt)")
    parser.add_argument("--instance", type=Path, required=True, help="Tahoe-100M fold 1's synthetic-answer instance")
    parser.add_argument("--out", type=Path, required=True, help="folder for the parents")
    args = parser.parse_args(argv)
    write(args.out, args.summary_donor, args.summary_base, args.deepsets_donor, args.instance)


if __name__ == "__main__":
    main()
