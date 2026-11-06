"""scSAILOR on your own data and your own model: the three components in a few minutes on a CPU.

Replace ``your_data`` with your control cells and measured responses, ``YourModel`` with your network and
``your_predictions`` with your model's predictions. The budgets here are a few hundred updates, so the model learns
little and the recoveries stay near zero: the run shows what each step takes and returns, not what a trained model
reads. Localization trains each starting point with ``starts.build_arm`` and ``starts.train``, because ``starts.run``,
which a full run calls, fixes the length of a run. ``examples/README.md`` describes the inputs and how a full run
differs.

    python examples/quickstart.py --out quickstart-run
"""
import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from scsailor.assay import Instance, generator, replay, score, train
from scsailor.assay.constants import FIRST_SCORED_GENE, N_GENES
from scsailor.assay.instance import HELDOUT, SEEN
from scsailor.assay.stream import TrainingStream
from scsailor.localization import contract, starts
from scsailor.localization.constants import ADAMW, BITS, LEVEL, TARGET_BIT
from scsailor.localization.draws import training_batch
from scsailor.localization.recovery import PanelSpec, evaluate
from scsailor.localization.runtime import set_precision
from scsailor.localization.task import Task
from scsailor.resolution import net_cost

SEEN_ROOT, HELDOUT_ROOT = 51001, 41001   # the evaluation support's seed roots (docs/data_contract.md)
SEEDS = (1, 2, 3)
CUES = ("shift", "spread", "skew", "depth trend", "covariation")             # planted state a: cue a // 5
RESPONSES = ("shared", "selective", "weak shared", "weak selective", "null")  # and response a % 5


# 1. Your data ---------------------------------------------------------------------------------------------------------

def your_data(rng, cells=300, perturbations=6, train_contexts=("c0", "c1", "c2", "c3"), heldout_contexts=("h0", "h1")):
    """Stand-in for your screen: replace the random arrays with your own.

    - ``controls``: context -> (control cells x 2,000 genes, log1p(CP10k); each cell's sequencing depth), at least 256
      cells per context. A context is one background of control cells, for example a cell line on one plate. The
      planted states overwrite genes 0-49 of every cell, so put genes you do not need to score first.
    - ``responses``: the measured responses of the training contexts, one row per unit (a context under one
      perturbation): ``target`` (the unit's mean treated profile), ``context`` (index into ``train_contexts``),
      ``p_code`` (perturbation index) and ``weight`` (summing to one). The response field of the synthetic target is
      calibrated on them.
    """
    controls = {c: (rng.gamma(2., .5, size=(cells, N_GENES)).astype(np.float32), rng.integers(1000, 20000, size=cells))
                for c in train_contexts + heldout_contexts}
    means = np.stack([controls[c][0].mean(axis=0, dtype=np.float64) for c in train_contexts])
    context = np.repeat(np.arange(len(train_contexts)), perturbations)
    p_code = np.tile(np.arange(perturbations), len(train_contexts))
    target = (means[context] + rng.normal(0, .3, size=(perturbations, N_GENES))[p_code]
              + rng.normal(0, .05, size=(len(context), N_GENES)))
    return {"controls": controls, "train_contexts": list(train_contexts), "heldout_contexts": list(heldout_contexts),
            "perturbations": [f"p{i}" for i in range(perturbations)],
            "responses": {"target": target, "context": context, "p_code": p_code,
                          "weight": np.full(len(context), 1 / len(context))}}


def write_instance(directory, data, name="quickstart"):
    """A synthetic-answer instance (docs/data_contract.md) from your data, with the package's own arithmetic."""
    directory.mkdir(parents=True)
    train_contexts, heldout_contexts, controls = data["train_contexts"], data["heldout_contexts"], data["controls"]
    pool_path = {**{c: f"train/{c}" for c in train_contexts}, **{c: f"{HELDOUT}/{c}" for c in heldout_contexts}}
    with h5py.File(directory / "input_pools.h5", "w") as pools:
        for context, path in pool_path.items():
            pools[f"{path}/x"] = controls[context][0].astype(np.float64)
            pools[f"{path}/depth"] = controls[context][1]
    means = np.stack([controls[c][0].mean(axis=0, dtype=np.float64) for c in train_contexts])
    r = data["responses"]
    count = len(data["perturbations"])
    pi = generator.perturbation_weights(r["p_code"], r["weight"], count)
    field = generator.response_field(r["target"], r["context"], r["p_code"], r["weight"], means, count)
    fields, _ = generator.calibrate(field, pi, means)
    np.savez(directory / "generator.npz", **fields, intercept=np.zeros(N_GENES, np.float32),
             p_ids=np.array(data["perturbations"]), context_keys=np.array(train_contexts))
    # The evaluation support: every held-out context, and the last eight training contexts (all, when fewer); q is the
    # context's index in its list (F1_contexts or train_contexts).
    evaluation = {HELDOUT: [{"context": c, "q": heldout_contexts.index(c), "root": HELDOUT_ROOT} for c in heldout_contexts],
                  SEEN: [{"context": c, "q": train_contexts.index(c), "root": SEEN_ROOT} for c in train_contexts[-8:]]}
    for entries in evaluation.values():
        for entry in entries:
            entry.update(pool_path=pool_path[entry["context"]], bag_key=[entry["root"], entry["q"], 999])
    parameters = generator.Parameters().record()
    (directory / "INSTANCE.json").write_text(json.dumps(
        {"instance": name, "parameters": parameters, "train_contexts": train_contexts,
         "F1_contexts": heldout_contexts, "evaluation": evaluation}, indent=1))
    instance = Instance(directory)
    with h5py.File(directory / "evaluation_bags.h5", "w") as bags, h5py.File(directory / "truth.h5", "w") as truth:
        truth.attrs["instance"], truth.attrs["parameters"] = name, json.dumps(parameters, sort_keys=True)
        for side, entries in evaluation.items():
            for entry in entries:
                x, depth = controls[entry["context"]]
                x = x.astype(np.float64)
                indices = generator.bag_indices(len(depth), entry["bag_key"])
                path = f"{side}/{entry['context']}"
                bags[f"{path}/x"], bags[f"{path}/depth"], bags[f"{path}/indices"] = x[indices], depth[indices], indices
                truth[f"{path}/states"], truth[f"{path}/truth"] = generator.truth_table(
                    generator.Background(x), instance.generator, entry["root"], entry["q"])
    return instance


# 2. Your model --------------------------------------------------------------------------------------------------------

class YourModel(nn.Module):
    """Stand-in for your network. It reads each gene's mean and variance over the input rows and adds a correction to
    the anchor (the rows' mean plus the perturbation's mean offset).

    What the package calls on a model (``scsailor.assay.contract``): ``forward(x, depth, p, level, canonicalize)``,
    ``canonical_rows``, ``anchor`` and the buffer ``r_pool``; localization also resets and copies the perturbation
    embedding ``backbone.pert``. ``CUTS`` divides the network into a front and the part behind each cut, as
    parameter-name prefixes (``scsailor.localization.contract.CUTS``).
    """
    CUTS = {"input": ((), None),
            "encoder": (("backbone.encoder",), ("backbone.pert", "backbone.head"))}

    def __init__(self, r_input, c_train, hidden=64):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.encoder = nn.Sequential(nn.Linear(2 * N_GENES, hidden), nn.GELU())
        self.backbone.pert = nn.Embedding(len(r_input), hidden)
        self.backbone.head = nn.Linear(hidden, N_GENES)
        nn.init.zeros_(self.backbone.head.weight)   # the untrained model predicts its anchor
        nn.init.zeros_(self.backbone.head.bias)
        self.register_buffer("r_pool", torch.as_tensor(r_input, dtype=torch.float32))
        self.register_buffer("c_train", torch.as_tensor(c_train, dtype=torch.float32))

    @staticmethod
    def canonical_rows(x, depth):
        order = torch.argsort(depth, dim=1, stable=True)
        return x.gather(1, order[..., None].expand_as(x)), depth.gather(1, order)

    def anchor(self, x, p, level):
        return (self.c_train.expand(len(p), -1) if level == "L0" else x.mean(dim=1)) + self.r_pool[p]

    def forward(self, x, depth, p, level, canonicalize=True):
        if canonicalize:
            x, depth = self.canonical_rows(x, depth)
        hidden = self.backbone.encoder(torch.cat([x.mean(dim=1), x.var(dim=1)], dim=1))
        return self.anchor(x, p, level) + self.backbone.head(hidden * (1 + self.backbone.pert(p)))


def build_model(instance, seed):
    """A fresh network of your architecture for one instance: ``r_pool`` is the instance's response field (one row per
    perturbation) and ``c_train`` the training contexts' control mean."""
    torch.manual_seed(seed)
    return YourModel(instance.generator.r, instance.generator.c_train)


# 3. The synthetic-answer assay ----------------------------------------------------------------------------------------

def assay(instance, out, updates, draws, peak_lr=1e-3):
    """Train the model on the synthetic targets at +Var, replay it on the evaluation support and score each planted
    state's recovery Q on the held-out contexts."""
    model = build_model(instance, SEEDS[0])
    fit = train.Fit(model="your_model", seed=SEEDS[0], level=LEVEL, peak_lr=peak_lr, steps=updates,
                    total=max(updates, train.WARMUP + 1))
    train.train(model, TrainingStream(instance), fit, instance, out / "training", "cpu")
    model.eval()
    replay.write_predictions(model, instance, LEVEL, updates, {HELDOUT: "full", SEEN: "base"}, draws,
                             out / "predictions.h5", "cpu")
    scores, _ = score.score_predictions(instance, out / "predictions.h5", output=out / "scores")
    print(f"\nSynthetic-answer assay, {updates} updates at {LEVEL}: risk on held-out contexts {scores[HELDOUT]['R']:.5f} "
          f"(anchor {scores[HELDOUT]['anchor_R']:.5f}), on training contexts {scores[SEEN]['R']:.5f}")
    print("  planted state                 recovery Q (held-out)")
    for a, pooled in enumerate(scores[HELDOUT]["modules"]):
        q = "undefined (no response)" if pooled["Q"] is None else f"{pooled['Q']:+.3f}"
        print(f"  {a:2d}  {RESPONSES[a % 5] + ' ' + CUES[a // 5]:25s} {q}")
    return model


# 4. Information localization ------------------------------------------------------------------------------------------

def localization(instance_dir, parent, out, updates, pairs, arms=("a", "e"), cut="encoder", rate=1e-3):
    """From the assay's model, train the part behind one cut from two starting points, fine-tuning (a) and a front
    reset (e), and read the recovery of the shared and selective shift and spread states before and after."""
    contract.CUTS["your_model"] = YourModel.CUTS            # your model's cuts, beside the built models'
    task = Task(instance_dir)
    panels = {"held-out": PanelSpec(root=61001, pool_group=HELDOUT, pairs=pairs, contexts=task.heldout_contexts)}
    reset_seed = 71001
    fresh = {k: v.clone() for k, v in build_model(task.instance, reset_seed).state_dict().items()}
    schedule = max(updates, starts.WARMUP + 1)

    def q(model):
        bits_ = evaluate(model, task, panels, BITS, TARGET_BIT, "cpu", panel_name="quickstart")["supports"]["held-out"]["bits"]
        return {bit: bits_[str(bit)]["Q"] for bit in BITS}

    readings = {"original": q(parent)}
    for arm in arms:
        model = copy.deepcopy(parent).train()
        parameters, embedding_copy, _ = starts.build_arm(model, "your_model", cut, arm, fresh, reset_seed, "cpu")
        optimizer = torch.optim.AdamW(parameters, lr=rate, **ADAMW)
        row = {"name": f"your_model {cut} {arm}", "model": "your_model", "cut": cut, "arm": arm}
        checkpoint = starts.RunCheckpoint(out / arm, row, updates)
        stream = task.stream()
        starts.train(model, optimizer, parameters, lambda u: training_batch(stream, u, 81001, starts.BATCH),
                     lambda u: starts.learning_rate(rate, u, schedule), updates, FIRST_SCORED_GENE, out / arm,
                     checkpoint, "cpu", copy=embedding_copy)
        checkpoint.close()
        readings[arm] = q(model)
    print(f"\nInformation localization at the cut '{cut}', {updates} updates per starting point: recovery Q")
    print("  " + " " * 12 + "".join(f"{RESPONSES[bit % 5] + ' ' + CUES[bit // 5]:>18s}" for bit in BITS))
    for start, values in readings.items():
        print(f"  {start:12s}" + "".join(f"{values[bit]:+18.3f}" for bit in BITS))


# 5. Origin resolution: the recommendation rule ------------------------------------------------------------------------

def your_predictions(rng, lines, strata=("plate1", "plate2"), perturbations=6, genes=200):
    """Stand-in for one side of your model's predictions on measured responses: its units on calibration lines (lines
    the original model was not trained on) or on held-out lines (disjoint from them).

    ``u`` holds the units (``scsailor.resolution.scoring``): ``truth``, ``context`` (``"<line>|<stratum>"``),
    ``p_code`` and ``weight``. Per seed, ``p`` is the original model's prediction; ``a`` is its anchor averaged over
    seeds, ``b`` the control mean of each unit's context and ``A`` the anchor plus the response's shift within the
    unit's design stratum (for example ``scsailor.resolution.net_cost.stratum_shift``). Two changes of the model are
    given, each its prediction per seed: one that moves the prediction toward the truth but overshoots it twofold, and
    one that moves it at random.
    """
    contexts = np.repeat([f"{line}|{s}" for line in lines for s in strata], perturbations)
    truth = rng.gamma(2., .5, size=(len(contexts), genes))
    u = {"truth": truth, "context": contexts, "p_code": np.tile(np.arange(perturbations), len(lines) * len(strata)),
         "weight": np.full(len(contexts), 1 / len(contexts))}
    a = truth + rng.normal(0, .3, truth.shape)
    p = {s: a + .5 * (truth - a) + rng.normal(0, .1, truth.shape) for s in SEEDS}
    side = {"u": u, "p": p, "a": a, "b": truth + rng.normal(0, .5, truth.shape), "A": truth + rng.normal(0, .35, truth.shape)}
    changes = {"toward the truth": {s: p[s] + 2 * (truth - p[s]) + rng.normal(0, .05, truth.shape) for s in SEEDS},
               "at random": {s: p[s] + rng.normal(0, .15, truth.shape) for s in SEEDS}}
    return side, changes


def resolution(rng):
    """Every amount is chosen on calibration lines; held-out lines report the same gaps and take no part in the
    choice."""
    calibration, cal_changes = your_predictions(rng, lines=("L0", "L1", "L2"))
    heldout, held_changes = your_predictions(rng, lines=("L3", "L4"))
    print(f"\nOrigin resolution, the recommendation rule (margin {net_cost.RULE_MARGIN}% of the original model's total MSE)")
    print("  change            kept fraction  comparison    worst gap, calibration  held-out  recommended")
    for change in cal_changes:
        cal, held = dict(calibration, R=cal_changes[change]), dict(heldout, R=held_changes[change])
        amounts = net_cost.calibrate_original(cal)
        kept = net_cost.calibrate_change(cal, amounts)
        alpha_min = min(k["alpha"] for k in kept.values())
        gaps = {"calibration": net_cost.cumulative_gaps(cal, amounts, kept),
                "held-out": net_cost.cumulative_gaps(held, amounts, kept)}
        for comparison in net_cost.COMPARISONS:
            worst = {side: net_cost.worst_gap(g, comparison) for side, g in gaps.items()}
            verdict = net_cost.recommend(alpha_min, worst["calibration"])
            print(f"  {change:17s} {alpha_min:13.2f}  {comparison:13s} {worst['calibration']:+22.2f}"
                  f"  {worst['held-out']:+8.2f}  {'yes' if verdict else 'no'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="a new directory for the instance and the runs")
    parser.add_argument("--updates", type=int, default=300, help="training updates of the assay's model")
    parser.add_argument("--start-updates", type=int, default=200, help="training updates of each starting point")
    parser.add_argument("--draws", type=int, default=1, help="technical draws per replayed bag")
    parser.add_argument("--pairs", type=int, default=8, help="flip pairs per planted state in localization's panel")
    args = parser.parse_args(argv)
    set_precision(threads=4)
    instance = write_instance(args.out / "instance", your_data(np.random.default_rng(0)))
    model = assay(instance, args.out / "assay", args.updates, args.draws)
    localization(args.out / "instance", model, args.out / "localization", args.start_updates, args.pairs)
    resolution(np.random.default_rng(1))


if __name__ == "__main__":
    main()
