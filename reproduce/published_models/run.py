"""Train one published model at one information level and seed, then predict (natural task) or replay and
score (synthetic-answer assay).

    python -m reproduce.published_models.run --model state natural --fold DATA/tahoe/F1 \\
        --level L2 --seed 31002 --output runs/state/tahoe-F1/L2-31002
    python -m reproduce.published_models.run --model cpa synthetic --instance DATA/tahoe/F1/synthetic \\
        --level L2 --seed 31002 --output runs/cpa/tahoe-F1-synthetic/L2-31002

Run from the repository root. Each model trains for a fixed number of updates (State 5,000, CPA 140,000,
PrePR-CT 10,000) and is read out there. ``--device cpu`` works but is slow at full size.
"""
from .common import runtime  # first: sets the thread variables before NumPy and PyTorch load

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from scsailor.assay import Instance  # noqa: E402
from scsailor.assay.score import score_predictions  # noqa: E402

from .common.fit import fit  # noqa: E402
from .common.natural import PairedStream, TrainingSet, predict  # noqa: E402
from .common.synthetic import TreatedCellStream, score_subset, write_predictions  # noqa: E402

MODELS = ("state", "cpa", "prepr_ct")
LEVELS = ("L1", "L2", "L3", "L3d", "L4")


def _features(path):
    with np.load(path, allow_pickle=False) as z:
        return z["normalized"]


def natural(args, module):
    data = TrainingSet(args.fold, features=args.fold / "drug_features.npz" if args.model == "prepr_ct" else None)
    if args.model == "prepr_ct":
        adapter = module.Adapter("natural", args.seed, data.r, data.features, args.device,
                                 feature_row=data.feature_row, r_row=data.r_row)
    else:
        adapter = module.Adapter("natural", args.seed, data.r, args.device)
    model = fit(adapter, PairedStream(data, args.level, args.seed), steps=adapter.steps, output=args.output,
                device=args.device, workers=args.workers)
    predict(adapter, model.eval(), data, level=args.level, step=adapter.steps, output=args.output)


def synthetic(args, module):
    instance = Instance(args.instance)
    if args.model == "prepr_ct":
        adapter = module.Adapter("synthetic", args.seed, instance.generator.r, _features(args.features), args.device)
    else:
        adapter = module.Adapter("synthetic", args.seed, instance.generator.r, args.device)
    model = fit(adapter, TreatedCellStream(instance, args.level, args.seed), steps=adapter.steps,
                output=args.output, device=args.device, workers=args.workers)
    # an instance with too many perturbations to store all synthetic targets is scored on a fixed subset
    subset = Path(args.instance) / "EVALUATION_SUBSET.json"
    codes = json.loads(subset.read_text())["codes"] if subset.exists() else None
    predictions = Path(args.output) / f"PREDICTIONS_step{adapter.steps}.h5"
    write_predictions(adapter.predictor(model.eval(), codes), instance, level=args.level, seed=args.seed,
                      step=adapter.steps, checkpoint=Path(args.output) / f"step{adapter.steps}.pt", output=predictions,
                      codes=codes)
    if codes is None:
        score_predictions(instance, predictions, output=Path(args.output))
    else:
        score_subset(instance, predictions, Path(args.output))


def main():
    parser = argparse.ArgumentParser(prog="python -m reproduce.published_models.run", description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    tasks = parser.add_subparsers(dest="task", required=True)
    for name, help_text in (("natural", "train on measured responses, then predict"),
                            ("synthetic", "train on the synthetic-answer assay, then replay and score")):
        task = tasks.add_parser(name, help=help_text)
        if name == "natural":
            task.add_argument("--fold", type=Path, required=True, help="dataset fold directory (docs/data_contract.md)")
        else:
            task.add_argument("--instance", type=Path, required=True, help="synthetic-answer instance directory")
            task.add_argument("--features", type=Path, help="PrePR-CT only: the fold's drug_features.npz")
        task.add_argument("--level", choices=LEVELS, required=True)
        task.add_argument("--seed", type=int, required=True)
        task.add_argument("--output", type=Path, required=True)
        task.add_argument("--device", default="cuda:0")
        task.add_argument("--workers", type=int, default=4, help="threads that prepare batches ahead")
    args = parser.parse_args()
    if args.task == "synthetic" and args.model == "prepr_ct" and args.features is None:
        parser.error("--features is required for PrePR-CT")
    runtime.configure(args.device)
    module = importlib.import_module(f".{args.model}.adapter", __package__)
    (natural if args.task == "natural" else synthetic)(args, module)


if __name__ == "__main__":
    main()
