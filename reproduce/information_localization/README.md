# Information localization: the paper's runs on Tahoe-100M

The settings, parents and run entries of the paper's information-localization runs of the three built models on Tahoe-100M, and the command that rebuilds, from the saved run records, the map tables of all six models, the training-longer table and the cue-profile tables. The method is `scsailor.localization`; these modules give it the study's settings and files.

| Module | What it does |
|---|---|
| `tahoe.py` | Settings (the parents' learning rate, the training-stream and re-initialization seeds, the panels), parent loading, and one map run, feasibility run or separate trajectory from a row |
| `planted_faults.py` | Builds the five planted-fault parents and the two unfaulted controls |
| `time_and_presence.py` | The cue profile of one parent; the score of one of a parent's later checkpoints (training longer) |
| `tables.py` | Rebuilds the tables from the saved run records |

Not included yet: the restoring changes re-run on measured responses (`scsailor.localization.real_task`), the published models' runs (State, CPA, PrePR-CT), the runs on the other screens, the retraining of the summary-statistics donor of the planted faults, and the DeepSets model's and State's training-longer scores and the later-checkpoint cue profiles.

## Tables from the saved run records

```bash
python -m reproduce.information_localization.tables --data DATA --out OUT --compare DATA
```

`DATA` is the data root ([`../data/FOLDERS.tsv`](../data/FOLDERS.tsv)). For each map folder in `tables.MAPS`, the runs under `<folder>/map/` are read by `scsailor.localization.readings` with the paper's settings (`tables.READINGS`): the map's seeds 31002 and 31003, seed 31004 for the restoring changes only, each model's designated cut and the planted faults. Each folder is read for the selective shift (bit 1) and, where `MAPS` lists them, the shared shift (bit 0) and the shared spread (bit 5). Training longer reads `information-localization/recovery-by-training-length/checkpoint-scores/`, and the cue profiles read `information-localization/presence/profiles/`. `--compare` prints `equal`, `DIFFERS` or `not deposited` for each table against the saved one at the same path; it leaves out the `run` column and the saved rows of fold 1's finer DeepSets cut, which is not one of the map's cuts. The tables take about two minutes on a CPU, with NumPy only.

The saved files, and so the tables, name the attention model `state` and State `official_state`.

## Runs

A run is one row of the map, the full grid of cuts, starting points and learning rates (fields: `tahoe.py`'s docstring). For example, the summary-statistics model trained on the synthetic task, retrained behind the cut `projections` from its own weights at 30 times its rate:

```json
{"name": "map-summary-synthetic-31002-projections-a-x30", "stage": "map", "model": "summary", "half": "synthetic",
 "fold": "F1", "seed": 31002, "case": "parent", "cut": "projections", "arm": "a", "rate_multiple": 30,
 "parent": "WEIGHTS/summary_31002_L2.pt", "inputs": "DATA/tahoe-100m/F1/synthetic", "out": "runs/map-summary-synthetic-31002-projections-a-x30"}
```

From the repository root, after `pip install ".[torch]"`:

```bash
python -m reproduce.information_localization.tahoe --row ROW.json --device cuda
```

A map run takes 40,000 updates, a feasibility run 2,000 and a separate trajectory 120,000 (`scsailor.localization.starts`). A run that stops continues from its last checkpoint when started again with the same row.

The parents are the built models' fits at 40,000 updates (on the synthetic task, `reproduce.assay.study.fit`); training longer reads the same fits' checkpoints saved later along the same schedule. The saved checkpoints name the attention model `state`; the loaders read them under its public name.

The planted-fault parents, from the donors they are built from:

```bash
python -m reproduce.information_localization.planted_faults --summary-donor PT --summary-base PT --deepsets-donor PT --instance DATA/tahoe-100m/F1/synthetic --out FOLDER
```

The cue profile of one parent, and the score of one later checkpoint (`time_and_presence.TIME_AXIS_STEPS` lists the summary-statistics and attention-model checkpoints the paper scored):

```bash
python -m reproduce.information_localization.time_and_presence presence --instance DATA/tahoe-100m/F1/synthetic --half synthetic --model summary --seed 31002 --parent PT --out RESULTS
python -m reproduce.information_localization.time_and_presence time-axis --instance DATA/tahoe-100m/F1/synthetic --model summary --seed 31002 --step 60000 --checkpoint PT --out RESULTS
```

## Reproducibility

On the CPU, the run entries give bit for bit the outputs of the code that made the paper's runs (short runs of every starting point, a resume, a cue profile, two later checkpoints and the planted-fault parents). Training on a GPU is not bit-for-bit repeatable, because some GPU operations accumulate in a varying order.
