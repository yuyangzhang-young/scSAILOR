# Information localization: the paper's runs on Tahoe-100M

The settings, parents and run entries of the paper's information-localization runs of the three built models on Tahoe-100M, and the command that rebuilds, from the saved run records, the map tables of all six models, the training-longer table and the cue-profile tables. The method is `scsailor.localization`; these modules give it the study's settings and files.

| Module | What it does |
|---|---|
| `tahoe.py` | Settings (the parents' learning rate, the training-stream and re-initialization seeds, the panels), parent loading, and one map run, feasibility run or separate trajectory from a row |
| `planted_faults.py` | Builds the five planted-fault parents and the two unfaulted controls |
| `time_and_presence.py` | The cue profile of one parent; the score of one of a parent's later checkpoints (training longer) |
| `tables.py` | Rebuilds the tables from the saved run records and cue profiles |

Not included yet: the restoring changes re-run on measured responses (`scsailor.localization.real_task`), the published models' runs (State, CPA, PrePR-CT), the runs on the other screens, the retraining of the summary-statistics donor of the planted faults, and the DeepSets model's and State's training-longer scores and the later-checkpoint cue profiles.

## Tables from the saved run records

```bash
python -m reproduce.information_localization.tables --data DATA --out OUT --compare DATA
```

`DATA` is the data root ([`../data/FOLDERS.tsv`](../data/FOLDERS.tsv)). For each map folder in `tables.MAPS`, the runs under `<folder>/map/` are read by `scsailor.localization.readings` with the paper's settings (`tables.READINGS`): the map's seeds 31002 and 31003, seed 31004 for the restoring changes only, each model's designated cut and the planted faults. Each folder is read for the selective shift (bit 1) and, where `MAPS` lists them, the shared shift (bit 0) and the shared spread (bit 5). Training longer reads `information-localization/recovery-by-training-length/checkpoint-scores/`, and the cue profiles read `information-localization/presence/profiles/`. `--compare` prints `equal`, `DIFFERS` or `not deposited` for each table against the saved one at the same path; it leaves out the `run` column and the saved rows of fold 1's finer DeepSets cut, which is not one of the map's cuts. The tables take about three minutes on a CPU, most of it reading the run records.

Three more tables hold the other numbers of Supplementary Notes S8.5-S8.7, Supplementary Tables 12 and 15, Fig. 4d and Supplementary Fig. 2:

| Table | One row per | Read from |
|---|---|---|
| `information-localization/presence/ONSET_TABLE.csv` | built model, seed, checkpoint, support and key cut: the energy coefficients of the selective-shift cue, the shared-shift cue and the null test of the shift cue, the selective one divided by the other two (`selective_to_shared`, `selective_to_null`, the quotients of `PRESENCE_TABLE`'s `ratio_to_learned` and `ratio_to_null` at full precision), and the checkpoint's selective-shift recovery | the synthetic-trained parents' profiles at 40,000 updates (`profiles/`), the later checkpoints' (`checkpoint-profiles/PRESENCE_synthetic_<model>_<seed>_step<updates>.json`) and `TIME_AXIS.csv` |
| `information-localization/planted-faults/PLANTED_FAULTS_TABLE.csv` | planted fault or unfaulted control and starting point: held-out recovery of the original model and after retraining, the lowest rate whose best reached 0.25, the rates run, and `asterisk` (run only at ×1 and ×30) | the rebuilt `MAP_CELLS` tables and the runs' `EVAL_PARENT.json` |
| `information-localization/light-version/LIGHT_VERSION_GPU_HOURS.csv` | model and fold of the Tahoe-100M maps: the GPU (`tables.GPU`, a stated table: the run records do not name it); the full procedure's runs at the map seeds and their recorded hours; the light runs (the light menu and its second-seed checks), those present in the map and their recorded hours; hours only where every run had a GPU to itself. For the built models, whose runs shared one workstation GPU: the median hours per run (training seconds over the runs sharing the GPU) and that median times the light runs, an estimate | the runs' `RUN_RECORD.json`, `EXECUTION.json` and `EVAL.json` in the nine Tahoe-100M map folders (fold 1's finer DeepSets cut left out), and `scsailor.localization.light` |

Fig. 4d's cue ratio is 100 × `selective_to_shared` on the held-out support at the inner cuts (`tables.INNER_CUTS`: the attention model's blocks 0, 3 and 7 and head input; the summary-statistics model's mean projection, two branch hidden representations and two output projections). Supplementary Note S8.5's ratio at the first trained layer (2.6-27%) reads `tables.FIRST_TRAINED_LAYER` at 40,000 updates on both supports.

The saved files, and so the map, training-longer and cue-profile tables, name the attention model `state` and State `official_state`; the three tables above print the paper's names.

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
