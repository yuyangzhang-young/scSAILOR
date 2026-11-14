# Synthetic-answer assay: the paper's settings and tables

`study.py` holds the settings with which the paper trained, replayed and scored the three built models in the synthetic-answer assay on Tahoe-100M (folds F1 to F3) and Parse PBMC (groups G1 to G3), and builds the models from `scsailor.models`. The method is `scsailor.assay`, whose modules take these settings as arguments. `references.py` computes the paper's attainable reference, the decoded-cue reference (Supplementary Table 9); `scsailor.assay.capability.judge` takes a reference as an argument.

| Name | Setting |
|---|---|
| `load_instance(root, dataset, fold)` | The instance of one fold (Tahoe-100M) or group (Parse PBMC): `<root>/tahoe-100m/<fold>/synthetic` or `<root>/parse-pbmc/<fold>/synthetic`, a directory as in [`docs/data_contract.md`](../../docs/data_contract.md) |
| `fit(model, seed, level)` | 40,000 updates of a learning-rate schedule defined over 160,000 updates (`SCHEDULE`: warm-up over 1,000 updates, then a cosine that would reach 10% of the peak at 160,000); peak learning rate `PEAK_LR` (3e-5 for the attention model, 1e-4 for the others); checkpoints at `SAVE_AT` |
| `SEEDS` | 31002, 31003, 31004 |
| `DRAWS` | Technical draws per bag in the replay: 1 at L0 and L1, 16 from L2 on |
| `replay_scopes(model, level, step)` | Which configurations the checkpoint at 40,000 updates replays on the held-out and the seen backgrounds |
| `make_model`, `load_checkpoint` | A new built model, or a trained one from its checkpoint, with the runtime settings of the paper's runs (one thread, full-precision matrix products, seeds) |

The study's saved checkpoints name the attention model `state`. `load_checkpoint` reads them, and the record it returns uses the public name `attention_model`.

## One run

The DeepSets model at +Var with seed 31002 on Tahoe-100M fold 1: training, replay and scores. From the repository root, after `pip install ".[torch]"`:

```python
from scsailor.assay import replay, score, train
from scsailor.assay.stream import TrainingStream
from reproduce.assay import study

instance = study.load_instance("DATA", "tahoe", "F1")
fit = study.fit("deepsets", 31002, "L2")
model = study.make_model(instance.generator, fit.model, fit.seed, "cuda:0")
train.train(model, TrainingStream(instance), fit, instance, "runs/deepsets_31002_L2", "cuda:0")

checkpoint = "runs/deepsets_31002_L2/step40000.pt"
model, _ = study.load_checkpoint(instance, checkpoint, "cuda:0")
replay.write_predictions(model, instance, "L2", 40000, study.replay_scopes("deepsets", "L2", 40000),
                         study.DRAWS["L2"], "runs/deepsets_31002_L2/predictions_step40000.h5", "cuda:0",
                         checkpoint=checkpoint)
scores, _ = score.score_predictions(instance, "runs/deepsets_31002_L2/predictions_step40000.h5",
                                    output="runs/deepsets_31002_L2/scores_step40000")
```

`scsailor.assay.capability` pools the scores of the three seeds (`seed_rows` with `seeds=study.SEEDS`, then `pool_seeds`) into the table of what each model reads.

## Tables

The tables of Results 3, Fig. 3 and Supplementary Note S6, computed on a CPU:

```bash
python -m reproduce.assay.tables --data DATA --out OUT --compare DATA
```

`DATA` is the data root ([`../data/FOLDERS.tsv`](../data/FOLDERS.tsv)); the deposited tables are under `DATA/synthetic-answer-assay/`. The tables are written under `OUT/synthetic-answer-assay/` at full precision. `--compare DATA` prints `equal`, `DIFFERS` or `not deposited` for each table against the deposited one at the same path, cell for cell and exactly; on another CPU a `DIFFERS` can come from last digits alone, so compare such a table numerically before reading it as a difference. The command needs no torch and takes about 25 minutes and 3 GB of memory, almost all of it for the constructed predictions and the attainable references. Python 3.11 with, from the repository root:

```bash
pip install . numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 pandas==2.3.3
```

The synthetic predictions (about 3 TB) are not deposited; the scores of every fit are, and they are the inputs of every model table here. Recovery pools each planted state's true energy and squared error equally over a fit's held-out contexts before the ratio (Methods); every held-out context is pooled, and on Parse PBMC these cover all 12 donors. Seed scores are averaged within folds (Parse PBMC: donor groups G1-G3), then folds equally.

Parse PBMC's data, its scores and the table rows computed from them are CC BY-NC 4.0: the Parse PBMC rows of `FITS.csv`, `RECOVERY_BY_FOLD.csv`, `RECOVERY.csv` and `ATTAINABLE_REFERENCES.csv`, and `ISOLATION.csv`, which pools every screen.

| Module | What it does |
|---|---|
| `scores.py` | Reads each fit's saved scores, pooled over its held-out contexts |
| `recovery.py` | The per-fit table with the attention model's training collapses, seed and fold means, the isolation checks of the saved predictions, the positive control |
| `whole_profile.py` | Recovery against the five whole-profile scores |
| `references.py` | The decoded-cue reference, from an instance |
| `checks.py` | The carrier check, from an instance |
| `fold_selection.py` | The genetic-screen folds the assay uses |
| `constructed.py` | Constructed predictions with known errors and their whole-profile and paired scores, from an instance |
| `tables.py` | Writes every table |

| Table | One row per | Printed in the paper |
|---|---|---|
| `FITS.csv` | fit (screen, fold, model, seed, level, learning rate) and planted state: status, whether the input carries the cue, true and predicted energy, squared error, recovery, projection slope | Supplementary Note S6.3 (GSE306429 fold 3 seeds: recovery, predicted/true energy, projection slopes); coverage counts (Fig. 3 legend, S6.3, S6.5) |
| `RECOVERY_BY_FOLD.csv` | screen, fold, model, level and non-null planted state: the mean over kept seeds (`n_seeds`) | Supplementary Data 1 (`synthetic_answer_by_fold`: the shared and selective rows); S6.3 (per-fold values) |
| `RECOVERY.csv` | screen, model, level and non-null planted state: the mean over folds of the seed means (Fig. 3c draws the shared and selective rows whose input carries the cue) | Results 3 (0.84-0.88, 0.857, 0.861, below 0.004, below 0.005, 0.054); Fig. 3c; S6.3 |
| `ISOLATION.csv` | model, over the kept fits of every screen: largest predicted energy where the input lacks the cue, and its ratio to the true energy; largest and median leakage ratio of the null states whose cue is in the input | Supplementary Note S6.6 |
| `WHOLE_PROFILE_SCORES.csv` | Tahoe-100M fit: the five whole-profile scores and the recovery summed over the five shared and the five selective tests | through `WHOLE_PROFILE_CORRELATIONS.csv` |
| `WHOLE_PROFILE_CORRELATIONS.csv` | all six models together, or one model, and pair of scores: the fold-mean Spearman correlation and each fold's | Results 3 (0.79, at most 0.39); Fig. 3b; Supplementary Note S6.5 |
| `POSITIVE_CONTROL.csv` | labels (exact, sampled) and non-null shift state: recovery | Supplementary Table 10 |
| `GENETIC_FOLD_SELECTION.csv` | Replogle–Nadig and Jiang et al. fold: held-out line, contrast error of no change and of the baseline on the natural task, noise floor, the fraction of the above-noise signal the baseline explains (%), and the folds with the largest and smallest fraction | Supplementary Note S6.3 (+16, −62, +23, −45) |
| `CONSTRUCTED_PREDICTIONS.csv` | Tahoe-100M fold, spread test (shared, selective), constructed case (Exact, Offset, Swapped, Halved, Absent, Reversed), assessment (whole profile, paired) and score | through `CONSTRUCTED_PREDICTIONS_FIG3D.csv` |
| `CONSTRUCTED_PREDICTIONS_FIG3D.csv` | case, score and assessment: the mean, minimum and maximum over the six tests (two spread tests in three folds) and the number of tests with a defined value | Fig. 3d; Supplementary Note S6.7 (0.98, 0.93, 1.00, −1, 0.50, 0.38; Q 1, 0.75, 0, −3; 4V) |
| `ATTAINABLE_REFERENCES.csv` | instance (Tahoe-100M folds 1-3, Parse PBMC group 1) and planted state: the decoded-cue reference's held-out recovery, and the number of decoded inputs and wrongly decoded states | Results 3 (0.918-0.934, 0.871-0.899); Supplementary Table 9 |
| `CARRIER_CHECK.csv` | input set of Tahoe-100M fold 1 (28 held-out, 8 training contexts): the largest paired difference on the output genes, on depths and, below the cue's first level, on the whole input; the pairs (states x levels), the pairs below the first level, and the states not read back | Supplementary Note S6.6 (36 input sets, maximum difference 0) |

The checks of Supplementary Note S6.6 are in two tables: `CARRIER_CHECK.csv` checks the model inputs (the carrier check), `ISOLATION.csv` the saved predictions. The term table in [`__init__.py`](__init__.py) maps the columns to the paper's terms. pandas reads the response label `null` as missing unless `keep_default_na=False`.

An attention-model fit is excluded as a training collapse when its predicted energy is below 1% of the true energy for every non-null planted state: four fits on GSE306429 fold 1 and five on Jiang et al., whose cells then average two seeds, and thirteen on Replogle–Nadig. Each of the thirteen has a fit of the same seed retrained at 0.3 times the standard peak learning rate, deposited as a separate fit (`learning_rate` `0.3x standard`) and kept like any other. Every kept attention-model fit predicts at least 8.3% of the true energy on some non-null planted state.

## Inputs

Under the data root:

- `synthetic-answer-scores/<screen>/<fold>/<model>_<seed>_<level>[_lr03]/SCORES.json` (screens `tahoe-100m`, `gse306429`, `parse-pbmc`, `replogle-nadig`, `jiang`; model codes `summary`, `deepsets`, `attention_model`, `state`, `cpa`, `prepr`): the scores of 703 fits, written by `scsailor.assay.score` or by the same scoring in the published models' packages ([`../published_models/`](../published_models/)); the held-out side `"F1"` holds the 25 planted states' rows pooled over the held-out contexts;
- `synthetic-answer-scores/tahoe-100m/<fold>/POOLED_SCORES_BUILT_MODELS.csv` (the built models, folds 1-3) and `POOLED_SCORES_PREPR.csv` (PrePR-CT, folds 2-3): the same pooled rows of the other 165 fits, whose score files are not deposited, with their saved columns (the attention model `state`; `scope` `F1` and `support` `heldout` are the held-out rows);
- `synthetic-answer-scores/tahoe-100m/<fold>/WHOLE_PROFILE_SCORES_BUILT_MODELS.csv` and `WHOLE_PROFILE_SCORES_PUBLISHED_MODELS.csv`: each Tahoe-100M fit's whole-profile scores on the synthetic task, computed from its predictions (saved model codes: `state` the attention model, `state_pub` State);
- `synthetic-answer-positive-control/<exact|sampled>/SCORES.json`: the positive control's two fits;
- `genetic-screens-held-out-references/<replogle-nadig|jiang>/<fold>/REFERENCE_SCORES.csv` and `NOISE_FLOOR.csv` (Replogle–Nadig folds F1-F4, Jiang et al. F1-F6): the natural task's held-out contrast errors of no change and of the baseline (saved word `anchor`), and the noise floor of the measured responses, as saved;
- the instances `tahoe-100m/<fold>/synthetic/` (folds F1-F3) and `parse-pbmc/G1/synthetic/`, for the constructed predictions, the attainable references and the carrier check.

Parse PBMC's inputs, `synthetic-answer-scores/parse-pbmc/` and `parse-pbmc/G1/synthetic/`, are CC BY-NC 4.0.

## Not included

Training and replay (above, and `scsailor.assay`), and the models' whole-profile scores, which need the predictions. The natural-task error change of CPA quoted in Results 3 comes from the natural task's tables.
