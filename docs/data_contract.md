# Data contract

The files a dataset fold must provide, and the conventions they share. Expression is log1p(CP10k) on one panel of 2,000 genes per fold, in the same gene order in every file. A *context* is one background of control cells (for example a cell line on one plate); a *unit* is one context under one perturbation.

## Synthetic-answer assay: an instance directory

Read by `scsailor.assay.Instance`.

| File | Holds | Layout |
|---|---|---|
| `INSTANCE.json` | contexts, generator parameters, evaluation support | below |
| `generator.npz` | the calibrated generator | below |
| `input_pools.h5` | every input control cell of each context | `train/<context>/x` (cells × 2000, float64) and `depth` (int64); held-out contexts under `F1/<context>` |
| `evaluation_bags.h5` | one fixed 256-cell bag per evaluation context | `{F1,seen}/<context>/x` (256 × 2000, float64), `depth`, `indices` (rows of the pool) |
| `truth.h5` | base states and synthetic targets | attributes `instance`, `parameters`; `{F1,seen}/<context>/states` (4 × 26 × 25, int8) and `truth` (4 × 26 × P × 2000, float64), P = number of perturbations |

**`INSTANCE.json`**

| Field | Meaning |
|---|---|
| `instance` | name that checkpoints and prediction files must carry |
| `parameters` | `alpha`, `I_mean`, `T_mean` (`scsailor.assay.generator.Parameters`) |
| `train_contexts` | training contexts, in generator order |
| `F1_contexts` | held-out contexts (the held-out side is called `F1` in every fold) |
| `evaluation` | `{"F1": [...], "seen": [...]}`, entries with `context`, `q` (index in the side's list) and `root` (seed root) |

**`generator.npz`**: `r_gen` (P × 2000 response field), `pi` (P training weights), `directions` (at most 5 × 2000), `code` (P perturbation codes), `beta` (25), `c_train`, `intercept` (float32), `p_ids` (P perturbation names) and `context_keys` (equal to `train_contexts`).

**Conventions**

- Genes 0–49 carry the planted states (state a writes genes 2a and 2a + 1); `directions` are zero there. Scoring and the training loss start at gene 50.
- Training and held-out contexts are disjoint; training reads only `train/` pools.
- Evaluation support: `F1` is every held-out context (`root` 41001); `seen` is the last eight training contexts (all of them when there are fewer; `root` 51001). Each has 4 base states × 26 configurations (the base state and its 25 single-state flips). The synthetic target is computed on the context's complete pool, not on the bag.
- Saved predictions (`scsailor.assay.replay.write_predictions`): `{side}/{context}/raw` and `anchor_raw` (4 × 26 × P × 2000, float32) and `done` (4 × 26), with the instance name and parameters as attributes. `scsailor.assay.score.score_predictions` refuses a file from another instance.

## Natural task: a training directory and its evaluation files

| File | Must hold | Read by |
|---|---|---|
| `controls.h5` | one group per training context code `"0"` … `"N-1"` (code i is `context_keys[i]`), each with `input/x` (cells × 2000, float32) and `input/depth` | `scsailor.natural.data.NaturalData` |
| `train_targets.npz` | `target` (units × 2000, the unit's mean treated profile), `parent` (context code), `p_code`, `weight` (sums to one), `intercept`, `p_ids` | `NaturalData` |
| `training_constants.npz` | `context_keys` in code order; `r_input` (P × 2000, mean perturbation offsets) and `c_train` (mean of the training controls), from which a model is built | `scsailor.models.RawModel` |
| held-out units (`.npz`) | `truth` (units × 2000), `context`, `p_code`, `weight` | scoring |
| `evaluation_inputs.h5` | `train/<context>` and `heldout/<context>`, each one fixed bag: `x` (256 × 2000, float32), `depth`, `indices` | prediction |

**Training stream.** `NaturalData.batch(step, m, level, seed)` draws 8 units by `weight` (seed key `[seed, step, 0]`), then for each slot m = 256 control cells of its context (`[seed, step, slot, 1]`) and the level's carrier (`[seed, step, slot, 2]`); the label is the unit's `target`.

**Perturbation codes.** Code p is row p of `r_input`, `r_gen`, `pi` and `code` and entry p of `p_ids`, in the same order on both tasks.

**Context keys.** The first `|`-separated field is the held-out unit of the fold (cell line, donor or cell identity).

## Datasets of the paper

| Dataset | Context key | Held out by | Perturbations | Contexts, training / held-out (assay) |
|---|---|---|---|---|
| Tahoe-100M | cell line\|plate | cell line (28 / 14 lines) | 93 | 56 / 28 |
| Parse PBMC | donor\|cell type | donor (8 / 4 donors) | 87 | 24 / 12 |
| GSE306429 | identity\|plate\|library\|replicate | cell identity (fold 1: 6 / 4; folds 2–3: 7 / 3) | fold 1: 102; fold 2: 104; fold 3: 111 | fold 1: 77 / 64; fold 2: 100 / 41; fold 3: 105 / 36 |
| Replogle–Nadig | cell line | K562 | 1,966 | 3 / 1 |

For Replogle–Nadig, `truth.h5` stores the states only, because one context's table would hold about 3.3 GB; its synthetic targets are computed one base state at a time.
