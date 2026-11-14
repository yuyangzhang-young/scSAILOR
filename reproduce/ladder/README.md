# The information ladder: Results 1 and 2

The tables of the paper's Results 1 and 2 (Figs. 1e-f and 2b-d, Supplementary Notes S3 and S5, Supplementary Tables 2, 6 and 7, Supplementary Fig. 1 and the ladder sheets of Supplementary Data 1 and 2): every natural-task fit on the five screens with its held-out scores, the information ladder's values, the model-free references, the intervals, the plate-aware baseline and the adapted weighted R². The code names and the paper's terms are listed in [`__init__.py`](__init__.py).

The fits' held-out predictions are not deposited. The per-fit table is read from the score files written when they were scored, which are deposited (`natural-scores/`, below): the scores of `scsailor.natural.score.risk` (total, common and contrast MSE) and of Supplementary Note S3's adapted weighted R². The model-free references, the plate-aware baseline and Parse PBMC's no-change weighted R² need no model and are computed here from the deposited fold data.

| Module | What it does |
|---|---|
| `fits.py` | Reads each fit's saved scores (model and baseline MSE, and the training MSE where saved), applies the fit rules and writes the per-fit table, with the adapted weighted R² of each fit |
| `ladder.py` | Fold points and fold means of relative MSE and of the model-minus-baseline components, and their changes from Mean; Parse PBMC on twelve and ten donors; the excluded fits and their reinstatement; no change against the models |
| `references.py` | The model-free references of Parse PBMC and Jiang et al. and the models against the strongest |
| `intervals.py` | Paired bootstrap of the held-out lines of Tahoe-100M for the change from Mean to Cells |
| `plate.py` | The plate-aware baseline on Tahoe-100M and Supplementary Table 7 |
| `weighted_r2.py` | The adapted weighted R² of a prediction; Supplementary Fig. 1's differences and Supplementary Data 2's references |
| `tables.py` | Writes every table, in this order |

## Environment

Python 3.11 and, from the repository root:

```bash
pip install . numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 pandas==2.3.3 pyarrow==25.0.0
```

These are the versions the tables were computed with. On another CPU the last digits of a value can differ (see `--compare` below).

## Tables

```bash
python -m reproduce.ladder.tables --data DATA --out OUT --compare DATA
```

`DATA` is the data root ([`../data/FOLDERS.tsv`](../data/FOLDERS.tsv)); the deposited tables are under `DATA/ladder/`. The tables are written under `OUT/ladder/` at full precision, in the order in which they depend on each other. `--compare DATA` prints `equal`, `DIFFERS` or `not deposited` for each table against the deposited one at the same path, cell for cell and exactly; on another CPU a `DIFFERS` can come from last digits alone, so compare such a table numerically before reading it as a difference. The command sets four BLAS threads; it takes about 20 seconds on a CPU and 2.3 GB of memory.

A fit is one model trained on one screen, fold, seed and level. Its relative MSE is its total MSE over its baseline's; its component values are the model's common or contrast MSE minus the baseline's, in % of the baseline's total MSE. A built model's baseline is the one read along its own scoring path, a published model's the fold baseline; the two agree to 2e-7 of the baseline's MSE. A fold point is the mean over the kept fits (seeds) of a model, level and fold; a fold mean is the mean of a level's fold points over folds. A change from Mean is the level's fold point minus the Mean level's fold point of the same model and fold, each over its own kept fits: the two seed sets differ where a fit is set aside at one level only (CPA on Replogle–Nadig and Jiang et al.) or a replacement seed enters (the attention model on GSE306429).

The fit rules: a fit is set aside when its training total MSE is at least 0.99 of the anchor's training total MSE (the anchor is the baseline the model adds its correction to, stored before clipping and scored, like the model, after clipping at zero; only the per-fit tables and the Tahoe-100M built models' records hold training MSE), and then when its held-out relative MSE exceeds 1.4 times the median over the fits of the same screen, fold, model and level that passed the training rule, itself included. Seeds 31005-31007 are replacement seeds trained where a seed failed training.

| Table | One row per | Printed in the paper |
|---|---|---|
| `FIT_TABLE.csv` | fit (1,525): status under the fit rules, model and baseline MSE (total, common, contrast), relative MSE, model-minus-baseline components, adapted weighted R² (total, contrast) | through the tables below |
| `LADDER.csv` | screen, model, fold (and fold mean) and level: relative MSE, model-minus-baseline components and their changes from Mean, seeds | Results 1-2; Fig. 1e-f; Fig. 2b-d |
| `PARSE_PBMC_DONORS.csv` | evaluation (all twelve donors; ten donors, Donor2 and Donor3 left out of scoring), model, donor group (and fold mean) and level: as `LADDER.csv`; the fit rules are applied within each evaluation (only the outlier rule can act, and it sets no fit aside) | Supplementary Table 2; Supplementary Data 1 (Parse PBMC sheets) |
| `FIT_EXCLUSIONS.csv` | fit set aside by the fit rules | Methods (13 attention-model fits, 3 CPA fits); Supplementary Data 1 |
| `FIT_SENSITIVITY.csv` | screen, model and level with an outlier: fold-mean relative MSE with the rules and with the outliers reinstated | Supplementary Note S4.4 (1.021 to 1.065, 1.201 to 1.231, 1.240 to 1.267); Supplementary Data 1 |
| `NO_CHANGE.csv` | fold of GSE306429, Replogle–Nadig and Jiang et al.: no change's relative MSE and the lowest model fold point, with its model and level | Results 2 (the folds where no change is lower; their held-out lines are in Supplementary Data 1's fold list); Supplementary Note S5.3 (0.926, the mean of the six Jiang et al. rows) |
| `REFERENCES.csv` | screen, fold and model-free reference: relative MSE, components, best scale, the strongest | Supplementary Note S5.3 |
| `MODELS_VS_REFERENCES.csv` | screen, model, level and fold (and fold mean) of Parse PBMC and Jiang et al.: model MSE minus the strongest reference's, in % of baseline total MSE | Results 2; Supplementary Note S5.3 |
| `INTERVALS.csv` | model and component: change from Mean to Cells with its 90% interval | Supplementary Table 6 |
| `PLATE_AWARE_BASELINE.csv` | fold: the plate-aware baseline's relative MSE, reductions and adapted weighted R² | Results 1; Supplementary Table 7 |
| `SUPPLEMENTARY_TABLE_7.csv` | the plate-aware baseline and each model | Supplementary Table 7 |
| `TAHOE_PLATE_COMPARISON.csv` | fold, model and level: seed-mean adapted weighted R² beside the plate-aware baseline's and the baseline's | Results 1 (38: the rows where `above_plate_aware` is true); Supplementary Data 2 (Tahoe plate comparison) |
| `WEIGHTED_R2.csv` | screen, model, version, fold (and fold mean) and level: seed-mean adapted weighted R² and its difference from the baseline's | Supplementary Fig. 1; Supplementary Data 2 (plotted) |
| `WEIGHTED_R2_REFERENCES.csv` | screen, fold and reference: adapted weighted R² of the baseline and of no change, prediction units left out | Supplementary Note S3; Supplementary Data 2 (references) |

The model-free references (`references.py`): the scale of the design-aware baseline is fitted on the calibration lines with the responses of the fitted lines of the nested split (six donors on Parse PBMC, three cell lines on Jiang et al.); the held-out references use the responses of all training lines. On a tie for the strongest reference the first of baseline, stratum mean and design-aware baseline is marked (Parse PBMC G1, where the best scale is 1).

## Inputs

Under the data root:

- `natural-scores/`, the saved held-out scores of every fit (the files name the attention model `state`, `STATE-like` or `attention`, and State `state_official`, `state_pub` or `State`):
  - `tahoe-100m/F1/<model>_<seed>_<level>.json`: the built models on fold 1, one record per fit (`sides`, `heldout` and `train`, `risk_columns`, `original` for the model and `anchor` for its baseline); `tahoe-100m/built_models_F2_F3.csv`: the same on folds 2-3; `tahoe-100m/<fold>/{state,cpa,prepr}.csv`: the published models' per-seed MSE divided by the fold baseline's total, which `tahoe-100m/<fold>/baseline.json` records with its two components' shares;
  - `gse306429/<fold>/built_models.csv`: per-fit model and anchor MSE and the training MSE; the held-out columns read are `heldout_matched99_*`, the ladder support every model is scored on (the held-out prediction units whose perturbation is among the fold's eligible training units and whose compound has a structure PrePR-CT can encode; Supplementary Note S2.3: 479, 288 and 245 units); `gse306429/<fold>/{state,cpa,prepr}.csv` with `{state,cpa,prepr}_baseline.json`, as on Tahoe-100M; `gse306429/<fold>/no_change.csv`;
  - `replogle-nadig/<fold>/` and `jiang/<fold>/`: `built_models.csv`, `state.csv`, `cpa.csv` (per-fit tables as GSE306429's, columns `heldout_*`) and `no_change.csv` (`input_only_raw` is no change, `anchor_raw` the baseline);
  - `held-out-unit-sums/part-{1,2,3}.csv`: each fit's weighted error sums per held-out cell line (Tahoe-100M) or donor (Parse PBMC), the files' `unit`, with the baseline's (`kind` `baseline`) and, for the built models, the fit's own anchor (`fit_anchor`); a fold baseline's row that appears in more than one part file is read once, from the first;
  - `weighted-r2/part-{1,2}.csv`: the adapted weighted R² of every fit and of each fold's baseline and no change, with the prediction units left out; `weighted-r2/parse_pbmc_donors.csv`: Parse PBMC's on all twelve donors (`support` `twelve`) and on ten;
- Tahoe-100M's fold data, read through the memorization layer's `reproduce.memorization.data.fold_data` (with `reproduce.origin_resolution.tahoe`): `tahoe-100m-inputs-fold-1/`, `tahoe-100m-inputs-folds-2-3/`, `tahoe-100m-evaluation-units-fold-1/`, `tahoe-100m-evaluation-units-folds-2-3/`, `tahoe-100m-line-subsets/` and `training-arrays-parse-pbmc-tahoe-100m/`;
- Parse PBMC's fold data (`parse-pbmc-inputs/`, `parse-pbmc-held-out-truth/`, `training-arrays-parse-pbmc-tahoe-100m/`, read through `reproduce.origin_resolution.parse`) and Jiang et al.'s, folds 1-6 (`jiang-inputs/`, `genetic-screens-predictions/jiang/truth/`, read through `reproduce.origin_resolution.jiang`).

## Not included

Training, the preparation of inputs from raw cells, the scoring of predictions into the deposited score files, and figure drawing. Not computed here: the control-cell signatures of Supplementary Table 1 and the counts of Supplementary Note S2 (raw cells); the training-length selection, the published models' native outputs and the reinstatement of GSE306429's excluded attention-model fits at their original seeds (Supplementary Notes S4.3-S4.4); the plate direction read from controls, the prediction slopes at 160,000 updates and the other-plate probe of Supplementary Note S5.2 (predictions and weights not deposited); the CPA carrier probes (Supplementary Note S7.1); and the Supplementary Data 1 sheets built from raw data or other parts of the paper.
