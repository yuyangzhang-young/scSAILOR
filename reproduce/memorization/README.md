# Memorization: Results 7

The tables of the paper's Results 7, computed on a CPU from the models' saved predictions: on Tahoe-100M folds 1-3 (Fig. 6d–f), how far the models store the training lines' responses and control levels, how this pulls their predictions for held-out lines toward the training lines' control-pool growth level, and what readers of a held-out context's own control cells give in place of a model's context-level correction; on Parse PBMC (Fig. 6g), the same readers adapted to donors and the interferon programs' dependence on the training donors; on the genetic screens, storage, the growth-program pull, Tahoe-100M's gene-wise readers applied unchanged and the gene-wise shrink. The method parts these tables read are `scsailor.resolution` (calibration, metrics, line bootstrap, prediction grids) and origin resolution's deposited cases and bindings (`reproduce.origin_resolution`). The code names and the paper's terms are listed in [`__init__.py`](__init__.py).

| Module | What it does |
|---|---|
| `data.py` | A fold's held-out and training units, input-set and control-pool means, drug average and design-aware amount; the saved Mean-level predictions on the ladder; the error readings (energy, context means, interaction part); the calibration-supported units |
| `programs.py` | The three Hallmark growth programs (E2F targets, G2M checkpoint, mitotic spindle) on each fold's gene panel |
| `readers.py` | The training contexts' offsets (source context responses); the gene-wise readers and the growth reader with the gut-lineage feature; the arms (removal, constant, readers) |
| `calibration.py` | Origin resolution's +Var cases (summary-statistics, DeepSets and attention models, State) and each seed's calibration toward the design-aware baseline |
| `arms.py` | The four-arms table |
| `metrics.py` | The arms on the nine metrics, their gains with 90% line intervals, and the summaries |
| `selection.py` | The drug x growth-program pairs selected on the training lines |
| `ranking.py` | The line ranking of the held-out lines by growth-program response, the plate reference, the pull and the ranking without it, and the pair-median growth relation |
| `storage.py` | Storage of the training lines' responses, and control-level memory with training length |
| `growth_genes.py` | The separate growth-gene correction with a linear reader, read by the routine read-outs and the pair-median growth relation |
| `parse.py` | Parse PBMC: the readers with cell-type centring, the arms and their calibration, total MSE by group and pooled over donors, and the nine metrics with donor intervals |
| `parse_interferon.py` | Parse PBMC: the interferon programs against the reference wells (control levels, measured and predicted changes, rankings, ratios, split halves) and the programs' between-donor SDs |
| `genetic.py` | The genetic screens: storage, the growth-program pull, Tahoe-100M's gene-wise readers, the gene-wise shrink and the nine metrics |
| `tables.py` | Writes every table of every screen |

## Environment

Python 3.11 and, from the repository root:

```bash
pip install . numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 pandas==2.3.3 pyarrow==25.0.0
```

These are the versions the tables were computed with; pyarrow reads the cell-line metadata. On another CPU the last digits of a value can differ (see `--compare` below).

## Tables

```bash
python -m reproduce.memorization.tables --data DATA --out OUT --compare DATA [--screens tahoe-100m parse-pbmc genetic-screens]
```

`DATA` is the data root ([`../data/FOLDERS.tsv`](../data/FOLDERS.tsv)); the deposited tables are under `DATA/memorization/<screen>/`. Each screen's tables (all three screens unless `--screens` names some) are written under `OUT/memorization/<screen>/` in the order in which they depend on each other; a table computed from others reads them as written, at six significant digits, as the paper's values were. `--compare DATA` prints `equal`, `DIFFERS` or `not deposited` for each table against the deposited one at the same path, cell for cell and exactly; on another CPU a `DIFFERS` can come from last digits alone, so compare such a table numerically before reading it as a difference. The command sets four BLAS threads, as the paper's tables were computed. On a CPU Tahoe-100M takes about 9 minutes and 8 GB of memory, Parse PBMC about 2 minutes and 1 GB, and the genetic screens about 7 minutes and 2 GB; all three in one command about 18 minutes and 9 GB.

### Tahoe-100M

Each training seed is scored alone; a seed mean is the mean of the seeds' values, and folds are weighted equally. Predictions are clipped at zero before scoring. Squared error is in % of the total MSE that each column names: the design-aware baseline's on the same units (`*_points`, `*_pct_of_design_aware`), the original prediction's (`*_pct_of_original`, and the gains table's `mse`) or the calibrated prediction's (`*_pct_of_calibrated`); only `NINE_METRICS_VALUES.csv`'s `mse` is the mean squared error itself. In `FOUR_ARMS.csv` each unit set has its own design-aware baseline: on all held-out units `b + kappa r_plate` with the fold's kappa (`data.fold_data`), on the calibration-supported units origin resolution's, with each case's kappa (`net-cost/AMOUNTS.json`); the two kappas differ by at most 2e-8.

| Table | One row per | Printed in the paper |
|---|---|---|
| `FOUR_ARMS.csv` | fold, model, seed, unit set and arm (original, removal, constant, readers; on the +Var cases also calibrated and calibrated + readers): total error and its common and contrast components (`total_points`, `common_points`, `contrast_points`), in % of the design-aware baseline's total MSE on the same units | through `FOUR_ARMS_SEED_MEANS.csv`, `NINE_METRICS_ABSOLUTE.csv` and `ARM_CHANGES.csv` |
| `FOUR_ARMS_SEED_MEANS.csv` | fold, model, unit set and arm: the seed mean of the total error, in % of the design-aware baseline's total MSE | Supplementary Data 4 (Output_MSE, the original networks of Retrain_original) |
| `NINE_METRICS_VALUES.csv` | +Var case, seed (and seed mean) and prediction: the nine metrics (`mse`: the mean squared error) | through `NINE_METRICS_ABSOLUTE.csv` |
| `NINE_METRICS_GAINS.csv` | +Var case, contrast, metric and seed (and seed mean): gain and 90% line interval, positive when the first prediction is better; `mse` in % of the original's total MSE, the other metrics as differences | Supplementary Data 4 (Tahoe metric intervals) |
| `NINE_METRICS_SUMMARY.csv` | contrast and metric: the cases of 12 whose seed-mean gain is positive | Supplementary Table 21; Supplementary Fig. 5d |
| `NINE_METRICS_ABSOLUTE.csv` | prediction: each metric, the mean over the 12 cases of the seed means; squared error in % of the design-aware baseline's total MSE | Results 7 (104.4%, 95.7%, 96.7%, 93.9%, 99.5%); Fig. 6f |
| `ARM_CHANGES.csv` | +Var case and arm: change of total, common and contrast error from the original, in % of the original's total MSE | Supplementary Fig. 5c |
| `GROWTH_SELECTION.csv` | fold and selected drug x program pair, with whether calibration covers its drug | Supplementary Note S11.1 (20, 42 and 19 pairs: the rows of folds 1-3, all covered by calibration; 26 drugs) |
| `GROWTH_GENE_CORRECTION_SEEDS.csv` | +Var case, seed and arm: the number of calibration-supported units (`n_units`), the seven routine read-outs, total error in % of the design-aware baseline's total MSE, and the pair-median growth relation | Supplementary Note S11.1 (between-predictor spread) |
| `GROWTH_GENE_CORRECTION_CHANGES.csv` | +Var case and seed (and seed mean): change of the calibrated prediction with the growth-gene correction in the read-outs and the growth relation, and in total error, in % of the design-aware baseline's total MSE with its 95% line interval (`total_error_points`, `low`, `high`) and in % of the calibrated prediction's | Results 7; Supplementary Note S11.1; Supplementary Data 4 (Growth_gene_changes) |
| `GROWTH_LINES.csv` | fold, held-out line, plate and series: measured and predicted growth-program responses | Fig. 6d |
| `GROWTH_RANKING.csv` | fold (plate reference, with its 95% interval) and fold, model, predictor and seed: line ranking | through `LINE_RANKING_TABLE.csv` |
| `PULL.csv` | fold, model and seed: pull slope and the line ranking without the pull | through `LINE_RANKING_TABLE.csv` |
| `LINE_RANKING_TABLE.csv` | fold and model: seed means of the pull slope and of the rankings (original, without pull, readers); the plate reference | Supplementary Fig. 5b; Supplementary Data 4 (Pull_ranking, Plate_reference) |
| `RANKING_FOLD_MEANS.csv` | model: the rankings' means over folds | Results 7; Fig. 6e |
| `STORAGE.csv` | fold, model, side and gene set: the storage coefficient | Supplementary Data 4 (Storage) |
| `TRAINING_LENGTH.csv` | model, training length and seed (and seed mean): control-level memory on fold 1's training lines | Supplementary Table 23 |

The two growth-gene correction tables' `systema_pearson_delta_observed_unweighted` and `pearson_delta_unweighted` are the nine metrics' `systema_pearson_delta_observed` and `pearson_delta` read without weights (the context's unweighted mean truth, and the plain mean over units); the nine metrics' `systema_pearson_delta` subtracts the pool mean plus the mean training response instead. `NINE_METRICS_SUMMARY.csv` gives a count in every cell, where Supplementary Table 21 and Supplementary Fig. 5d leave `weighted_r2_contrast` blank (a dash) in the contrasts `readers - original`, `readers - constant`, `calibrated + readers - calibrated`, `removal - original` and `constant - original`: their two predictions share each context's drug differences before clipping, so the count there comes from clipping alone.

### Parse PBMC

The tables under `memorization/parse-pbmc/`, from the Mean-level fits of the five models (summary-statistics, DeepSets and attention models, State, CPA; three seeds each) on donor groups 1-3, each scored on its four held-out donors; all 12 donors are evaluated. Parse PBMC's data and these tables are CC BY-NC 4.0. The definitions are in [`parse.py`](parse.py) and [`parse_interferon.py`](parse_interferon.py): the readers are Tahoe-100M's gene-wise readers with the cell type in place of the plate and the donor in place of the line (no growth or lineage feature); total MSE is in % of the original model's on the same donors, each seed's ratio of sums first, then the seed mean; intervals resample donors (2,000 draws for the 95% intervals, 10,000 paired draws for the nine metrics' 90% intervals) or, for a donor's measured interferon response, its cytokines and reference cells.

The calibrated arm (summary-statistics model and DeepSets) uses one pair of amounts per model and group, picked on the training units of the group's two calibration donors, with their training weights, from the nested parents (+Var fits of 40,000 updates on the group's six nested-fit donors) and applied to the Mean-level fits: the pair (context scale, perturbation scale) on 0-1.5 in steps of 1/8 with the lowest calibration error of the mean of the three seeds' nested-parent predictions, a tie going to the smaller amounts. This differs from the shared calibration (`scsailor.resolution.menu.pick_two`, used on Tahoe-100M), which picks a pair per seed on {0, 1/8, 1/4, 1/3, 1/2, 3/4, 1} with ties to the larger amounts. Each of the three choices is one constant in `parse.py` (`GRID`, `TIES_TO_SMALLER`, `ONE_PAIR_FROM_SEED_MEAN`).

| Table | One row per | Printed in the paper |
|---|---|---|
| `READERS_BY_GROUP.csv` | group, model, arm (readers; for the summary-statistics model and DeepSets also calibrated and calibrated + readers) and seed (and seed mean): total MSE in % of the original's | Abstract, Results 7, Discussion and Supplementary Note S11.4 (14 of 15 model–group comparisons: the readers' seed means below 100; attention, group 1, 102.9%); Supplementary Note S11.4 (all six nested comparisons: calibrated + readers below calibrated) |
| `READERS_POOLED.csv` | model: the readers' total MSE pooled over the 12 donors, in % of the original's, with its 95% donor interval | Results 7 (82.2–96.5%); Fig. 6g; Supplementary Table 27; Supplementary Data 4 (Parse_pooled_MSE) |
| `NINE_METRICS_GAINS.csv` | comparison (readers - original; calibrated + readers - calibrated), model, scope (a group, or `pooled`: the 12 donors), metric and seed (and seed mean): gain, positive when the first is better (`mse` in % of the original's total MSE), and the seed mean's 90% interval | Results 7 and Supplementary Note S11.4 (the pooled gains: positive for all five models on eight metrics, all but `weighted_r2_contrast`; after calibration on seven, also not `systema_pearson_delta_observed`); Supplementary Data 4 (Parse metric intervals) |
| `NINE_METRICS_SUMMARY.csv` | comparison and metric: the model–group cases, how many gain, the median gain and the intervals above or below zero | Supplementary Note S11.4 (six nested comparisons improved) |
| `INTERFERON_RANKING.csv` | series (each model's original, the readers): the Spearman correlation over the 36 held-out contexts of the predicted with the measured interferon-alpha change, centred within group x cell type; seed mean and 95% donor interval | Supplementary Note S11.4 (0.400, 0.604, 0.584, 0.510, 0.287; readers 0.587 [0.241, 0.771]) |
| `INTERFERON_DONORS.csv` | held-out donor and series (measured, each model's original, the readers): the interferon-alpha change, mean over the donor's three cell types, the measured one with its 95% interval; the donor's control level relative to training donors | Supplementary Fig. 5e |
| `INTERFERON_RATIOS.csv` | donor with interferon-high contexts and series (measured, readers): the mean over those contexts of change / control level, the measured one with its 95% interval | Supplementary Table 26; Supplementary Data 4 (Parse_donor_ratios) |
| `INTERFERON_SPLIT_HALVES.csv` | cytokine split: Donor3's ratio and that of Donors 1, 4 and 10 in each half (half 1 against the input wells, half 2 against the reference wells) | Supplementary Note S11.4 (Donor3's ratio above the others' in both halves of all 200 partitions) |
| `CONTROL_WELLS.csv` | interferon program: the Pearson correlation over the 36 held-out contexts of the control level from the input wells (H7-H9) and from the reference wells (H10-H12) | Supplementary Note S11.4 (0.982/0.979) |
| `PROGRAM_SDS.csv` | Hallmark program with at least ten panel genes in every group: the SD of its control level over each group's eight training donors, averaged over cell types and groups | Supplementary Note S11.4 (0.230/0.203; the largest other, 0.071) |

### Genetic screens

The tables under `memorization/genetic-screens/`, from the +Var fits of the five models (summary-statistics, DeepSets and attention models after 40,000 updates, State after 5,000, CPA after 140,000; three seeds each) on Replogle-Nadig folds 1-4 (K562, RPE1, HepG2, Jurkat held out) and Jiang et al. folds 1-6 (A549, BXPC3, HAP1, HT29, K562, MCF7 held out): 50 model–fold cases. The definitions are in [`genetic.py`](genetic.py). The baseline, the denominator of every `mse`, is each model's anchor: the input-set mean plus the average training response for the built models; State and CPA store their own anchor (the same in both), which differs from it in a minority of entries. Contexts are read in storage and pull when they have at least five scored knockdowns. Storage here is the slope of the context-level correction on the whole line offset, with the sampling difference beside it, each context centred over its genes; Tahoe-100M's `STORAGE.csv` is the slope on the unreadable part beside the readable part, centred per plate across contexts. The whole offset is used because the readers have no out-of-line skill on these screens (`reader_r2` below zero), and one held-out line leaves too few contexts to centre across. The nine metrics keep the knockdown's target gene in every metric and give it weight zero only in the weighted R² (`genetic.UNIT_VALUES`); this differs from the shared genetic-screen convention, `scsailor.resolution.metrics.target_excluded_values` (used by origin resolution), which also sets the target gene to the control value in the correlations, the top-50 direction and PDS. The values and changes tables give each seed; the summary reads them as written and takes the seed means.

| Table | One row per | Printed in the paper |
|---|---|---|
| `STORAGE.csv` | screen, fold, model, seed (and seed mean), side (training lines, held-out line) and gene set (growth, other): the slope of the context-level correction on the line offset; the contexts read; the readers' out-of-line R² of the offset | Results 7; Supplementary Note S11.5 (in all 50 cases the seed-mean slope higher on training lines than on the held-out line, both gene sets; training-line medians over folds per model, screen and gene set 0.90–1.02; held-out slopes −1.10 to 0.83; the readers' training-line R² negative on Jiang et al.) |
| `GROWTH_PULL.csv` | screen, fold, model, held-out context and seed (and seed mean): on the growth composite, the context's control growth level, the model's correction and the measured correction; per seed the most-pulled Hallmark program | Supplementary Note S11.5 (RPE1: summary-statistics −0.045, DeepSets −0.011, measured −0.080; growth or MYC pull not recurring on Jiang et al.: a growth or MYC program the most pulled in all three seeds in 1 of 140 model–context cases) |
| `PULL_ACROSS_LINES.csv` | model: on Replogle-Nadig's four held-out lines, the Spearman correlation of the seed-mean growth correction, and of the measured correction, with the control growth level | Supplementary Note S11.5 (−1.0, −1.0, −0.8; measured 0.0; State and CPA +0.8) |
| `NINE_METRICS_VALUES.csv` | screen, fold, model, seed and prediction (original, removal, gene-wise shrink, removal + readers, no change): the nine metrics, `mse` in % of the baseline's total MSE | Supplementary Note S11.5 (the summary-statistics and DeepSets cases on Replogle-Nadig folds 1-3 and Jiang et al. folds 1, 4 and 5: the removal's seed-mean squared error below the original's in 9 of 12, above it for DeepSets on K562 and both models on RPE1, where removal + readers is above it too; the baseline below no change on RPE1, HepG2 and HT29) |
| `NINE_METRICS_CHANGES.csv` | screen, fold, model, seed, comparison (gene-wise shrink - original; removal + readers - removal) and metric: the first prediction's value minus the second's | through `NINE_METRICS_SUMMARY.csv` |
| `NINE_METRICS_SUMMARY.csv` | comparison and metric: the 50 cases, those whose seed-mean change improves (lower squared error, higher otherwise), the median seed-mean change, and the median seed-mean values of the two predictions | Results 7, Discussion and Supplementary Note S11.5 (removal + readers below removal in 0/50); Supplementary Note S11.5 (median MSE 109.2% to 80.1% of the baseline's; median paired change −31.1); Supplementary Table 28; Supplementary Data 4 (Genetic_metrics) |

## Inputs

Under the data root, for Tahoe-100M:

- the fold files of Tahoe-100M folds 1-3 (evaluation units, inputs, training controls and targets, line subsets; `reproduce.origin_resolution.tahoe`) and the cell-line metadata (`tahoe-100m/`);
- `tahoe-100m-ladder/<fold>/<model>_<seed>_L1.npz` (model codes `summary`, `deepsets`, `attention_model`, `cpa`, `prepr`; `L1`: Mean inputs): the Mean-level fits of the summary-statistics, DeepSets and attention models, CPA and PrePR-CT, three seeds each; for fold 1 also `<model>_<seed>_L1_step<N>.npz`, the built models' runs after 80,000 and 120,000 updates (`storage.LENGTH_RUNS`). Every file holds the run's predictions on the training lines and on the held-out lines (every training and held-out unit of the fold). `STORAGE.csv` and `TRAINING_LENGTH.csv` read the built models' training-line predictions too, the other tables only the held-out ones;
- `tahoe-100m-gene-panels/tahoe_<fold>.csv`: each fold's 2,000-gene panel;
- `msigdb-hallmark/h.all.v2025.1.Hs.symbols.gmt`: the MSigDB Hallmark gene sets, release 2025.1.Hs (CC BY 4.0; Liberzon et al. 2015, Cell Systems 1:417-425);
- origin resolution's deposited +Var cases, read through `reproduce.origin_resolution` (`origin-resolution/`, `information-localization-runs-fold-1/` to `-3/`, `state-nested-chains/`, `state-folds-2-3/`, `tahoe-100m-noise-floor/`), and its calibration amounts, `net-cost/AMOUNTS.json`.

For Parse PBMC (CC BY-NC 4.0):

- the groups' files, read through `reproduce.origin_resolution.parse` (`parse-pbmc-held-out-truth/`, `parse-pbmc-inputs/`, `training-arrays-parse-pbmc-tahoe-100m/`, the donor orders in `origin-resolution/parse-pbmc-donor-orders/`) and the nested parents of the summary-statistics model and DeepSets (`origin-resolution/parse/<group>/nested-repair/`), from which the calibration amounts are chosen;
- `parse-pbmc-ladder/<group>/<model>_<seed>_L1.npz` (model codes `summary`, `deepsets`, `attention_model`, `state`, `cpa`): the Mean-level fits, three seeds each; only their held-out predictions are read;
- `parse-pbmc-gene-panels/parse_outer_<group>.csv`: each group's 2,000-gene panel;
- `parse-pbmc-reference-wells/`: the reference wells H10-H12, 256 control cells per donor and cell type that no model received (`UNIT_INDEX.csv`, raw counts in compressed rows: `CELL_INDPTR.npy`, `GENE_INDEX.npy`, `RAW_COUNT.npy`, `RAW_TOTAL.npy`), and `GENE_REGISTRY.csv`, which maps their genes to the panels;
- `msigdb-hallmark/` as above.

For the genetic screens:

- `replogle-nadig-inputs/` (folds 1-4) and `jiang-inputs/` (folds 1-6): per fold `training-arrays/<fold>/train_targets.npz` and `controls.h5` (the training targets and control pools) and `inputs/<fold>/evaluation_inputs.h5`, `training_constants.npz` and `TARGET_GENE_MASK.csv` (input sets and pool means, the average training response, each knockdown's target gene on the panel);
- `genetic-screens-predictions/<screen>/truth/<fold>_HELDOUT_TRUTH.npz` and `genetic-screens-predictions/<screen>/<fold>/<model>_<seed>_L2/PREDICTIONS.npz` (screen codes `replogle`, `jiang`; model codes as above; `L2`: +Var inputs): the fits' predictions on the training lines and the held-out line;
- `replogle-nadig-gene-symbols/hgnc_replogle_panels.tsv`: the HGNC symbol of each Ensembl identifier on the Replogle-Nadig panels (from the HGNC complete set; HGNC data are CC0);
- `msigdb-hallmark/` as above.

## Not included

The tables that need retraining runs: on Tahoe-100M the target interventions, the two-phase retraining, the planted tests and the retrieval of planted training responses; on Parse PBMC the planted interferon carrier (the transferred fractions 0.43, 1.13–1.77 and −0.045 to 0.179 of Results 7) and the input-relative fits of Supplementary Note S11.2.
