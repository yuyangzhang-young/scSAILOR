# Memorization: Results 7 on Tahoe-100M

The tables of the paper's Results 7 on Tahoe-100M folds 1-3 (Fig. 6d–f), computed on a CPU from the models' saved predictions: how far the models store the training lines' responses and control levels, how this pulls their predictions for held-out lines toward the training lines' control-pool growth level, and what readers of a held-out context's own control cells give in place of a model's context-level correction. The method parts these tables read are `scsailor.resolution` (calibration, metrics, line bootstrap) and origin resolution's deposited cases (`reproduce.origin_resolution`). The code names and the paper's terms are listed in [`__init__.py`](__init__.py).

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
| `tables.py` | Writes every table |

## Environment

Python 3.11 and, from the repository root:

```bash
pip install . numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 pandas==2.3.3 pyarrow==25.0.0
```

These are the versions the tables were computed with; pyarrow reads the cell-line metadata. On another CPU the last digits of a value can differ (see `--compare` below).

## Tables

```bash
python -m reproduce.memorization.tables --data DATA --out OUT --compare DATA
```

`DATA` is the data root ([`../data/FOLDERS.tsv`](../data/FOLDERS.tsv)); the deposited tables are under `DATA/memorization/tahoe-100m/`. The tables are written under `OUT/memorization/tahoe-100m/` in the order in which they depend on each other; a table computed from others reads them as written, at six significant digits, as the paper's values were. `--compare DATA` prints `equal`, `DIFFERS` or `not deposited` for each table against the deposited one at the same path, cell for cell and exactly; on another CPU a `DIFFERS` can come from last digits alone, so compare such a table numerically before reading it as a difference. The command sets four BLAS threads, as the paper's tables were computed; it takes about 9 minutes on a CPU and 8 GB of memory.

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

## Inputs

Under the data root:

- the fold files of Tahoe-100M folds 1-3 (evaluation units, inputs, training controls and targets, line subsets; `reproduce.origin_resolution.tahoe`) and the cell-line metadata (`tahoe-100m/`);
- `tahoe-100m-ladder/<fold>/<model>_<seed>_L1.npz` (model codes `summary`, `deepsets`, `attention_model`, `cpa`, `prepr`; `L1`: Mean inputs): the Mean-level fits of the summary-statistics, DeepSets and attention models, CPA and PrePR-CT, three seeds each; for fold 1 also `<model>_<seed>_L1_step<N>.npz`, the built models' runs after 80,000 and 120,000 updates (`storage.LENGTH_RUNS`). Every file holds the run's predictions on the training lines and on the held-out lines (every training and held-out unit of the fold). `STORAGE.csv` and `TRAINING_LENGTH.csv` read the built models' training-line predictions too, the other tables only the held-out ones;
- `tahoe-100m-gene-panels/tahoe_<fold>.csv`: each fold's 2,000-gene panel;
- `msigdb-hallmark/h.all.v2025.1.Hs.symbols.gmt`: the MSigDB Hallmark gene sets, release 2025.1.Hs (CC BY 4.0; Liberzon et al. 2015, Cell Systems 1:417-425);
- origin resolution's deposited +Var cases, read through `reproduce.origin_resolution` (`origin-resolution/`, `information-localization-runs-fold-1/` to `-3/`, `state-nested-chains/`, `state-folds-2-3/`, `tahoe-100m-noise-floor/`), and its calibration amounts, `net-cost/AMOUNTS.json`.

## Not included

The same readings on Parse PBMC and the genetic screens are not included here. Neither are the tables that need retraining runs: the target interventions, the two-phase retraining, the planted tests and the retrieval of planted training responses.
