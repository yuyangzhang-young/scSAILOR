# Origin resolution: the paper's results from the deposited data

The stages that rebuild the tables behind origin resolution's results (Results 5 and 6, Figs. 5 and 6, Supplementary Tables 19, 24, 24b and 25b, and Supplementary Data 3) from the models' saved predictions, and compare them with the tables record. The method they call is `scsailor.resolution`; this folder holds the dataset bindings and the stages that read and write the deposited files.

## Environment

Python 3.11 and, from the repository root:

```bash
pip install . numpy==2.4.6 scipy==1.17.1 h5py==3.16.0 pandas==2.3.3 openpyxl==3.1.5
```

These are the versions the tables were computed with; openpyxl is needed only for the Supplementary Data 3 workbook. Some saved tables were written with a set number of BLAS threads, and the runner sets it for those steps. On another CPU the last digits of a value can differ.

## Data

The deposited files sit under one folder, the data root, named by `SCSAILOR_DATA` or `--data`. [`../data/FOLDERS.tsv`](../data/FOLDERS.tsv) lists its folders and what each holds. The data records hold the saved predictions, inputs and run records the stages read; the tables record holds every table the stages write, at the path where they write it. A path stored in a saved file is relative to the data root, and `not-deposited:<name>` names a file that was used but is not deposited.

## Running

One case, the card of the summary-statistics model on Jiang et al.'s fold 4, from its 18 input files (0.62 GB):

```bash
python -m reproduce.origin_resolution.one_case --data DATA --out OUT
```

It prints `equal` or `DIFFERS` for each expected file in `one_case_expected/`.

The steps behind one Results row, compared with the tables record:

```bash
python -m reproduce.origin_resolution.results --row R5P1 --out OUT --data DATA --compare DATA
```

The rows are `R5P1` to `R5P9` and `R6P1` to `R6P5` (the paragraphs of Results 5 and 6), `Fig5`, `Fig6`, `ST19`, `SI` (numbers in the Supplementary Information), `SD3` (every sheet's source, then the workbook), `upstream` (the cards, selections and cost readings the other steps read by path) and `all`. Rows share one `--out`: a step whose files are already there is not run again. `--list` prints every file of every step with its place in the tables record. All 218 steps take about 11 hours, one at a time, on a 14-core CPU.

Each stage also runs on its own: `python -m reproduce.origin_resolution.<stage> --help`.

## Layout

| Path | Contents |
|---|---|
| `__init__.py` | The data root and the readers the bindings share |
| `tahoe.py`, `gse306429.py`, `replogle.py`, `jiang.py`, `parse.py` | Dataset bindings: each screen's case, documented origin rows and evidence |
| `runs.py` | Where information localization's map tables and real-task runs are |
| `selections.py`, `cards.py` | The selections the cards read, and one model's card on one fold |
| `il_cells.py`, `genetic_cells.py` | Recovery and cost of the restoring changes, on Tahoe-100M and on the genetic screens |
| `slopes.py`, `net_cost.py`, `planted.py` | The layers' slopes, the matched comparisons and the recommendation rule, and the known-answer test |
| `cost_readings.py`, `probes.py` | Cost on the screens read on the full chain only, and the fixed-weight probes' tables |
| `baselines.py`, `coverage.py`, `other_metrics.py`, `recovery_readings.py`, `weight_decay.py` | The field's baselines, training coverage, metrics other than squared error, recovery from run records, weight decay |
| `paper_tables.py`, `workbook/` | The paper's tables, and the Supplementary Data 3 workbook's sheet map and README sheet |
| `results.py` | One command per Results row |
| `one_case.py`, `one_case_expected/` | One case and its expected files |
| `inputs/` | The scripts that computed two deposited input tables from the measured cells ([`inputs/README.md`](inputs/README.md)) |

## Inputs made elsewhere

Some inputs were computed from raw cells or trained weights and are deposited as they are: the dataset references (cell sampling, replicate and batch terms), the models' saved predictions, the fixed-weight probes' forward passes, and three tables the rows read directly (`results.INPUTS`: Jiang et al.'s split-half reference and correction alignment, and information localization's recovery by training length). The stages here start from those files. The scripts behind Jiang et al.'s two tables are in `inputs/`.
