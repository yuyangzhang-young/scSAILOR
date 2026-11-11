# Inputs computed from the measured cells

Three scripts computed two of the deposited tables that the Results rows read directly (`results.INPUTS`). They ran on a compute cluster, on CPU, from the measured cells and the models' saved natural-task fits; the stages in the folder above do not run them.

| Script | Deposited output |
|---|---|
| `jiang_common_split_half.py` | `jiang-split-half-reference/JIANG_SPLIT_HALF_POOLED.csv`: whether the held-out line's context-level correction in Jiang et al. is reproducible between halves of its cells |
| `diagnose_fold.py`, with the scoring functions of `score_fold.py` | `jiang-correction-alignment/jiang/<fold>/OFFSET_READINGS.csv`, folds F1-F6: the size and alignment of each fit's context-level correction against the truth's |

`reliability/` holds the unit-reliability tables that `diagnose_fold.py` reads, one per screen and fold (signal, noise and whether a held-out unit is reliable; computed from the held-out truth only).

The folders the scripts read are command-line arguments; each script's docstring says what they must hold:

```bash
python jiang_common_split_half.py BANK COUNTS TRUTH OUT
python diagnose_fold.py jiang F1 BANK NATURAL STATE CPA OUT
```

`BANK` is the dataset's natural-task bank (training arrays, model inputs and the cells' quality-control table), `COUNTS` Jiang et al.'s raw counts per stimulus, `TRUTH` and `NATURAL` the held-out truth and the built models' fits, `STATE` and `CPA` the published models' fits.

The scripts are the ones that produced the deposited tables, with two changes: the folders, which were fixed paths of the cluster, are now arguments, and the docstrings and comments use the paper's terms. The computation is unchanged (the syntax trees outside the paths and the command line are the same), but the changed scripts have not been run again on the cluster.
