# Running scSAILOR on your own data and model

[`quickstart.py`](quickstart.py) runs the three components on a small screen and a small network, on a CPU, in a few minutes:

```bash
python examples/quickstart.py --out quickstart-run
```

It needs the package with PyTorch (`pip install -e ".[torch]"`). Random arrays stand in for the data and the predictions, so the script runs as it is. To use your own, replace the three stand-ins below. The budgets are a few hundred updates, so the network learns little and its recoveries stay near zero. The run shows what each step takes and returns, not what a trained model reads.

## What you provide

| Replace | With | Used by |
|---|---|---|
| `your_data` | the control cells of every context (cells × 2,000 genes, log1p(CP10k), with each cell's sequencing depth; at least 256 cells per context), split into training and held-out contexts; the measured mean treated profile of every training unit (a context under one perturbation), with its weight | the synthetic-answer assay and information localization |
| `YourModel` | your network, as a torch module with the calls below | the assay and localization |
| `your_predictions` | your model's predictions on measured responses, per seed, on calibration lines (lines the original model was not trained on) and on held-out lines (disjoint from them), with the change to be judged | origin resolution |

The planted states overwrite genes 0–49 of every control cell, and scoring starts at gene 50. Put genes you do not need to score first. The file layout the assay reads is in [`docs/data_contract.md`](../docs/data_contract.md); `write_instance` writes it from your arrays. The assay trains a fresh network of your architecture: `build_model` gives it the instance's response field as `r_pool` (one row per perturbation) and the training contexts' control mean as `c_train`.

**What the package calls on your model** (`scsailor.assay.contract`):

- `forward(x, depth, p, level, canonicalize=True)`: rows (bags × 512 × 2,000), their depths and perturbation indices give a raw prediction (bags × 2,000).
- `canonical_rows`: the rows sorted by depth.
- `anchor`: the rows' mean plus the perturbation's mean offset `r_pool[p]`.
- The buffer `r_pool`.
- For localization, the perturbation embedding `backbone.pert`.
- For localization, the cuts, as parameter-name prefixes of the front and of the part behind each cut. `YourModel.CUTS` gives them, added beside the built models' in `scsailor.localization.contract.CUTS`.

The front is frozen by its parameters. If your front also reads `backbone.pert`, a starting point that trains the embedding changes the front's input too. The built models handle this with a separate copy of the embedding for the part behind the cut (`contract.drug_embedding_copy`); a model of your own needs the same. The three built models in `scsailor.models` already meet this contract.

## What each step does

1. **The synthetic-answer assay** builds an instance from your data. The response field of the synthetic target is calibrated on your measured responses. The step then trains the model at +Var (`L2`), replays it on the fixed evaluation bags and prints each planted state's recovery Q on the held-out contexts. Outputs go to `assay/`: the training run, `predictions.h5` and `scores/SCORES.json`.
2. **Information localization** trains the part behind one cut of the assay's model from two starting points, fine-tuning (`a`) and a front reset (`e`). It prints the shared and selective shift and spread recoveries before and after. Each starting point writes its training curve and checkpoint to `localization/<start>/`.
3. **Origin resolution** applies the recommendation rule to two changes of a model's predictions. The kept fraction and every calibration are chosen on calibration lines. A change is recommended when, in every seed, some of it is kept and its error is lower than every reference of the comparison by more than 0.1% of the original model's total MSE. The three comparisons are standard, calibrated and design-aware. Held-out lines report the same gaps and take no part in the choice. The script's change toward the truth overshoots it twofold, so the rule keeps about half of it and recommends it. The change at random is refused.

## From the quickstart to a full run

The paper's Methods give the settings of its runs. A full run differs from the quickstart in these ways:

- Each model's training length.
- The number of seeds.
- The replay draws: 16 per bag from +Var up, against 1 here.
- In localization, the full map: every starting point at every cut, at the rate multiples `scsailor.localization.readings.GRID`, run by `starts.run`, which fixes the length of a run. The quickstart calls `starts.build_arm` and `starts.train` directly so that it can run a few hundred updates. The reading rules in `scsailor.localization.readings` read the map.
- In origin resolution, the model's fixed intervention, calibrated the same way, as an extra comparator in every comparison (`net_cost.own_control_gap`). Recovery and cost, and attribution, are the other modules of `scsailor.resolution`.
