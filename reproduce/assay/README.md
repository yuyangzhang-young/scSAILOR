# Synthetic-answer assay: the paper's settings

`study.py` holds the settings with which the paper trained, replayed and scored the three built models in the synthetic-answer assay on Tahoe-100M (folds F1 to F3) and Parse PBMC (groups G1 to G3), and builds the models from `scsailor.models`. The method is `scsailor.assay`, whose modules take these settings as arguments. The attainable references that `scsailor.assay.capability.judge` sets the recoveries against are not part of these settings.

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
