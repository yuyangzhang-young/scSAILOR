# Published models: State, CPA and PrePR-CT

Training, prediction and scoring of the three published models of the paper, on the natural task (measured responses) and on the synthetic-answer assay.

| Model | Upstream code (`third_party/`) | Fixed training length |
|---|---|---|
| State | `arc-state-9bbfe78/` | 5,000 updates |
| CPA, as implemented in the State repository | `arc-state-9bbfe78/` (with `patches/cpa.patch`) | 140,000 updates |
| PrePR-CT | `prepr-ct-v1.0.0/` (with `patches/prepr_ct.patch`) | 10,000 updates |

Every model is trained for its fixed length at each information level (Mean `L1`, +Var `L2`, +Dist `L3`, +DepCp `L3d`, Cells `L4`) and seed (31002, 31003, 31004), and read out there.

## Layout

| Path | Contents |
|---|---|
| `run.py` | Entry point: train one model at one level and seed, then predict (natural task) or replay and score (synthetic task) |
| `common/runtime.py` | One numerical thread and full-precision matrix products, as in the paper's runs |
| `common/fit.py` | The training loop, with exact resume |
| `common/natural.py` | Natural task: training units, the stream that pairs control and treated cells, and prediction |
| `common/synthetic.py` | Synthetic task: the training stream, replay and scoring (including scoring on a fixed subset of perturbations) |
| `state/adapter.py`, `cpa/adapter.py`, `prepr_ct/adapter.py` | What is specific to each model: how it is built and loaded, its optimizer, one training update, and its prediction. Each file's docstring lists the changes made for this task |
| `prepr_ct/graph.py` | The gene graph of one input set (PrePR-CT's co-expression graph, in our own code) |

## Environment

Python 3.11 and, in addition to `pip install ".[torch]"` from the repository root:

```bash
pip install numpy==2.1.2 torch==2.8.0 h5py==3.16.0 scipy==1.17.1 pyyaml==6.0.3 lightning==2.6.6 torchmetrics==1.9.0 transformers==5.20.2 geomloss==0.3.1 POT==0.9.4 torch_geometric==2.6.1
```

These are the versions the code was checked with. The paper's runs used NumPy 2.1.2 and PyTorch 2.8.0 on NVIDIA H200 (State) and V100 (CPA, PrePR-CT) GPUs. Scores computed with another NumPy version can differ in the last digits.

## Data

A natural-task fold is one directory with the files listed in [`docs/data_contract.md`](../../docs/data_contract.md#published-models) (training targets, control pools, treated cells and their index, evaluation bags, held-out prediction menu; for PrePR-CT also `drug_features.npz`). A synthetic-task instance is the assay's instance directory; Replogle–Nadig's adds `EVALUATION_SUBSET.json`.

## Running

From the repository root, for example State at +Var with seed 31002 on Tahoe-100M fold 1:

```bash
python -m reproduce.published_models.run --model state natural --fold DATA/tahoe-100m/F1 --level L2 --seed 31002 --output runs/state/tahoe-100m-F1/L2-31002
```

```bash
python -m reproduce.published_models.run --model state synthetic --instance DATA/tahoe-100m/F1/synthetic --level L2 --seed 31002 --output runs/state/tahoe-100m-F1-synthetic/L2-31002
```

PrePR-CT's synthetic runs also take `--features` (the fold's `drug_features.npz`). `--device` defaults to `cuda:0`; `--workers` (default 4) sets the threads that prepare batches ahead and does not change the results. An interrupted run continues from its last resume point when started again with the same arguments.

## Outputs

| File | Contents |
|---|---|
| `TRAIN_CURVE.jsonl` | Training loss every 100 updates |
| `step{N}.pt` | The model at the read-out step |
| `latest.pt` | The resume point |
| `PREDICTIONS_step{N}.npz` (natural) | Per side (`train`, `heldout`): raw and clipped predicted profiles, contexts, perturbation codes, evaluation draws |
| `PREDICTION_INPUT_IDS.npz` (natural) | The evaluation bag, cells and seed key of every draw |
| `PREDICTIONS_step{N}.h5` (synthetic) | Replayed predictions in the assay's layout |
| `SCORES.json` (synthetic) | Recovery of every planted state, per side |

## Reproducibility

The code is a consolidation of the scripts that produced the paper's results. In short checks on CPU, on all three models and both tasks, it gives bit for bit the same output as those scripts: the same model, optimizer state, training loss, predictions, replays and scores. Training on a GPU is not bit-for-bit repeatable, because some GPU operations accumulate in a varying order, so a repeated GPU run does not reproduce the paper's fits exactly.
