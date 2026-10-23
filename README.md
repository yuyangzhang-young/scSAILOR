# scSAILOR

**Which information from control cells does a perturbation-response model read?**

Models that predict how cells respond to a perturbation usually receive a set of control cells as input. scSAILOR asks which statistics of that set a trained model actually uses, and whether using them helps it predict. It combines an *information ladder* of inputs that keep progressively more of the control cells' statistics with three components: the **synthetic-answer assay**, **information localization** and **origin resolution**.

> **Status.** This is the first part of the public release that accompanies [paper reference to be added]. It contains the information ladder, the three models built for the study, the natural-task training stream and scores, the synthetic-answer assay, and the code that trains, predicts and scores the three published models. Information localization, origin resolution and the remaining reproduction scripts will follow.

## Contents

| Path | Contents | Licence |
|---|---|---|
| `src/scsailor/carrier.py` | The information ladder: input sets that keep exactly one level of the control cells' statistics | Apache-2.0 |
| `src/scsailor/models/` | The three built models: attention, DeepSets, summary statistics | Apache-2.0 |
| `src/scsailor/natural/` | The natural task: training stream on measured responses, and held-out scores | Apache-2.0 |
| `src/scsailor/assay/` | The synthetic-answer assay | Apache-2.0 |
| `tests/` | Tests that need no study data | Apache-2.0 |
| `docs/data_contract.md` | The files a dataset must provide | Apache-2.0 |
| `reproduce/published_models/` | Training, prediction and scoring of the published models State, CPA and PrePR-CT on both tasks | Apache-2.0 |
| `third_party/` | Unchanged upstream code of State (with its CPA implementation) and PrePR-CT, used to reproduce the published-model results | Each directory's own licence |

## Installation

Python 3.11 or later. To install the package with PyTorch:

```bash
pip install "scsailor[torch] @ git+https://github.com/yuyangzhang-young/scSAILOR"
```

The information ladder, the assay's data handling and the scores need only NumPy, h5py and SciPy; PyTorch is needed to build and train models. To work on the code, clone the repository, install it in editable mode with the test extra, and run the tests:

```bash
pip install -e ".[torch,test]"
```

```bash
pytest
```

## The information ladder

From the control cells of one context (log-normalized expression and sequencing depth), `scsailor.carrier.make` builds the 512 rows a model receives. Each level keeps exactly the statistics named below and nothing more:

| Level (code key) | The rows keep |
|---|---|
| Mean (`L1`) | each gene's mean |
| +Var (`L2`) | each gene's mean and sample variance |
| +Dist (`L3`) | each gene's marginal distribution |
| +DepCp (`L3d`) | each gene's marginal distribution within sequencing-depth strata |
| Cells (`L4`) | the cells themselves |

```python
import numpy as np
from scsailor.carrier import make, check_carriers

rng = np.random.default_rng(0)
cells = np.log1p(rng.gamma(0.5, 4.0, size=(256, 2000)))   # 256 control cells x 2,000 genes
depth = np.sort(rng.integers(1_000, 20_000, size=256))     # sequencing depth of each cell

rows, row_depth = make(cells, depth, "L2", rng)            # 512 x 2,000 rows at +Var
check_carriers(cells, depth)                               # verifies the moments and multisets
```

## Built models

The study's three models share one form: the prediction is a baseline (the input set's mean plus the perturbation's mean offset) plus a correction computed by the model from the rows, their sequencing depths and a learned perturbation embedding.

| Model | Code key | How it combines cells |
|---|---|---|
| Attention model | `state` | Self-attention across the cells (8 blocks, width 768, 12 heads, after State's published sizes; it is not the published State model) |
| DeepSets model | `deepsets` | Encodes each cell, then averages |
| Summary-statistics model | `summary` | Reads means, variances, sorted marginals, depth-stratified marginals and a learned pairwise term, each from its first level on |

```python
import torch
from scsailor.models import RawModel, set_runtime

set_runtime(31002)                                         # seeds and full-precision matrix products
model = RawModel("deepsets", intercept=np.zeros(2000, np.float32), p_count=4,
                 r_input=np.zeros((4, 2000), np.float32), c_train=cells.mean(0).astype(np.float32))
with torch.no_grad():
    profile = model(torch.from_numpy(rows)[None], torch.from_numpy(row_depth)[None],
                    torch.tensor([2]), "L2")               # 1 x 2,000 predicted profile
```

Held-out predictions are scored by `scsailor.natural.score.risk`, which splits the mean squared error into a common error (shared by all perturbations of a context) and a contrast error (the rest).

## The synthetic-answer assay

The assay plants 25 known states into measured control cells. Each planted state changes one cue (shift, spread, skew, depth trend or covariation) of two genes, and a response field calibrated on measured responses turns the states into a synthetic target. A model trained on these inputs at one level of the ladder is then scored on how much of each planted state's effect it recovers (recovery Q = 1 − M/E).

`scsailor.assay` holds one module per step: `generator` (the synthetic target), `carrier` (the model input), `stream` and `train` (training), `replay` (predictions on the evaluation support), `score` (recovery), `probe` (linear readers on internal features) and `capability` (the table of what each model reads). An instance is a directory of five files, described in [`docs/data_contract.md`](docs/data_contract.md). [`tests/assay/test_tiny_instance.py`](tests/assay/test_tiny_instance.py) builds a tiny instance from random arrays and runs it end to end.

Code and saved files keep short names; the paper's terms are listed in [`src/scsailor/assay/__init__.py`](src/scsailor/assay/__init__.py) (for example, a *bit* is a planted state and the *known answer* is the synthetic target).

## Third-party code

The published models of the paper (State, CPA and PrePR-CT) were adapted from their authors' code. `third_party/` holds the upstream files that this needs, unchanged and pinned to the versions used: State repository commit `9bbfe78` (which also contains the CPA implementation used) and PrePR-CT release `v1.0.0`. Each directory carries its upstream licence and a `NOTICE.md` with the origin and the changes made: none for State, two one-line changes for CPA (`patches/cpa.patch`) and one removed seed statement for PrePR-CT (`patches/prepr_ct.patch`), each applied at run time to a copy. Only `reproduce/published_models/` loads these files; nothing under `src/` imports them. How to run the published models is described in [`reproduce/published_models/README.md`](reproduce/published_models/README.md).

## Licence

The code written for scSAILOR (`src/`, `tests/`, `docs/`, `reproduce/`) is released under the [Apache License 2.0](LICENSE). The files under `third_party/` are **not** covered by it: they keep their upstream licences, which include the non-commercial terms of State's licences (CC BY-NC-SA 4.0 and the Arc Research Institute State Model Non-Commercial License) and GPL v3.0 for PrePR-CT. See [`third_party/README.md`](third_party/README.md).

## Citation

[Paper reference to be added.]
