# State repository, commit 9bbfe78

Files from the State repository by Arc Institute, used for the official State model and for CPA as implemented in that repository. The files are unchanged.

## Origin

- Repository: https://github.com/ArcInstitute/state
- Commit: `9bbfe78a434a55205e4de834e1ea99f85f7a3add` (2026-07-23), package `arc-state` 0.11.3
- State: Adduri AK, et al. Predicting cellular responses to perturbation across diverse contexts with State. *Cell* 189(19), 5914–5931.e20 (2026). https://doi.org/10.1016/j.cell.2026.07.052
- CPA: Lotfollahi M, et al. Predicting cellular responses to complex perturbations in high-throughput screens. *Molecular Systems Biology* 19(6), e11517 (2023). https://doi.org/10.15252/msb.202211517

## Files

| File | Use |
|---|---|
| `src/state/tx/models/state_transition.py`, `base.py`, `utils.py` | The State transition model |
| `src/state/tx/models/cpa/*.py` (7 files), `base.py`, `utils.py` | CPA as implemented in the State repository |
| `src/state/configs/model/cpa.yaml`, `src/state/configs/training/cpa.yaml` | CPA hyperparameters |
| `official_config.yaml` | Hyperparameters of the released ST-HVG-Tahoe model: Hugging Face `arcinstitute/ST-HVG-Tahoe`, `zeroshot/state_generalization_zeroshot_X_hvg/config.yaml` |
| `patches/cpa.patch` | Two one-line changes to `cpa/_module.py` and `cpa/_model.py` for a model with a single batch: the batch term is 0 when no batch embedding exists, and batch-embedding parameters are left out of the optimizer. `reproduce/published_models/cpa/adapter.py` applies them to a temporary copy at run time. |

No released State weights or embeddings are used: every model in this study was trained from random initialization.

## Licences

- `LICENSE`: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International, the licence of the State source code, copied from the commit above. It covers the files under `src/` and `patches/cpa.patch`.
- `MODEL_LICENSE.md`: Arc Research Institute State Model Non-Commercial License, copied from the same commit. It covers `official_config.yaml`, which Arc released with the model.
- `LICENSE-CPA-theislab-v0.5.0`, `LICENSE-CPA-theislab-current`: BSD 3-Clause notices of theislab/cpa. State's CPA files share code with theislab/cpa (for example `FocalLoss`, `GeneralizedSigmoid`, the CPA module loss and `adv_lambda`); the State repository carries no theislab notice, and which theislab release the code derives from is not established, so both notices are included.
- `LICENSE-scvi-tools`: BSD 3-Clause notice of scvi-tools (current text). State's `cpa/_base_modules.py` and `cpa/_dists.py` say parts are inspired by scvi-tools source code.
