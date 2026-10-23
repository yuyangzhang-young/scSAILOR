# PrePR-CT, release v1.0.0

Files from PrePR-CT, unchanged.

## Origin

- Repository: https://github.com/reem12345/Cell-Type-Specific-Graphs
- Release `v1.0.0`, commit `b7b5b195a0227418b0893e4316c68430386c2ca8` (2025-04-21); archived as Zenodo record 15255075
- PrePR-CT: Alsulami et al. [complete the reference]

## Files

| File | Use |
|---|---|
| `model.py` | The `MLP` and `GNN` definitions |
| `utils.py` | The `loss_fct` definition |
| `patches/prepr_ct.patch` | The one modification, shown as a diff against `model.py` |

## Licence

GNU General Public License v3.0 (`LICENSE`, copied from release v1.0.0). It covers `model.py`, `utils.py` and `patches/prepr_ct.patch`.

## Modification (GPL v3.0, section 5(a))

First made on 2026-10-05 by the scSAILOR authors: the statement `torch.manual_seed(42)` is removed from `GNN.__init__`, so that the caller's seed sets the initialization. `reproduce/published_models/prepr_ct/adapter.py` makes this change at run time on the definitions it reads from `model.py` and `utils.py` (`MLP`, `GNN`, `loss_fct`); the files in this directory stay unchanged.
