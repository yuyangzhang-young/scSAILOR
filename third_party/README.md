# Third-party code

Code from other projects that the paper's published-model results depend on. Each directory holds one upstream release, unchanged, under its own licence; `NOTICE.md` in each directory names the origin, the files and any modification.

| Directory | Upstream | Licence |
|---|---|---|
| `arc-state-9bbfe78/` | State repository (Arc Institute), commit `9bbfe78`: the State model, CPA as implemented there, and the released ST-HVG-Tahoe configuration | CC BY-NC-SA 4.0 for the code; State Model License for the configuration; BSD-3 notices for code shared with theislab/cpa and scvi-tools |
| `prepr-ct-v1.0.0/` | PrePR-CT release v1.0.0 | GPL v3.0 |

Nothing in `src/scsailor` imports these files. Only the scripts in `reproduce/published_models/` load them.
