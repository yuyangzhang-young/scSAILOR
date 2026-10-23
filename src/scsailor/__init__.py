"""scSAILOR: diagnose which information from control cells a perturbation-response model reads.

Modules:

- ``carrier``: the information ladder; input sets that keep exactly one level of the control cells'
  statistics (Mean, +Var, +Dist, +DepCp, Cells);
- ``models``: the three built models (attention, DeepSets, summary statistics);
- ``natural``: training on measured responses and scoring held-out predictions;
- ``assay``: the synthetic-answer assay, which plants known states into control cells and measures how
  much of each a trained model recovers.
"""
__version__ = "0.2.0.dev0"
