"""Synthetic-answer assay: which statistics of the control cells does a perturbation model read?

Measured control cells carry 25 planted states, each written into one cue (shift, spread, skew, depth
trend or covariation) of two genes. The synthetic target moves with the planted states through a
response field calibrated on measured responses. A model trained on these inputs at one information
level is scored on how much of each planted state's effect it recovers.

Steps, one module each:

1. ``generator``   the synthetic target (calibration, shifts, exact clipped means, evaluation support);
2. ``carrier``     the model input (planted states, information levels);
3. ``stream``      the ordered training batches;
4. ``train``       the training loop;
5. ``replay``      predictions of a trained model on the evaluation support;
6. ``score``       risk and recovery Q of each planted state;
7. ``probe``       linear readers on internal features, and local flip pairs;
8. ``capability``  the seed-pooled table of what each model reads.

``design`` describes the planted states and levels as tables, ``constants`` holds the shared constants,
``contract`` states what the assay calls on a model, and ``instance`` reads an instance from disk.

Every module takes its settings as arguments; the settings used in the paper (instances per dataset and
fold, training recipe, replay draws and scopes, probe designs) are in ``reproduce/study.py``. Importing a
module has no side effects, and torch is imported only inside the functions that run a model.

Code names and the paper's terms (saved files keep the code names):

==============================  ============================================================
Code                            Paper
==============================  ============================================================
bit                             planted state (25 = 5 cues x 5 responses)
known answer                    synthetic target
levels L1, L2, L3, L3d, L4      Mean, +Var, +Dist, +DepCp, Cells (L0: sequencing depth only)
model ``state``                 the attention model (built for this study; not the published State)
model ``deepsets``, ``summary`` the DeepSets model, the summary-statistics model
``anchor``                      the model's baseline: the input set's mean (L0: the training
                                controls' mean) plus the perturbation's mean offset
Q, ``recovery``                 recovery, 1 - M/E
==============================  ============================================================
"""
from .instance import Instance

__all__ = ["Instance", "capability", "carrier", "constants", "contract", "design", "generator", "probe", "replay",
           "score", "stream", "train"]
