"""Method constants shared by several modules, each defined once with its reason.

The saved parents and runs were made with these values; changing one changes the method. Constants used by one
module live in that module; the synthetic-answer task's own constants (genes, planted states, cells per bag) are
``scsailor.assay.constants``.
"""
LEVEL = "L2"                # input level of every parent and run
BITS = (0, 1, 5, 6)         # the scored planted states: shared shift, selective shift, shared spread, selective spread
TARGET_BIT = 1              # the raw error is read on the target's endpoints

BAGS_PER_NATIVE_BATCH = 8   # one ``stream.batch(index, ...)`` of a training stream gives eight bags

# The training recipe of every start on the built models (``starts`` and ``real_task``).
ADAMW = {"betas": (.9, .999), "eps": 1e-8, "weight_decay": 0}   # AdamW without weight decay
GRADIENT_CLIP = 1.          # clipped total gradient norm
CURVE_EVERY = 100           # a training-curve row at update 1 and every 100th update

__all__ = ["LEVEL", "BITS", "TARGET_BIT", "BAGS_PER_NATIVE_BATCH", "ADAMW", "GRADIENT_CLIP", "CURVE_EVERY"]
