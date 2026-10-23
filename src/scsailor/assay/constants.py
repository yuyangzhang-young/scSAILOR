"""Constants shared by the assay's modules, each defined once with its reason.

The saved instances, checkpoints and predictions were made with these values; changing one changes
the assay.
"""
N_GENES = 2000                              # width of the gene panel of every instance
N_BITS = 25                                 # planted state bits: 5 statistics x 5 roles (``design``)
FIRST_SCORED_GENE = 2 * N_BITS              # genes 0-49 carry the bits, two per bit; they are neither trained on nor scored
SCORED = slice(FIRST_SCORED_GENE, N_GENES)
N_SCORED = N_GENES - FIRST_SCORED_GENE      # 1950 scored genes
N_CELLS = 256                               # cells per bag: evaluation, training and probe bags
N_ROWS = 512                                # model input rows: each of the 256 cells (or atoms) is written twice

# Random keys. Every random draw is keyed by a SeedSequence entropy list; these values end the lists
# named next to them, so that two kinds of draw never share a key.
BASE_STATE_KEY = 777    # [root, context_index, base, 777]: the evaluation base states (the stored states used this key)
REPLAY_DRAW_KEY = 888   # [root, context_index, base, draw, 888]: replay's technical draws; the configuration is not in the key, so a flip pair shares every draw
CELLS_KEY = 1           # [seed, step, slot, 1] in training, [root, context_index, rep, 1] in the probe: which cells form a bag
STATE_KEY = 3           # [seed, step, slot, 3], [root, context_index, rep, 3]: the bag's random 25-bit state
CARRIER_KEY = 4         # [seed, step, slot, 4]: the training carrier's technical draws
# A training step draws its backgrounds and perturbations from [seed, step, 0]. Key 2 is skipped: it is the
# natural task's carrier key (scsailor.natural.data).

__all__ = ["N_GENES", "N_BITS", "FIRST_SCORED_GENE", "SCORED", "N_SCORED", "N_CELLS", "N_ROWS", "BASE_STATE_KEY",
           "REPLAY_DRAW_KEY", "CELLS_KEY", "STATE_KEY", "CARRIER_KEY"]
