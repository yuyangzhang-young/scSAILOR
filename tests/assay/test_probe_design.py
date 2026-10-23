"""A probe design with 4 bags per background, not the 8 used in the paper: every row and bag computation follows it.

With the bag count fixed at 8 anywhere in the split arithmetic, the test rows of a 4-bag design would be
empty and test backgrounds would enter reader fitting or strength selection. No study data:

    python -m pytest tests/assay/test_probe_design.py
"""
import unittest
from types import SimpleNamespace

import numpy as np

from scsailor.assay import probe
from scsailor.assay.constants import N_BITS, N_GENES, N_ROWS
from scsailor.assay.generator import Parameters

DESIGN = probe.ProbeDesign(root=31001, draw_key=20260926, bags_per_context=4, split=(5, 2, 3))
ROWS = sum(DESIGN.split) * DESIGN.bags_per_context


class FourBags(unittest.TestCase):
    def test_row_slices_follow_the_bag_count(self):
        fit_end, tune_end = probe._rows(DESIGN)
        tests = range(ROWS)[probe.test_rows(np.zeros((ROWS, 1)), DESIGN)]
        self.assertEqual((fit_end, tune_end - fit_end, len(tests)), tuple(4 * n for n in DESIGN.split))
        self.assertEqual(tests[0], tune_end)

    def test_test_backgrounds_enter_neither_fitting_nor_strength_selection(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(ROWS, 12))
        states = rng.choice([-1, 1], size=(ROWS, N_BITS))
        tests = slice(ROWS - DESIGN.split[2] * DESIGN.bags_per_context, ROWS)   # the last 3 backgrounds' 12 bags
        clean = probe.fit_readers(features, states, "L2", DESIGN)
        features[tests], states[tests] = np.nan, 0   # anything a test row contributed would show
        poisoned = probe.fit_readers(features, states, "L2", DESIGN)
        for field in ("mean", "sd", "center", "weights", "alpha"):
            self.assertTrue(np.array_equal(getattr(clean, field), getattr(poisoned, field)), field)
            self.assertTrue(np.isfinite(getattr(poisoned, field)).all(), field)

    def test_probe_bags_take_four_per_background(self):
        rng = np.random.default_rng(1)
        pools = {f"c{i}": (rng.gamma(2., .5, size=(300, N_GENES)), rng.integers(1000, 20000, size=300))
                 for i in range(sum(DESIGN.split))}
        instance = SimpleNamespace(train_contexts=list(pools), params=Parameters(),
                                   pool=lambda group, context: pools[context])
        bags = probe.probe_bags(instance, "L1", DESIGN)
        self.assertEqual([(bag.context_index, bag.rep) for bag in bags],
                         [(c, r) for c in range(sum(DESIGN.split)) for r in range(4)])
        self.assertEqual(bags[0].x.shape, (N_ROWS, N_GENES))


if __name__ == "__main__":
    unittest.main()
